use crate::{
    common::{build_runtime, consume_and_log_err, TOKIO_RUNTIME},
    do_in,
    error::ServiceResult,
};
use bytes::{Buf, BufMut, BytesMut};
use chrono::{DateTime, Duration, Utc};
use core::fmt;
use cron;
use futures::{future::join_all, TryFutureExt};
use indexmap::IndexMap;
use lru::LruCache;
use pyo3::{
    exceptions::PyTypeError,
    pyclass, pymethods,
    types::{PyByteArray, PyFunction, PyNone, PyString},
    IntoPy, Py, PyAny, PyResult, Python,
};
use serde::Serialize;
use std::{
    collections::VecDeque,
    fs,
    future::{self, Future},
    io::Read,
    num::NonZeroUsize,
    path::PathBuf,
    pin::Pin,
    str::FromStr,
    sync::Arc,
};
use strum::Display;
use thiserror::Error;
use tokio::{
    sync::{
        mpsc::{
            self,
            error::{SendError, TrySendError},
        },
        oneshot, OnceCell,
    },
    task::{JoinError, JoinHandle},
};
use tracing::{error, info, info_span, instrument, warn};

const DEFAULT_CACHE_SIZE: usize = 128;

#[derive(Clone, Debug)]
pub enum Schedule {
    Cron(cron::Schedule),
    Interval(Duration),
}

impl Schedule {
    fn next_update(&self, last_updated: &Option<DateTime<Utc>>) -> UpdateTime {
        let next_update: UpdateTime = match (&self, last_updated) {
            (Schedule::Cron(schedule), _) => UpdateTime::DateTime(
                *schedule.upcoming(Utc).take(1).collect::<Vec<_>>().first().unwrap(),
            ),
            (Schedule::Interval(max_wait), Some(dt_last_updated)) => {
                UpdateTime::DateTime(*dt_last_updated + *max_wait)
            }
            (Schedule::Interval(_), None) => UpdateTime::Now,
        };
        next_update.into()
    }
}

impl Default for Schedule {
    fn default() -> Self {
        Schedule::Cron(cron::Schedule::from_str("0 0 0 * * *").unwrap())
    }
}

impl From<cron::Schedule> for Schedule {
    fn from(value: cron::Schedule) -> Self {
        Schedule::Cron(value)
    }
}
impl From<u32> for Schedule {
    fn from(seconds: u32) -> Self {
        Schedule::Interval(Duration::seconds(seconds as i64))
    }
}

#[derive(Clone)]
enum UpdateTime {
    DateTime(DateTime<Utc>),
    Now,
}

impl From<DateTime<Utc>> for UpdateTime {
    fn from(dt: DateTime<Utc>) -> Self {
        Self::DateTime(dt)
    }
}

fn __str__(obj: &Py<PyAny>) -> String {
    match Python::with_gil(|py| obj.call_method0(py, "__str__")?.extract::<String>(py)) {
        Ok(model_str) => model_str,
        Err(_) => "<Failure calling `__str__` for Python object>".into(),
    }
}

pub trait ArtifactSpec {
    fn spec(&self) -> Box<&dyn erased_serde::Serialize>;

    fn as_map(&self) -> Result<IndexMap<String, String>, serde_json::Error> {
        let mut writer = BytesMut::new().writer();
        serde_json::to_writer(&mut writer, self.spec().as_ref())?;
        serde_json::from_reader(writer.into_inner().reader())
    }

    fn as_key(&self) -> Result<CacheKey, serde_json::Error> {
        let parts = self.as_map()?.into_iter().map(|(k, v)| [k, v].join("="));
        let key = parts.collect::<Vec<_>>().join("/");
        Ok(key)
    }
}

impl Serialize for dyn ArtifactSpec {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.spec().as_ref().serialize(serializer)
    }
}

impl std::fmt::Display for dyn ArtifactSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let map = self.as_map().map_err(|_| fmt::Error::default())?;
        let display = map.into_iter().map(|(k, v)| [k, v].join("=")).collect::<Vec<_>>().join(", ");
        f.write_str(display.as_str())
    }
}

impl std::fmt::Debug for dyn ArtifactSpec + Send + Sync {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!(
            "<impl ArtifactSpec, fields: {{ {:#?} }}>",
            serde_json::to_string(self.spec().as_ref())
        ))
    }
}

impl<T: Serialize> ArtifactSpec for T {
    fn spec(&self) -> Box<&dyn erased_serde::Serialize> {
        Box::new(self as &dyn erased_serde::Serialize)
    }
}

pub type BoxedSpec = Box<dyn ArtifactSpec + Send + Sync>;

#[derive(Clone)]
enum Uri {
    Path(Option<PathBuf>),
    Id { extractor: Option<Arc<dyn Fn(PathBuf) -> String + Send + Sync>>, id: Option<String> },
}

impl Default for Uri {
    fn default() -> Self {
        Uri::Path(None)
    }
}

impl fmt::Debug for Uri {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self {
            &Uri::Path(Some(path)) => f.write_fmt(format_args!(
                "<Uri::Path path = '{}'>",
                path.to_str().unwrap_or("[error]")
            )),
            &Uri::Path(None) => f.write_fmt(format_args!("<Uri::Path path = unset'>")),
            &Uri::Id { extractor: _, id: Some(id_str) } => {
                f.write_fmt(format_args!("<Uri::Id id = '{}'>", id_str))
            }
            &Uri::Id { extractor: _, id: None } => {
                f.write_fmt(format_args!("<Uri::Id in = [unset]'>"))
            }
        }
    }
}

impl Uri {
    // TODO: Figure out more parsimonious ownership model (not pressing)
    fn parse(self, path: &PathBuf) -> Self {
        match self {
            Uri::Path(None) => Uri::Path(Some(path.clone())),
            Uri::Id { extractor: Some(func), id: None } => {
                let id = func(path.clone());
                Uri::Id { extractor: None, id: Some(id.clone()) }
            }
            _ => panic!(),
        }
    }
}

#[derive(Debug, Display)]
enum FetchAs {
    Uri(Option<Uri>),
    Bytes(Option<BytesMut>),
}

impl IntoPy<PyResult<Py<PyAny>>> for FetchAs {
    fn into_py(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            FetchAs::Uri(Some(Uri::Path(Some(path)))) => {
                let res: Py<PyAny> = match path.to_str() {
                    Some(path_str) => PyString::new_bound(py, path_str).as_any().clone().unbind(),
                    None => PyNone::get_bound(py).as_any().clone().unbind(),
                };
                Ok(res)
            }
            FetchAs::Uri(Some(Uri::Id { id: Some(id), .. })) => {
                Ok(PyString::new_bound(py, &id).as_any().clone().unbind())
            }
            FetchAs::Bytes(Some(bytes)) => {
                let mut buf = Vec::<u8>::new();
                consume_and_log_err(bytes.reader().read_to_end(&mut buf));
                Ok(PyByteArray::new_bound(py, buf.as_slice()).into())
            }
            FetchAs::Uri(Some(Uri::Id { id: None, .. }))
            | FetchAs::Uri(Some(Uri::Path(None)))
            | FetchAs::Uri(None)
            | FetchAs::Bytes(None) => {
                Err(PyTypeError::new_err("Cannot load model from type `None`"))
            }
        }
    }
}

impl Clone for FetchAs {
    fn clone(&self) -> Self {
        use FetchAs::*;
        match self {
            Uri(maybe_uri) => FetchAs::Uri(maybe_uri.clone()),
            Bytes(maybe_bytes) => FetchAs::Bytes(maybe_bytes.clone()),
        }
    }
}

impl From<Uri> for FetchAs {
    fn from(uri: Uri) -> Self {
        FetchAs::Uri(Some(uri))
    }
}
impl From<BytesMut> for FetchAs {
    fn from(bytes: BytesMut) -> Self {
        FetchAs::Bytes(Some(bytes))
    }
}

impl FetchAs {
    fn new_like(fetch_as: &FetchAs) -> FetchAs {
        match &fetch_as {
            &FetchAs::Uri(Some(Uri::Path(_))) => {
                FetchAs::Uri(Some(Uri::Path(Some(PathBuf::new()))))
            }
            &FetchAs::Uri(Some(Uri::Id { extractor, id: _ })) => FetchAs::Uri(Some(Uri::Id {
                extractor: extractor.clone(),
                id: Some(String::new()),
            })),
            &FetchAs::Uri(None) => FetchAs::Uri(Some(Uri::default())),
            &FetchAs::Bytes(_) => FetchAs::Bytes(Some(BytesMut::new())),
        }
    }
}

enum Command {
    UpdateEntry(BoxedSpec, FetchAs, oneshot::Sender<Result<RefreshState, CacheError>>),
    InsertEntry(CacheKey, FetchAs, oneshot::Sender<Result<(), CacheError>>),
    GetEntry(CacheKey, oneshot::Sender<CacheResult<Py<PyAny>>>),
    ListEntries(oneshot::Sender<IndexMap<String, String>>),
}

impl std::fmt::Debug for Command {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Command::*;
        match self {
            UpdateEntry(spec, _, _) => f.write_fmt(format_args!(
                "<UpdateEntry: spec = {}>",
                serde_json::to_string(spec.spec().as_ref())
                    .unwrap_or("<Failure displaying `ArtifactSpec`>".into())
            )),
            InsertEntry(key, ..) => f.write_fmt(format_args!("<InsertEntry: key = {key}>")),
            GetEntry(spec, _) => f.write_fmt(format_args!(
                "<GetEntry: spec = {}>",
                serde_json::to_string(spec.spec().as_ref())
                    .unwrap_or("<Failure displaying `ArtifactSpec`>".into())
            )),
            ListEntries(_) => f.write_str("<ListEntries>"),
        }
    }
}

impl Command {
    fn update_entry(
        spec: impl ArtifactSpec + Send + Sync + 'static,
        fetch_as: FetchAs,
    ) -> (Command, oneshot::Receiver<CacheResult<RefreshState>>) {
        let (tx, rx) = oneshot::channel::<CacheResult<RefreshState>>();
        (Command::UpdateEntry(Box::new(spec), fetch_as, tx), rx)
    }

    fn insert_entry(
        key: CacheKey,
        model_data: FetchAs,
    ) -> (Command, oneshot::Receiver<CacheResult<()>>) {
        let (tx, rx) = oneshot::channel::<CacheResult<()>>();
        let cmd = Command::InsertEntry(key, model_data, tx);
        (cmd, rx)
    }

    fn get_entry(key: CacheKey) -> (Command, oneshot::Receiver<CacheResult<Py<PyAny>>>) {
        let (tx, rx) = oneshot::channel::<CacheResult<Py<PyAny>>>();
        let cmd = Command::GetEntry(key, tx);
        (cmd, rx)
    }

    fn list_entries() -> (Command, oneshot::Receiver<IndexMap<String, String>>) {
        let (tx, rx) = oneshot::channel::<IndexMap<String, String>>();
        let cmd = Command::ListEntries(tx);
        (cmd, rx)
    }
}

pub type CacheKey = String;

struct TaskFlow {
    handle: JoinHandle<Result<(), CacheError>>,
}

struct RefreshConfig {
    spec: BoxedSpec,
    last_updated: Option<DateTime<Utc>>,
    tx: Arc<mpsc::Sender<Command>>,
    client: Arc<ArtifactsClient>,
    fetch_as: FetchAs,
    tx_result: OnceCell<oneshot::Sender<CacheResult<RefreshState>>>,
}

impl std::fmt::Display for RefreshConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let key = self.spec.as_key().map_err(|_| fmt::Error::default())?;
        f.write_fmt(format_args!(
            "<[RefreshConfig] `spec='{}'`, `last_updated={:#?}`>",
            key, self.last_updated
        ))
    }
}

impl RefreshConfig {
    fn new(
        spec: BoxedSpec,
        last_updated: Option<DateTime<Utc>>,
        client: Arc<ArtifactsClient>,
        tx: Arc<mpsc::Sender<Command>>,
        fetch_as: FetchAs,
        tx_result: oneshot::Sender<CacheResult<RefreshState>>,
    ) -> RefreshConfig {
        RefreshConfig {
            spec,
            last_updated,
            tx,
            client,
            fetch_as,
            tx_result: OnceCell::new_with(Some(tx_result)),
        }
    }
}

#[derive(Display)]
enum RefreshState {
    PendingFetch(PendingFetch),
    FetchingData(FetchingData),
    UpdatingCache(UpdatingCache),
    Done(DateTime<Utc>),
}

trait Transition<F: Send>
where
    Self: Sized,
{
    type Error;

    fn from(
        state: F,
        config: Arc<RefreshConfig>,
    ) -> impl Future<Output = Result<Self, Self::Error>>;
}

impl Transition<RefreshState> for RefreshState {
    type Error = CacheError;

    async fn from(state: RefreshState, config: Arc<RefreshConfig>) -> Result<Self, Self::Error> {
        match state {
            RefreshState::PendingFetch(state) => Ok(RefreshState::FetchingData(
                <FetchingData as Transition<PendingFetch>>::from(state, config).await?,
            )),
            RefreshState::FetchingData(state) => Ok(RefreshState::UpdatingCache(
                <UpdatingCache as Transition<FetchingData>>::from(state, config).await?,
            )),
            RefreshState::UpdatingCache(state) => match state.task.await? {
                Ok(_) => Ok(RefreshState::Done(Utc::now())),
                Err(err) => Err(err),
            },
            RefreshState::Done(next_update) => {
                let state = PendingFetch::new(next_update.into())?;
                Ok(RefreshState::PendingFetch(state))
            }
        }
    }
}

struct PendingFetch {
    next_update: UpdateTime,
    task: JoinHandle<ServiceResult<()>>,
}

impl PendingFetch {
    fn new(next_update: UpdateTime) -> Result<PendingFetch, CacheError> {
        let wait_task: Pin<Box<dyn Future<Output = ServiceResult<()>> + Send>> = match next_update {
            UpdateTime::Now => Box::pin(future::ready(Ok(()))),
            UpdateTime::DateTime(dt) => {
                let wait_seconds = (dt - Utc::now()).num_seconds();
                let wait_duration = Duration::seconds(wait_seconds)
                    .to_std()
                    .map_err(|err| CacheError::from(err.to_string()))?;

                Box::pin(async move {
                    tokio::time::sleep(wait_duration).await;
                    ServiceResult::Ok(())
                })
            }
        };
        let state = PendingFetch {
            next_update: next_update.clone(),
            task: tokio::spawn(async move { wait_task.await }),
        };
        Ok(state)
    }
}

impl From<PendingFetch> for RefreshState {
    fn from(state: PendingFetch) -> Self {
        Self::PendingFetch(state)
    }
}

struct FetchingData {
    task: JoinHandle<Result<FetchAs, CacheError>>,
}

impl From<FetchingData> for RefreshState {
    fn from(state: FetchingData) -> Self {
        Self::FetchingData(state)
    }
}

struct UpdatingCache {
    task: JoinHandle<Result<(), CacheError>>,
}

impl From<UpdatingCache> for RefreshState {
    fn from(state: UpdatingCache) -> Self {
        Self::UpdatingCache(state)
    }
}

impl Transition<PendingFetch> for FetchingData {
    type Error = CacheError;

    async fn from(
        state: PendingFetch,
        config: Arc<RefreshConfig>,
    ) -> Result<FetchingData, CacheError> {
        match state.task.await {
            Ok(Ok(_)) => {}
            Ok(Err(err)) => {
                error!(
                    "Successfully joined `PendingFetch` task for key `{}` but the task failed: {}",
                    config.spec.as_key().unwrap_or("<Unknown key>".into()),
                    err
                )
            }
            Err(_) => {
                error!(
                    "Failed to join `PendingFetch` task for key {}",
                    config.spec.as_key().unwrap_or("<Unknown key>".into())
                )
            }
        };
        let RefreshConfig { client, spec, .. } = &config.as_ref();
        let container = FetchAs::new_like(&config.fetch_as);
        let args = spec.as_map()?;
        let path = config.client.substitute(args)?;
        let fut = client.fetch(path.into(), container).into_future();
        let task = tokio::spawn(async move { fut.await });
        Ok(FetchingData { task })
    }
}

impl Transition<FetchingData> for UpdatingCache {
    type Error = CacheError;

    async fn from(
        state: FetchingData,
        config: Arc<RefreshConfig>,
    ) -> Result<UpdatingCache, CacheError> {
        let Ok(data) = state.task.await? else {
            let msg = "Failed to receive artifact data from `RefreshTask<FetchingData>`";
            return Err(CacheError::from(msg));
        };

        let (cmd, rx) = Command::insert_entry(config.spec.as_key()?, data);
        config.tx.send(cmd).await?;
        let task = tokio::spawn(async move {
            match rx.await {
                Ok(res) => res,
                Err(err) => Err(CacheError::from(err.to_string())),
            }
        });
        Ok(UpdatingCache { task })
    }
}

enum ModelEntry {
    Empty,
    Ready(Py<PyAny>, Option<DateTime<Utc>>),
    InUse(oneshot::Receiver<Py<PyAny>>, Option<DateTime<Utc>>),
    Refreshing(Option<Py<PyAny>>, Option<DateTime<Utc>>),
}

impl std::fmt::Display for ModelEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use ModelEntry::*;
        let formatted = match self {
            InUse(_, None) => "<In use, (never updated -- this shouldn't happen!)>".to_string(),
            InUse(_, Some(last_updated)) => format!("<In use, (last updated {})>", last_updated),
            Refreshing(Some(model), Some(last_updated)) => {
                format!("<Refreshing, {}>, (last updated {})", __str__(model), last_updated)
            }
            Refreshing(None, None) => "<Refreshing, no model, (never updated)>".into(),
            Refreshing(Some(model), None) => {
                format!("<Refreshing, {}, (never updated)>", __str__(model))
            }
            Refreshing(None, Some(last_updated)) => {
                format!("<Refreshing, no model, (last updated {} -- what??)>", last_updated)
            }
            Ready(model, Some(last_updated)) => {
                format!("<Ready, {}, (last updated {})>", __str__(model), last_updated)
            }
            Ready(model, None) => {
                format!("<Ready, {}, (never updated -- this shouldn't happen!)>", __str__(model))
            }
            Empty => "<Empty>".to_string(),
        };
        f.write_str(formatted.as_str())
    }
}

impl ModelEntry {
    fn last_updated(&self) -> Option<&DateTime<Utc>> {
        use ModelEntry::*;
        match self {
            Empty => None,
            Ready(_, last_updated) | InUse(_, last_updated) | Refreshing(_, last_updated) => {
                last_updated.as_ref()
            }
        }
    }

    fn borrow(self) -> CacheResult<((oneshot::Sender<Py<PyAny>>, Py<PyAny>), ModelEntry)> {
        use ModelEntry::*;
        let (tx, rx) = oneshot::channel();
        match self {
            Ready(model, last_updated) => Ok(((tx, model), InUse(rx, last_updated))),
            Refreshing(Some(model), last_updated) => Ok(((tx, model), InUse(rx, last_updated))),
            // TODO: Wait for refresh
            Refreshing(None, _) => Err(CacheError::EmptyEntryError),
            Empty => Err(CacheError::EmptyEntryError),
            InUse(_, _) => Err(CacheError::InUseError),
        }
    }

    fn refreshing(entry: ModelEntry) -> CacheResult<ModelEntry> {
        use ModelEntry::*;
        match entry {
            Empty => Ok(Refreshing(None, None)),
            Ready(model, last_updated) => Ok(Refreshing(Some(model), last_updated)),
            Refreshing(Some(model), last_updated) => Ok(Refreshing(Some(model), last_updated)),
            entry => Err(CacheError::from(format!(
                "Failed to update model entry from previous entry state `{}`",
                entry
            ))),
        }
    }
}

#[pyclass]
#[derive(Debug)]
pub struct Cache {
    client: Arc<ArtifactsClient>,
    schedule: Schedule,
    max_size: Option<usize>,
    pre_fetch_all: Option<bool>,
    tx: OnceCell<Arc<mpsc::Sender<Command>>>,
    task: OnceCell<JoinHandle<Result<(), CacheError>>>,
}

impl std::fmt::Display for Cache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("<Cache (client `{:#?}`)>", self.client))
    }
}

#[derive(Clone)]
pub struct CacheHandle {
    tx: Arc<mpsc::Sender<Command>>,
}

impl CacheHandle {
    pub async fn get(&self, key: &CacheKey) -> CacheResult<Py<PyAny>> {
        let (cmd, rx) = Command::get_entry(key.clone());
        self.tx.try_send(cmd)?;
        match rx.await.map_err(CacheError::from) {
            Err(_) => Err(CacheError::from("Failed to receive response from cache task")),
            Ok(Err(err)) => {
                Err(CacheError::from(format!("Received error response from cache task: {}", err)))
            }
            Ok(Ok(model)) => Ok(model),
        }
    }
}

fn list_entries(
    model_cache: &LruCache<CacheKey, (Option<TaskFlow>, ModelEntry)>,
) -> IndexMap<String, String> {
    model_cache.iter().map(|(key, (_, entry))| (key.clone(), format!("{}", entry))).collect()
}

impl Cache {
    pub fn new(
        client: ArtifactsClient,
        max_size: Option<usize>,
        schedule: Option<Schedule>,
        pre_fetch_all: Option<bool>,
    ) -> Cache {
        Cache {
            client: client.into(),
            schedule: schedule.unwrap_or_default(),
            max_size,
            pre_fetch_all,
            tx: OnceCell::<Arc<mpsc::Sender<Command>>>::default(),
            task: OnceCell::<JoinHandle<Result<(), CacheError>>>::default(),
        }
    }

    pub fn run(&self) -> CacheResult<()> {
        let runtime = TOKIO_RUNTIME.get_or_init(build_runtime);
        let max_size = self.max_size.unwrap_or(DEFAULT_CACHE_SIZE);
        let (tx, rx) = mpsc::channel::<Command>(max_size);
        let fut_task = Cache::task(
            self.client.clone().into(),
            max_size,
            tx.clone().into(),
            rx,
            self.pre_fetch_all,
        );
        if let Err(err) = self.tx.set(tx.into()) {
            error!("Failed to set Cache tx: {}", err)
        };
        self.task
            .set(runtime.spawn(fut_task))
            .map_err(|_| CacheError::from("Cannot start already running Cache"))
    }

    pub fn handle(&self) -> CacheResult<CacheHandle> {
        let tx = self.tx.get().clone().ok_or_else(|| {
            CacheError::from("Cannot obtain handle for Cache that has not yet been started")
        })?;
        Ok(CacheHandle { tx: tx.clone() })
    }

    #[instrument(skip_all)]
    async fn task(
        client: Arc<ArtifactsClient>,
        max_size: usize,
        tx_cmd: Arc<mpsc::Sender<Command>>,
        mut rx_cmd: mpsc::Receiver<Command>,
        pre_fetch_all: Option<bool>,
    ) -> Result<(), CacheError> {
        let mut model_cache: LruCache<CacheKey, (Option<TaskFlow>, ModelEntry)> =
            LruCache::new(NonZeroUsize::new(max_size).unwrap());

        info!(
            "Starting model cache task with `max_size = {}`, `pre_fetch_all = {}`",
            max_size,
            pre_fetch_all.unwrap_or(false)
        );

        if pre_fetch_all.is_some_and(|pre_fetch_all| pre_fetch_all) {
            let paths_by_spec = client.list(None).await;
            let rxs = paths_by_spec
                .iter()
                .map(|(spec, path)| {
                    info!(
                        "Found artifact for spec `{}` at path `{}`",
                        &spec.as_key().unwrap_or("<Error deriving key from spec>".into()),
                        path.display()
                    );
                    let (cmd, rx_result) = Command::update_entry(spec.clone(), FetchAs::Uri(None));
                    let cloned_tx_cmd = tx_cmd.clone();
                    let error_msg =
                        format!("Failed to send update entry command for spec: {:#?}", spec);
                    tokio::spawn(async move {
                        let fut_update_entry = cloned_tx_cmd.send(cmd);
                        if let Err(_) = fut_update_entry.await {
                            warn!(error_msg);
                        };
                        rx_result.await
                    })
                })
                .collect::<Vec<_>>();

            let (cmd, rx) = Command::list_entries();
            let _ = tx_cmd.send(cmd).await;
            let _fut_list_entries = tokio::spawn(async move {
                join_all(rxs).await;
                let Ok(entries) = rx.await else {
                    error!("Failed to receive `ListEntries` result");
                    return;
                };
                info!("Model cache contains entries: {:#?}", entries)
            });
        }

        loop {
            let Some(msg) = rx_cmd.recv().await else { return Err(CacheError::from("")) };
            match msg {
                Command::UpdateEntry(spec, fetch_as, tx_result) => {
                    let key = spec.as_key()?;
                    let model_entry = match model_cache.pop(&key) {
                        // TODO: What if there is an extant TaskFlow?
                        Some((_taskflow, model_entry)) => model_entry,
                        None => ModelEntry::Empty,
                    };
                    let config = RefreshConfig::new(
                        spec,
                        model_entry.last_updated().cloned(),
                        client.clone(),
                        tx_cmd.clone(),
                        fetch_as,
                        tx_result,
                    );

                    let taskflow = match Self::run_taskflow(config.into()) {
                        Ok(taskflow) => {
                            info!("Initiated refresh taskflow for artifact key `{}`", key);
                            taskflow
                        }
                        Err(err) => {
                            error!(
                                "Failed to initiate refresh taskflow for artifact spec: `{}`: {}",
                                key, err
                            );
                            continue;
                        }
                    };
                    let entry = match ModelEntry::refreshing(model_entry) {
                        Ok(entry) => entry,
                        Err(err) => {
                            error!(
                                "Failed to create model entry for refreshing artifact with key {}: {}",
                                key, err
                            );
                            continue;
                        }
                    };
                    model_cache.put(key, (Some(taskflow), entry));
                }
                Command::InsertEntry(key, data, tx) => {
                    let result = Self::insert_entry(&client, &mut model_cache, key.clone(), data);
                    match &result {
                        Ok(_) => {
                            info!("Successfully inserted refreshed model entry for key `{}`", key);
                        }
                        Err(err) => {
                            error!(
                                "Failed to load or insert refreshed model entry for key `{}`: {}",
                                key, err
                            )
                        }
                    };

                    if let Err(_) = tx.send(result) {
                        error!(
                            "Failed to return result of `InsertEntry` command to original issuer"
                        );
                    };
                }
                Command::GetEntry(key, tx) => {
                    let span = info_span!("get_entry");
                    let _entered = span.enter();
                    let models_by_key = list_entries(&model_cache);
                    info!("models: {:#?}", models_by_key);
                    info!("Fetching model entry for key: '{}'", key);
                    let result_send = match model_cache.get(&key) {
                        Some((_, ModelEntry::Ready(model, _))) => tx.send(Ok(model.clone())),
                        // TODO: Make errors more specific
                        None => {
                            warn!("No entry for key {}", key);
                            tx.send(Err(CacheError::Error("Failed to obtain model".into())))
                        }
                        _ => tx.send(Err(CacheError::Error("Failed to obtain model".into()))),
                    };
                    if let Err(_) = result_send {
                        warn!("Failed to send response");
                    };
                }
                Command::ListEntries(tx) => {
                    let models_by_key = list_entries(&model_cache);
                    tx.send(models_by_key);
                }
            }
        }
    }

    fn run_taskflow(mut config: RefreshConfig) -> Result<TaskFlow, CacheError> {
        let key = config.spec.as_key()?;
        let handle: JoinHandle<Result<(), CacheError>> = tokio::spawn(async move {
            let start = PendingFetch::new(UpdateTime::Now)?;
            let mut state = <RefreshState as From<PendingFetch>>::from(start);
            let tx_result = (&mut config).tx_result.take();

            let cloneable_config = Arc::new(config);
            let result = loop {
                // Precompute error message to avoid borrow issues later in loop body
                let err_msg = format!("Failed to transition from `{}`", state);
                let next = <RefreshState as Transition<RefreshState>>::from(
                    state,
                    cloneable_config.clone(),
                );
                match next.await {
                    Ok(RefreshState::Done(last_updated)) => {
                        info!("Finished cache update taskflow for key {} at {}", key, last_updated);
                        break Ok(RefreshState::Done(last_updated));
                    }
                    Ok(next) => {
                        info!(
                            "Successfully transitioned to taskflow state `{}` for key `{}`",
                            next, key
                        );
                        state = next;
                    }
                    Err(err) => {
                        error!("{}: {}", err_msg, err);
                        break Err(err);
                    }
                }
            };

            do_in!(|| {
                if let Err(_) = tx_result?.send(result) {
                    warn!("Failed to send result to original command issuer");
                };
            });

            Ok(())
        });

        Ok(TaskFlow { handle })
    }

    #[instrument(skip(client, cache, data))]
    fn insert_entry(
        client: &ArtifactsClient,
        cache: &mut LruCache<CacheKey, (Option<TaskFlow>, ModelEntry)>,
        key: CacheKey,
        data: FetchAs,
    ) -> Result<(), CacheError> {
        info!("Attempting to load and insert model entry");
        match (client.load_model(data), cache.pop(&key)) {
            (Ok(new_model), Some((taskflow, model_entry))) => {
                info!("Successfully loaded model: `{}`", __str__(&new_model));
                // We expect `model_entry` to be in the `Refreshing` state
                match model_entry {
                    // We're trying to update a model for which a refresh has not been initiated
                    ModelEntry::Ready(old_model, _) => Err(CacheError::from(format!(
                        "Tried to update model entry {:#?} at key {} without properly initiating refresh taskflow",
                        old_model, key
                    ))),
                    ModelEntry::Refreshing(old_model, last_updated_prev) => {
                        let new_model_str = __str__(&new_model);
                        let last_updated = Utc::now();
                        cache.put(
                            key.clone(),
                            (taskflow, ModelEntry::Ready(new_model, Some(last_updated.clone()))),
                        );
                        info!(
                            %key,
                            timestamp = Utc::now().to_string(),
                            "Replaced model `{:#?}` with {} (previously updated {})",
                            old_model,
                            new_model_str,
                            last_updated_prev.map(|dt|dt.to_string()).unwrap_or("never".into())
                        );
                        Ok(())
                    }
                    ModelEntry::InUse(rx, last_updated) => {
                        // TODO: Await return of model
                        cache.put(key, (taskflow, ModelEntry::Ready(new_model, last_updated)));
                        Ok(())
                    }
                    ModelEntry::Empty => {
                        let msg = format!(
                            "Tried to update empty model entry at key {} without properly initiating refresh taskflow",
                            key
                        );
                        Err(CacheError::from(msg))
                    }
                }
            }
            (Ok(_new_model), None) => {
                // We should not reach this state because initiating a refresh taskflow
                // puts the `TaskFlow` struct in the cache entry.
                let msg = "Trying to replace model without taskflow";
                Err(CacheError::from(msg))
            }
            (Err(err), _) => {
                error!("Failed to load model: {}", err);
                Err(err.into())
            }
        }
    }

    // TODO: We may remove this later because so far the `Cache` API is handled through the `CacheHandle`
    fn try_send(&self, command: Command) -> Result<(), CacheError> {
        self.tx
            .get()
            .ok_or_else(|| {
                CacheError::from("Cannot send command to Cache that has not yet been started")
            })?
            .try_send(command)
            .map_err(|err| err.into())
    }
}

#[pymethods]
impl Cache {
    #[new]
    // TODO: Include optional `schedule` and `interval` parameters for entry refresh
    fn __new__(client: FsClient, load_model: Py<PyFunction>) -> Cache {
        Cache::new(ArtifactsClient::FsClient { client, load_model }, None, None, Some(true))
    }
}

#[derive(Error, Debug)]
pub enum CacheError {
    #[error("Error: {0}")]
    Error(String),
    #[error("Failed to send command: {0}")]
    SendCommandError(#[from] SendError<Command>),
    #[error("Failed to send command: {0}")]
    TrySendCommandError(#[from] TrySendError<Command>),
    // #[error("CacheError: Failed to receive response (channel closed)")]
    // ReceiveResponseError(#[from] Canceled),
    #[error("Error while awaiting step: {0}")]
    JoinError(#[from] JoinError),
    #[error("Error serializing `ArtifactSpec`: {0}")]
    SerializeError(#[from] serde_json::Error),
    #[error("IO Error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Python Error: {0}")]
    PythonError(#[from] pyo3::PyErr),
    #[error("Cannot get model for key: entry is empty")]
    EmptyEntryError,
    #[error("Cannot get model for key: entry is in use")]
    InUseError,
    #[error("Failed to receive message over oneshot channel")]
    RecvError(#[from] oneshot::error::RecvError),
}

impl From<String> for CacheError {
    fn from(err: String) -> Self {
        CacheError::Error(err)
    }
}

impl From<&str> for CacheError {
    fn from(err: &str) -> Self {
        CacheError::Error(err.to_string())
    }
}

pub type CacheResult<T> = Result<T, CacheError>;

trait Client: Send + Sync + 'static {
    fn substitute(&self, args: IndexMap<String, String>) -> CacheResult<String>;

    fn list(
        &self,
        path: Option<&PathBuf>,
    ) -> Pin<Box<dyn Future<Output = Vec<(IndexMap<String, String>, PathBuf)>> + Send>>;

    fn fetch_uri(
        &self,
        path: &PathBuf,
        uri: Uri,
    ) -> Pin<Box<dyn Future<Output = Result<Uri, CacheError>> + Send + Sync + 'static>> {
        Box::pin(future::ready(Ok(uri.parse(path))))
    }

    fn fetch_bytes(
        &self,
        path: PathBuf,
        bytes: BytesMut,
    ) -> Pin<Box<dyn Future<Output = Result<BytesMut, CacheError>> + Send + 'static>>;
}

#[derive(Clone, Debug)]
pub struct PathTemplate {
    template: String,
    base: PathBuf,
}

enum PathComponent {
    Fixed(String),
    Identifier(String),
}

impl PathTemplate {
    fn components(&self) -> VecDeque<PathComponent> {
        self.template
            .split("/")
            .filter_map(|part| {
                if part.starts_with("{") && part.ends_with("}") {
                    let n_chars = part.len();
                    let identifier: String = part.chars().skip(1).take(n_chars - 2).collect();
                    Some(PathComponent::Identifier(identifier))
                } else if part != "" {
                    Some(PathComponent::Fixed(part.to_string()))
                } else {
                    None
                }
            })
            .collect()
    }

    fn parse(&self, path: &PathBuf) -> IndexMap<String, String> {
        let mut base_parts: VecDeque<_> = self.base.iter().collect();
        let rel_path: PathBuf = path
            .into_iter()
            .skip_while(|&part| {
                let Some(base_part) = base_parts.pop_front() else { return false };
                base_part == part
            })
            .collect();

        self.components()
            .into_iter()
            .filter_map(|part| match part {
                PathComponent::Fixed(_) => None,
                PathComponent::Identifier(identifier) => Some(identifier),
            })
            .zip(rel_path.into_iter().filter_map(|part| part.to_str().map(str::to_string)))
            .collect()
    }

    fn substitute(&self, args: IndexMap<String, String>) -> CacheResult<String> {
        self.components()
            .iter()
            .try_fold(None, |path: Option<String>, component| {
                let next_path_component = match component {
                    PathComponent::Fixed(dir_name) => dir_name,
                    PathComponent::Identifier(identifier) => {
                        args.get(identifier).ok_or_else(|| {
                            CacheError::from(format!(
                                "Substitution identifier `{}` is not present in the provided args",
                                identifier
                            ))
                        })?
                    }
                };
                let path = match path {
                    None => next_path_component.to_string(),
                    Some(path) => vec![path, next_path_component.to_string()].join("/"),
                };
                CacheResult::<Option<String>>::Ok(Some(path))
            })?
            .ok_or_else(|| CacheError::from("Failed to substitute args into path template"))
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct FsClient {
    base_path: PathBuf,
    path_template: PathTemplate,
}

impl FsClient {
    pub fn new(base_path: PathBuf, path_template: String) -> FsClient {
        FsClient {
            // TODO: Remove redundant `base_path` field somewhere
            base_path: base_path.clone(),
            path_template: PathTemplate { template: path_template, base: base_path },
        }
    }
}

#[pymethods]
impl FsClient {
    #[new]
    pub fn __new__(base_path: Py<PyAny>, path_template: Py<PyString>) -> PyResult<FsClient> {
        Python::with_gil(|py| {
            let client = FsClient::new(
                base_path.extract::<PathBuf>(py)?,
                path_template.extract::<String>(py)?,
            );
            Ok(client)
        })
    }
}

#[derive(Clone, Debug)]
pub enum ArtifactsClient {
    FsClient { client: FsClient, load_model: Py<PyFunction> },
}

impl ArtifactsClient {
    fn fetch(
        &self,
        path: PathBuf,
        fetch_as: FetchAs,
    ) -> Pin<Box<dyn Future<Output = Result<FetchAs, CacheError>> + Send + 'static>> {
        // TODO: This may be fine for now but we should wrap in an `Arc``
        let client = self.clone();
        match fetch_as {
            FetchAs::Uri(Some(uri)) => {
                Box::pin(client.fetch_uri(&path, uri).and_then(|uri| async { Ok(uri.into()) }))
            }
            FetchAs::Bytes(Some(bytes)) => {
                if !bytes.is_empty() {
                    return Box::pin(future::ready(Err(CacheError::from("Bytes must be empty"))));
                };
                Box::pin(async move { Ok(client.fetch_bytes(path, bytes).await?.into()) })
            }
            FetchAs::Bytes(None) => {
                Box::pin(async move { Ok(client.fetch_bytes(path, BytesMut::new()).await?.into()) })
            }
            _ => panic!(),
        }
    }

    fn load_model(&self, data: FetchAs) -> PyResult<Py<PyAny>> {
        match self {
            Self::FsClient { load_model, .. } => {
                Python::with_gil(|py| load_model.call_bound(py, (data.into_py(py)?,), None))
            }
        }
    }
}

impl Client for ArtifactsClient {
    fn substitute(&self, args: IndexMap<String, String>) -> CacheResult<String> {
        match self {
            ArtifactsClient::FsClient { client, .. } => client.substitute(args),
        }
    }

    fn fetch_bytes(
        &self,
        path: PathBuf,
        bytes: BytesMut,
    ) -> Pin<Box<dyn Future<Output = Result<BytesMut, CacheError>> + Send + 'static>> {
        use ArtifactsClient::*;
        match self {
            FsClient { client, .. } => client.fetch_bytes(path, bytes),
        }
    }

    fn list(
        &self,
        path: Option<&PathBuf>,
    ) -> Pin<Box<dyn Future<Output = Vec<(IndexMap<String, String>, PathBuf)>> + Send>> {
        use ArtifactsClient::*;
        match self {
            FsClient { client, .. } => client.list(path),
        }
    }
}

fn _list(
    mut paths_by_spec: Vec<(IndexMap<String, String>, PathBuf)>,
    mut remaining_components: VecDeque<PathComponent>,
) -> Vec<(IndexMap<String, String>, PathBuf)> {
    match remaining_components.pop_front() {
        Some(PathComponent::Fixed(dir_name)) => {
            paths_by_spec.iter_mut().for_each(|(_, ref mut path)| path.push(dir_name.clone()));
            _list(paths_by_spec, remaining_components)
        }
        Some(PathComponent::Identifier(identifier)) => {
            _list(
                paths_by_spec
                    .into_iter()
                    .filter_map(|(spec, path)| {
                        //
                        Some(((spec, path.clone()), fs::read_dir(path).ok()?))
                    })
                    .flat_map(|((spec, path), dir)| {
                        let cloned_identifier = identifier.clone();
                        dir.filter_map(move |entry| {
                            let name = entry.ok()?.file_name().to_string_lossy().to_string();
                            let (mut cloned_spec, mut cloned_path) = (spec.clone(), path.clone());
                            cloned_spec.insert(cloned_identifier.clone(), name.clone());
                            cloned_path.push(name);
                            Some((cloned_spec, cloned_path))
                        })
                    })
                    .collect(),
                remaining_components,
            )
        }
        None => paths_by_spec,
    }
}

impl Client for FsClient {
    fn substitute(&self, args: IndexMap<String, String>) -> CacheResult<String> {
        self.path_template.substitute(args)
    }

    fn list(
        &self,
        path: Option<&PathBuf>,
    ) -> Pin<Box<dyn Future<Output = Vec<(IndexMap<String, String>, PathBuf)>> + Send>> {
        let spec = IndexMap::<String, String>::default();
        let paths_by_spec = _list(
            [(spec, path.unwrap_or(&self.base_path).clone())].into(),
            self.path_template.components(),
        );
        Box::pin(future::ready(paths_by_spec))
    }

    fn fetch_bytes(
        &self,
        path: PathBuf,
        mut bytes: BytesMut,
    ) -> Pin<Box<dyn Future<Output = Result<BytesMut, CacheError>> + Send + 'static>> {
        Box::pin(async move {
            bytes.extend(tokio::fs::read(path).await?);
            Ok(bytes)
        })
    }
}
