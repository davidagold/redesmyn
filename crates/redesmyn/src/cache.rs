use crate::error::ServiceResult;
use bytes::{Buf, BufMut, BytesMut};
use chrono::{DateTime, Duration, Utc};
use core::fmt;
use cron;
use futures::{channel::oneshot::Canceled, TryFutureExt};
use lru::LruCache;
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    pyclass, pymethods,
    types::{PyByteArray, PyFunction, PyNone, PyString},
    FromPyObject, IntoPy, Py, PyAny, PyResult, Python,
};
use serde::Serialize;
use std::{
    collections::{BTreeMap, VecDeque},
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
        oneshot,
    },
    task::{JoinError, JoinHandle},
};
use tracing::{error, info, instrument, warn};

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

pub trait ArtifactSpec {
    fn spec(&self) -> Box<&dyn erased_serde::Serialize>;

    fn as_map(&self) -> Result<BTreeMap<String, String>, serde_json::Error> {
        let mut writer = BytesMut::new().writer();
        serde_json::to_writer(&mut writer, self.spec().as_ref())?;
        serde_json::from_reader(writer.into_inner().reader())
    }

    fn as_key(&self) -> Result<CacheKey, serde_json::Error> {
        let parts = self.as_map()?.into_iter().map(|(k, v)| [k, v].join("="));
        let key = parts.collect::<Vec<_>>().join("/");
        Ok(key)
    }

    fn as_path(&self) -> Result<PathBuf, serde_json::Error> {
        let path = self.as_map()?.values().collect();
        Ok(path)
    }
}

impl Serialize for dyn ArtifactSpec {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.spec().as_ref().serialize(serializer)
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
    // Utf8String(Option<String>),
    // TmpFile(Option<tokio::fs::File>),
}

impl IntoPy<PyResult<Py<PyAny>>> for FetchAs {
    fn into_py(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            FetchAs::Uri(Some(Uri::Path(Some(path)))) => {
                let res: &PyAny = match path.to_str() {
                    Some(path_str) => PyString::new(py, path_str).into(),
                    None => PyNone::get(py).into(),
                };
                Ok(res.into())
            }
            FetchAs::Uri(Some(Uri::Id { id: Some(id), .. })) => Ok(PyString::new(py, &id).into()),
            FetchAs::Bytes(Some(bytes)) => {
                let mut buf = Vec::<u8>::new();
                bytes.reader().read_to_end(&mut buf);
                Ok(PyByteArray::new(py, buf.as_slice()).into())
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
            // Utf8String(maybe_s) => FetchAs::Utf8String(maybe_s.clone()),
            // Not exactly sure what to do with this one.
            // TmpFile(_) => FetchAs::TmpFile(None),
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
            // &FetchAs::Utf8String(_) => FetchAs::Utf8String(Some(String::new())),
            // &FetchAs::TmpFile(_) => FetchAs::TmpFile(None),
        }
    }
}

enum Command {
    UpdateEntry(BoxedSpec, FetchAs),
    InsertEntry(CacheKey, FetchAs, oneshot::Sender<Result<(), CacheError>>),
    GetEntry(CacheKey, oneshot::Sender<CacheResult<Py<PyAny>>>),
}

impl std::fmt::Debug for Command {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Command::*;
        match self {
            UpdateEntry(spec, _) => f.write_fmt(format_args!(
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
        }
    }
}

impl Command {
    fn update_entry(spec: impl ArtifactSpec + Send + Sync + 'static, fetch_as: FetchAs) -> Command {
        Command::UpdateEntry(Box::new(spec), fetch_as)
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
}

pub type CacheKey = String;

struct TaskFlow {
    config: Arc<RefreshConfig>,
    handle: JoinHandle<Result<(), CacheError>>,
}

#[derive(Debug)]
struct RefreshConfig {
    spec: BoxedSpec,
    last_updated: Option<DateTime<Utc>>,
    tx: Arc<mpsc::Sender<Command>>,
    client: Arc<ArtifactsClient>,
    fetch_as: FetchAs,
}

impl RefreshConfig {
    fn new(
        spec: BoxedSpec,
        last_updated: Option<DateTime<Utc>>,
        client: Arc<ArtifactsClient>,
        tx: Arc<mpsc::Sender<Command>>,
        fetch_as: FetchAs,
    ) -> RefreshConfig {
        RefreshConfig { spec, last_updated, tx, client, fetch_as }
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
            RefreshState::UpdatingCache(state) => {
                Ok(RefreshState::Done(state.task.await.map(|_| Utc::now())?))
            }
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
        state.task.await?;
        let RefreshConfig { client, spec, .. } = &config.as_ref();

        let container = FetchAs::new_like(&config.fetch_as);
        let fut = client.fetch(spec.as_path()?, container).into_future();
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
        let (tx, rx) = oneshot::channel::<Result<(), CacheError>>();
        let cmd = Command::InsertEntry(config.spec.as_key()?.clone(), data, tx);
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

#[derive(Display)]
enum ModelEntry {
    Empty,
    Ready(Py<PyAny>),
    InUse(oneshot::Receiver<Py<PyAny>>),
    Refreshing(Option<Py<PyAny>>),
}

impl ModelEntry {
    fn borrow(self) -> CacheResult<((oneshot::Sender<Py<PyAny>>, Py<PyAny>), ModelEntry)> {
        use ModelEntry::*;
        let (tx, rx) = oneshot::channel();
        match self {
            Ready(model) => Ok(((tx, model), InUse(rx))),
            Refreshing(Some(model)) => Ok(((tx, model), InUse(rx))),
            // TODO: Wait for refresh
            Refreshing(None) => Err(CacheError::EmptyEntryError),
            Empty => Err(CacheError::EmptyEntryError),
            InUse(_) => Err(CacheError::InUseError),
        }
    }

    fn refreshing(entry: ModelEntry) -> CacheResult<ModelEntry> {
        use ModelEntry::*;
        match entry {
            Empty => Ok((Refreshing(None))),
            Ready(model) => Ok(Refreshing(Some(model))),
            Refreshing(Some(model)) => Ok(Refreshing(Some(model))),
            entry => Err(CacheError::from(format!(
                "Failed to update model entry from previous entry state `{}`",
                entry
            ))),
        }
    }
}

#[derive(Debug)]
pub struct Cache {
    client: Arc<ArtifactsClient>,
    schedule: Schedule,
    max_size: Option<usize>,
    pre_fetch_all: Option<bool>,
    tx: Arc<mpsc::Sender<Command>>,
    task: JoinHandle<Result<(), CacheError>>,
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

impl Cache {
    pub fn new(
        client: ArtifactsClient,
        max_size: Option<usize>,
        schedule: Option<Schedule>,
        pre_fetch_all: Option<bool>,
    ) -> Cache {
        let (tx, rx) = mpsc::channel::<Command>(max_size.unwrap_or(DEFAULT_CACHE_SIZE));
        let fut_task =
            Cache::task(client.clone().into(), max_size, tx.clone().into(), rx, pre_fetch_all);
        let task = tokio::spawn(fut_task);

        Cache {
            client: client.into(),
            schedule: schedule.unwrap_or_default(),
            max_size,
            pre_fetch_all,
            tx: tx.into(),
            task,
        }
    }

    pub fn handle(&self) -> CacheHandle {
        CacheHandle { tx: self.tx.clone() }
    }

    #[instrument(skip_all)]
    async fn task(
        client: Arc<ArtifactsClient>,
        max_size: Option<usize>,
        tx_cmd: Arc<mpsc::Sender<Command>>,
        mut rx_cmd: mpsc::Receiver<Command>,
        pre_fetch_all: Option<bool>,
    ) -> Result<(), CacheError> {
        let mut model_cache: LruCache<CacheKey, (TaskFlow, ModelEntry)> =
            LruCache::new(NonZeroUsize::new(max_size.unwrap_or(DEFAULT_CACHE_SIZE)).unwrap());

        info!(
            "Starting model cache task with `max_size = {}`, `pre_fetch_all = {}`",
            max_size.unwrap_or(DEFAULT_CACHE_SIZE),
            pre_fetch_all.unwrap_or(false)
        );

        if pre_fetch_all.is_some_and(|pre_fetch_all| pre_fetch_all) {
            let paths_by_spec = client.list(None).await;
            for (spec, path) in paths_by_spec {
                info!("`spec`: {:#?}", spec);
                info!("Path: {:#?}", path);

                let cmd = Command::update_entry(spec.clone(), FetchAs::Uri(None));
                if let Err(_) = tx_cmd.send(cmd).await {
                    warn!("Failed to send update entry command for spec: {:#?}", spec);
                };
            }
        }

        loop {
            let Some(msg) = rx_cmd.recv().await else { return Err(CacheError::from("")) };
            match msg {
                Command::UpdateEntry(spec, fetch_as) => {
                    let key = spec.as_key()?;
                    let (last_updated, model_entry) = match model_cache.pop(&key) {
                        Some((taskflow, model)) => (taskflow.config.last_updated, model),
                        None => (None, ModelEntry::Empty),
                    };
                    let config = RefreshConfig::new(
                        spec,
                        last_updated,
                        client.clone(),
                        tx_cmd.clone(),
                        fetch_as,
                    );
                    let taskflow = match Self::spawn_taskflow(config.into()) {
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
                    model_cache.put(key, (taskflow, entry));
                }
                Command::InsertEntry(key, data, tx) => {
                    // TODO: It's a little strange to include the refresh task flow and the model in the same cache entry.
                    //       Furthermore, we need to use a concurrent hashmap, or something like it, to lock access
                    //       to individual keys while we are updating the respective models.
                    if let Err(err) =
                        Self::insert_entry(&client, &mut model_cache, key.clone(), data, tx)
                    {
                        error!("Failed to load and insert model for key {}: {}", key, err)
                    };
                }
                Command::GetEntry(key, tx) => {
                    info!("Fetching model entry for key: '{}'", key);
                    let result_send = match model_cache.get(&key) {
                        Some((_, ModelEntry::Ready(model))) => tx.send(Ok(model.clone())),
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
            }
        }
    }

    fn spawn_taskflow(config: Arc<RefreshConfig>) -> Result<TaskFlow, CacheError> {
        // The following does not refresh the cache entry directly, but rather spawns a new task
        // to handle data fetching for the given key and tracks the task's association with said key.
        // self.refresh(key).await;
        //
        // Will need to get the che entry, lock the key, get last updated, pass everything to `refresh_entry`
        let _config = config.clone();

        let key = config.spec.as_key()?;
        let handle: JoinHandle<Result<(), CacheError>> = tokio::spawn(async move {
            let start = PendingFetch::new(UpdateTime::Now)?;
            let mut state = <RefreshState as From<PendingFetch>>::from(start);

            // THIS IS WHY WE NEED ASYNC ITERATORS
            let end = loop {
                // Pre-compute error message to avoid borrow issues later in loop body
                let err_msg = format!("Failed to transition from `{}`", state);
                let next = <RefreshState as Transition<RefreshState>>::from(state, _config.clone());
                match next.await {
                    Ok(RefreshState::Done(last_updated)) => {
                        break Ok(RefreshState::Done(last_updated));
                    }
                    Ok(next) => {
                        info!("Entering taskflow state `{}` for key `{}`", next, key);
                        state = next;
                    }
                    Err(err) => {
                        error!("{}: {}", err_msg, err);
                        break Err(err);
                    }
                }
            }?;

            info!("Reached state `{}` for key {}", end, key);
            Ok(())
        });

        Ok(TaskFlow { config, handle })
    }

    fn insert_entry(
        client: &ArtifactsClient,
        cache: &mut LruCache<CacheKey, (TaskFlow, ModelEntry)>,
        key: CacheKey,
        data: FetchAs,
        tx: oneshot::Sender<Result<(), CacheError>>,
    ) -> Result<(), CacheError> {
        let result = match (client.load_model(data), cache.pop(&key)) {
            // Successfully refreshed an existing model entry
            (Ok(new_model), Some((taskflow, model_entry))) => {
                match model_entry {
                    // We're trying to update a model for which a refresh has not been initiated
                    ModelEntry::Ready(old_model) => {
                        // info!(%key, timestamp = Utc::now().to_string(), "Replacing model {:#?}", old_model);
                        Err(CacheError::from(format!(
                            "Tried to update model entry {:#?} at key {} without properly initiating refresh taskflow",
                            old_model, key
                        )))
                    }
                    ModelEntry::Refreshing(old_model) => {
                        info!(%key, timestamp = Utc::now().to_string(), "Replacing model {:#?}", old_model);
                        cache.put(key, (taskflow, ModelEntry::Ready(new_model)));
                        Ok(())
                    }
                    ModelEntry::InUse(rx) => {
                        // TODO: Await return of model
                        cache.put(key, (taskflow, ModelEntry::Ready(new_model)));
                        Ok(())
                    }
                    ModelEntry::Empty => Err(CacheError::from(format!(
                        "Tried to update empty model entry at key {} without properly initiating refresh taskflow",
                        key
                    ))),
                }
            }
            (Ok(model), None) => {
                // TODO: Handle this case better.
                let msg = "Trying to replace model without taskflow";
                Err(CacheError::from(msg))
            }
            (Err(err), _) => Err(err.into()),
        };
        match tx.send(result) {
            Err(Err(err)) => return Err(err),
            Err(Ok(())) => {
                let msg = "Successfully fetched and loaded model but failed to communicate result to refresh task";
                return Err(CacheError::from(msg));
            }
            _ => Ok(()),
        }
    }

    fn try_send(&self, command: Command) -> Result<(), CacheError> {
        self.tx.try_send(command).map_err(|err| err.into())
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
    #[error("CacheError: Failed to receive response (channel closed)")]
    ReceiveResponseError(#[from] Canceled),
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

#[pyclass]
#[derive(Debug)]
pub struct PyCache {
    pub cache: Arc<Cache>,
}

#[pymethods]
impl PyCache {
    #[new]
    fn new(
        client_spec: ClientSpec,
        load_model: &PyFunction,
        max_size: Option<usize>,
        cron_expr: Option<String>,
        max_age_seconds: Option<u32>,
        pre_fetch_all: Option<bool>,
    ) -> PyResult<PyCache> {
        let schedule: Option<Schedule> = match (cron_expr, max_age_seconds) {
            (Some(expr), None) => Some(
                cron::Schedule::from_str(expr.as_str())
                    .map_err(|err| PyValueError::new_err(err.to_string()))?
                    .into(),
            ),
            (None, Some(seconds)) => Some(seconds.into()),
            (None, None) => None,
            (Some(_), Some(_)) => {
                let msg = "Specify *at most* one of either `cron_expr` or `max_age_seconds`.";
                return Err(PyValueError::new_err(msg));
            }
        };

        Ok(PyCache {
            cache: Cache::new(
                client_spec.to_client(load_model.into()),
                max_size,
                schedule,
                pre_fetch_all,
            )
            .into(),
        })
    }

    #[pyo3(signature = (**kwargs))]
    fn get<'py>(
        &'py self,
        py: Python<'py>,
        kwargs: Option<BTreeMap<String, String>>,
    ) -> PyResult<&'py PyAny> {
        let spec = kwargs.unwrap_or_default();
        let key = spec.as_key().map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        let (cmd, rx) = Command::get_entry(key);
        let _ = self.cache.try_send(cmd);
        pyo3_asyncio::tokio::future_into_py(py, async move {
            match rx.await {
                Ok(Ok(model)) => Ok(model),
                Ok(Err(err)) => Err(PyRuntimeError::new_err(err.to_string())),
                Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
            }
        })
    }
}

#[derive(FromPyObject)]
enum ClientSpec {
    FsClient { protocol: String, base_path: PathBuf, path_template: String },
}

impl ClientSpec {
    fn to_client(self, load_model: Py<PyFunction>) -> ArtifactsClient {
        match self {
            ClientSpec::FsClient { protocol, base_path, path_template } => {
                ArtifactsClient::FsClient {
                    client: FsClient::new(base_path, path_template),
                    load_model,
                }
            }
        }
    }
}

trait Client: Send + Sync + 'static {
    fn parse_path(&self, path: &PathBuf) -> BTreeMap<String, String>;

    fn list(
        &self,
        path: Option<&PathBuf>,
    ) -> Pin<Box<dyn Future<Output = Vec<(BTreeMap<String, String>, PathBuf)>> + Send>>;

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
struct PathTemplate {
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
                    Some(PathComponent::Identifier(part.to_string()))
                } else if part != "" {
                    Some(PathComponent::Fixed(part.to_string()))
                } else {
                    None
                }
            })
            .collect()
    }

    fn parse(&self, path: &PathBuf) -> BTreeMap<String, String> {
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
}

#[derive(Clone, Debug)]
pub struct FsClient {
    base_path: PathBuf,
    path_template: PathTemplate,
}

impl FsClient {
    pub fn parse_path(&self, path: &PathBuf) -> BTreeMap<String, String> {
        self.path_template.parse(path)
    }

    pub fn new(base_path: PathBuf, path_template: String) -> FsClient {
        FsClient {
            // TODO: Remove redundant `base_path` field somewhere
            base_path: base_path.clone(),
            path_template: PathTemplate { template: path_template, base: base_path },
        }
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
                Python::with_gil(|py| load_model.call(py, (data.into_py(py)?,), None))
            }
        }
    }
}

impl Client for ArtifactsClient {
    fn parse_path(&self, path: &PathBuf) -> BTreeMap<String, String> {
        match self {
            ArtifactsClient::FsClient { client, .. } => client.parse_path(path),
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
    ) -> Pin<Box<dyn Future<Output = Vec<(BTreeMap<String, String>, PathBuf)>> + Send>> {
        use ArtifactsClient::*;
        match self {
            FsClient { client, .. } => client.list(path),
        }
    }
}

fn _list(
    mut paths_by_spec: Vec<(BTreeMap<String, String>, PathBuf)>,
    mut remaining_components: VecDeque<PathComponent>,
) -> Vec<(BTreeMap<String, String>, PathBuf)> {
    match remaining_components.pop_front() {
        Some(PathComponent::Fixed(dir_name)) => {
            paths_by_spec.iter_mut().for_each(|(_, ref mut path)| {
                //
                path.push(dir_name.clone())
            });
            _list(paths_by_spec, remaining_components)
        }
        Some(PathComponent::Identifier(identifier)) => {
            //
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
    fn parse_path(&self, path: &PathBuf) -> BTreeMap<String, String> {
        self.parse_path(path)
    }

    fn list(
        &self,
        path: Option<&PathBuf>,
    ) -> Pin<Box<dyn Future<Output = Vec<(BTreeMap<String, String>, PathBuf)>> + Send>> {
        let spec = BTreeMap::<String, String>::default();
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
