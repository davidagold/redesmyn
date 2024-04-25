use crate::{
    common::Wrap,
    error::{ServiceError, ServiceResult},
};
use bytes::{buf::Reader, Buf, BufMut, BytesMut};
use chrono::{DateTime, Duration, Utc};
use core::fmt;
use cron;
use futures::{
    channel::oneshot::{self, Canceled},
    TryFutureExt,
};
use lru::LruCache;
use polars::lazy::dsl::reduce_exprs;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    ffi::PyObject,
    pyclass, pymethods,
    types::PyFunction,
    FromPyObject, Py, PyAny, PyResult, Python,
};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    fs,
    future::{self, Future},
    num::NonZeroUsize,
    ops::Deref,
    path::PathBuf,
    pin::Pin,
    str::FromStr,
    sync::Arc,
};
use thiserror::Error;
use tokio::{
    sync::{
        mpsc::{
            self,
            error::{SendError, TrySendError},
        },
        Mutex,
    },
    task::{JoinError, JoinHandle},
};

enum Schedule {
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

// For ease of use, should consider:
// - Keeping this trait as an internal, renamed trait, exposing an `ArtifactSpec` trait
//   that implements the internal trait for all `Serialize` types,
// - Including a `derive` macro
trait ArtifactSpec<'a> {
    type Spec: Serialize;
    type Ref: Deref<Target = Self::Spec> + 'a;

    fn spec(&'a self) -> Self::Ref;
}

impl<'a> ArtifactSpec<'a> for BTreeMap<String, String> {
    type Spec = &'a Self;
    type Ref = Box<&'a Self>;

    fn spec(&'a self) -> Box<&'a Self> {
        self.into()
    }
}

impl<'a, T> ArtifactSpec<'a> for Wrap<T>
where
    T: ArtifactSpec<'a, Spec = T, Ref = &'a T> + Serialize + 'a,
{
    type Spec = T;
    type Ref = &'a T;

    fn spec(&'a self) -> &'a T {
        self.inner().spec()
    }
}

impl<'a, T> ArtifactSpec<'a> for Box<T>
where
    T: ArtifactSpec<'a, Spec = T, Ref = &'a T> + Serialize + 'a,
{
    type Spec = &'a T;
    type Ref = Box<&'a T>;

    fn spec(&'a self) -> Box<&'a T> {
        self.as_ref().spec().into()
    }
}

impl<S> std::fmt::Debug for dyn for<'a> ArtifactSpec<'a, Spec = S, Ref = &'a S>
where
    S: Serialize + for<'a> ArtifactSpec<'a>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!(
            "<impl ArtifactSpec, fields: {{ {:#?} }}>",
            serde_json::to_string(self.spec())
        ))
    }
}

type BoxedSpec = Box<
    dyn for<'a> ArtifactSpec<
            'a,
            Spec = &'a BTreeMap<String, String>,
            Ref = Box<&'a BTreeMap<String, String>>,
        > + Send
        + Sync,
>;

impl Wrap<&BoxedSpec> {
    fn try_as_key(&self) -> Result<CacheKey, serde_json::Error> {
        let mut writer = BytesMut::new().writer();
        serde_json::to_writer(&mut writer, self.inner().spec().as_ref());
        let spec: BTreeMap<String, String> = serde_json::from_reader(writer.into_inner().reader())?;
        let parts = spec.into_iter().map(|(k, v)| [k, v].join("/")).collect::<Vec<_>>();
        let key = parts.join("/");
        Ok(key)
    }

    fn try_as_path(&self) -> Result<PathBuf, serde_json::Error> {}
}

impl Serialize for Wrap<BTreeMap<String, String>> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

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
            &Uri::Path(Some(path)) => f.write_fmt(format_args!("<Uri::Path path = '{:#?}'>", path)),
            &Uri::Path(None) => f.write_fmt(format_args!("<Uri::Path path = unset'>")),
            &Uri::Id { extractor: _, id: Some(id_str) } => {
                f.write_fmt(format_args!("<Uri::Id id = '{:#?}'>", id_str))
            }
            &Uri::Id { extractor: _, id: None } => {
                f.write_fmt(format_args!("<Uri::Id in = unset'>"))
            }
        }
    }
}

impl Uri {
    fn parse(self, path: PathBuf) -> Self {
        match self {
            Uri::Path(None) => Uri::Path(Some(path)),
            Uri::Id { extractor: Some(func), id: None } => {
                let id = func(path);
                Uri::Id { extractor: None, id: Some(id.clone()) }
            }
            _ => panic!(),
        }
    }
}

#[derive(Debug)]
enum FetchAs {
    Uri(Option<Uri>),
    Bytes(Option<BytesMut>),
    Utf8String(Option<String>),
    TmpFile(Option<tokio::fs::File>),
}

impl Clone for FetchAs {
    fn clone(&self) -> Self {
        use FetchAs::*;
        match self {
            Uri(maybe_uri) => FetchAs::Uri(maybe_uri.clone()),
            Bytes(maybe_bytes) => FetchAs::Bytes(maybe_bytes.clone()),
            Utf8String(maybe_s) => FetchAs::Utf8String(maybe_s.clone()),
            // Not exactly sure what to do with this one.
            TmpFile(_) => FetchAs::TmpFile(None),
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
    fn new_empty_not_none(fetch_as: &FetchAs) -> FetchAs {
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
            &FetchAs::Utf8String(_) => FetchAs::Utf8String(Some(String::new())),
            &FetchAs::TmpFile(_) => FetchAs::TmpFile(None),
        }
    }
}

enum Command {
    UpdateEntry(BoxedSpec, FetchAs, oneshot::Sender<Py<PyAny>>),
    InsertEntry(CacheKey, FetchAs, oneshot::Sender<Result<(), CacheError>>),
    GetEntry(BoxedSpec, oneshot::Sender<Py<PyAny>>),
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
        }
    }
}

type CacheKey = String;

#[derive(Clone, Debug)]
struct RefreshConfig {
    key: String,
    path: PathBuf,
    last_updated: Option<DateTime<Utc>>,
    tx: Arc<mpsc::Sender<Command>>,
    client: Arc<ArtifactsClient>,
    fetch_as: FetchAs,
}

impl RefreshConfig {
    fn new(
        key: CacheKey,
        path: PathBuf,
        last_updated: Option<DateTime<Utc>>,
        client: Arc<ArtifactsClient>,
        tx: Arc<mpsc::Sender<Command>>,
        fetch_as: FetchAs,
    ) -> RefreshConfig {
        RefreshConfig {
            key,
            path,
            last_updated,
            tx,
            client,
            fetch_as,
        }
    }
}

struct RefreshTask<T: RefreshState + Send> {
    config: RefreshConfig,
    state: T,
}

impl<T: RefreshState + Send> RefreshTask<T> {
    fn is_done(&self) -> bool {
        self.state.refresh_complete()
    }
}

impl RefreshTask<PendingFetch> {
    fn new(
        schedule: &Schedule,
        config: RefreshConfig,
        fetch_now: bool,
    ) -> Result<RefreshTask<PendingFetch>, CacheError> {
        let RefreshConfig { last_updated, .. } = &config;
        let next_update =
            if fetch_now { UpdateTime::Now } else { schedule.next_update(last_updated) };
        let state = PendingFetch::new(next_update)?;
        Ok(RefreshTask { config, state })
    }
}

trait Refresh {
    type State: RefreshState + Send;
}

impl<T: Send + RefreshState> Refresh for RefreshTask<T> {
    type State = T;
}

trait RefreshState {
    fn refresh_complete(&self) -> bool {
        false
    }
}

struct PendingFetch {
    next_update: UpdateTime,
    task: JoinHandle<ServiceResult<()>>,
}
impl RefreshState for PendingFetch {}

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

struct FetchingData {
    task: JoinHandle<Result<FetchAs, CacheError>>,
}
impl RefreshState for FetchingData {}

struct UpdatingCache {
    task: JoinHandle<Result<(), CacheError>>,
}
impl RefreshState for UpdatingCache {}

struct RefreshComplete {}

impl RefreshState for RefreshComplete {
    fn refresh_complete(&self) -> bool {
        true
    }
}

trait Transition {
    // type Current;
    type Next: RefreshState + Send;
    type Error;

    // fn new<T, S>(config: RefreshConfig, state: S) -> RefreshTask<Self::Current>
    // where
    //     T: Transition<Current = S, Next = Self::Current>
    // {

    // }

    fn into_next(self) -> impl Future<Output = Result<RefreshTask<Self::Next>, Self::Error>>;
}

impl Transition for RefreshTask<PendingFetch> {
    type Next = FetchingData;
    type Error = CacheError;

    async fn into_next(self) -> Result<RefreshTask<FetchingData>, CacheError> {
        let RefreshTask { config, state: _ } = self;
        let RefreshConfig { client, path, .. } = config.clone();

        let container = FetchAs::new_empty_not_none(&config.fetch_as);
        let fut = client.fetch(path, container).into_future();
        let task = tokio::spawn(async move { fut.await });
        Ok(RefreshTask { config, state: FetchingData { task } })
    }
}

impl Transition for RefreshTask<FetchingData> {
    type Next = UpdatingCache;
    type Error = CacheError;

    async fn into_next(self) -> Result<RefreshTask<UpdatingCache>, CacheError> {
        let RefreshTask { config, state } = self;
        let Ok(data) = state.task.await? else {
            let msg = "Failed to receive artifact data from `RefreshTask<FetchingData>`";
            return Err(CacheError::from(msg));
        };
        let (tx, rx) = oneshot::channel::<Result<(), CacheError>>();
        let cmd = Command::InsertEntry(config.key.clone(), data, tx);
        config.tx.send(cmd).await?;

        let task = tokio::spawn(async move {
            match rx.await {
                Ok(res) => res,
                Err(err) => Err(CacheError::from(err.to_string())),
            }
        });
        Ok(RefreshTask { config, state: UpdatingCache { task } })
    }
}

struct Cache {
    schedule: Schedule,
    load_model: Py<PyFunction>,
    tx: Arc<mpsc::Sender<Command>>,
    task: JoinHandle<Result<(), CacheError>>,
}

type RefreshEntry = Box<dyn Refresh<State = dyn RefreshState + Send> + Send>;

impl Cache {
    pub fn new(
        max_size: Option<usize>,
        load_model: Py<PyFunction>,
        schedule: Option<Schedule>,
    ) -> Cache {
        let default_cache_size = 128;
        let model_cache: Arc<Mutex<LruCache<CacheKey, (RefreshEntry, Py<PyAny>)>>> = Arc::new(
            LruCache::new(NonZeroUsize::new(max_size.unwrap_or(default_cache_size)).unwrap())
                .into(),
        );

        let (tx, rx) = mpsc::channel::<Command>(max_size.unwrap_or(default_cache_size));
        let task = tokio::spawn(Cache::task(model_cache, rx));

        Cache {
            schedule: schedule.unwrap_or_default(),
            load_model,
            tx: tx.into(),
            task,
        }
    }

    async fn task(
        model_cache: Arc<Mutex<LruCache<CacheKey, (RefreshEntry, Py<PyAny>)>>>,
        mut rx: mpsc::Receiver<Command>,
    ) -> Result<(), CacheError> {
        loop {
            let Some(msg) = rx.recv().await else { return Err(CacheError::from("")) };
            match msg {
                Command::UpdateEntry(spec, fetch_as, tx) => {

                    // The following does not refresh the cache entry directly, but rather spawns a new task
                    // to handle data fetching for the given key and tracks the task's association with said key.
                    // self.refresh(key).await;
                    //
                    // Will need to get the cache entry, lock the key, get last updated, pass everything
                    // to `refresh_entry`
                }
                Command::InsertEntry(key, payload, tx) => {}
                Command::GetEntry(key, tx) => {}
            }
        }
    }

    async fn refresh_entry(
        &self,
        client: &Arc<ArtifactsClient>,
        spec: BoxedSpec,
        last_updated: Option<DateTime<Utc>>,
        fetch_as: FetchAs,
        fetch_now: bool,
    ) -> Result<(), CacheError> {
        let config = RefreshConfig {
            client: client.clone(),
            key: Wrap(&spec).try_as_key()?,
            fetch_as,
            path: Wrap(&spec).try_as_path()?,
            last_updated,
            tx: self.tx.clone(),
        };
        let result = RefreshTask::new(&self.schedule, config, fetch_now)?
            .into_next()
            .await?
            .into_next()
            .await?;

        // while Refre
        Ok(())
    }

    fn try_send(&self, command: Command) -> Result<(), CacheError> {
        self.tx.try_send(command).map_err(|err| err.into())
    }
}

#[derive(Error, Debug)]
pub enum CacheError {
    #[error("Error: {}", 0)]
    Error(String),
    #[error("Failed to send command {:#?}", 0.0)]
    SendCommandError(#[from] SendError<Command>),
    #[error("Failed to send command {:#?} ({})", 0.0, 0)]
    TrySendCommandError(#[from] TrySendError<Command>),
    #[error("CacheError: Failed to receive response (channel closed)")]
    ReceiveResponseError(#[from] Canceled),
    #[error("Error while awaiting step: {}", 0.to_string())]
    JoinError(#[from] JoinError),
    #[error("Error serializing `ArtifactSpec`: {}", 0.to_string())]
    SerializeError(#[from] serde_json::Error),
    #[error("IO Error: {}", 0.to_string())]
    IoError(#[from] std::io::Error),
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

#[pyclass]
struct PyCache {
    cache: Cache,
    client: ArtifactsClient,
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
            cache: Cache::new(max_size, load_model.into(), schedule),
            client: client_spec.to_client(),
        })
    }

    #[pyo3(signature = (**kwargs))]
    fn get<'py>(
        &'py self,
        py: Python<'py>,
        kwargs: Option<BTreeMap<String, String>>,
    ) -> PyResult<&'py PyAny> {
        let (tx, rx) = oneshot::channel::<Py<PyAny>>();
        let spec = kwargs.unwrap_or_default();
        let cmd = Command::GetEntry(Box::new(spec), tx);
        self.cache.try_send(cmd).map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        pyo3_asyncio::tokio::future_into_py(py, async move {
            let model = rx.await.map_err(|err| {
                PyRuntimeError::new_err(format!("Failed to receive response from cache: {err}"))
            })?;
            Ok(model)
        })
    }
}

#[derive(FromPyObject)]
enum ClientSpec {
    FsClient { protocol: String },
}

impl ClientSpec {
    fn to_client(self) -> ArtifactsClient {
        match self {
            ClientSpec::FsClient { protocol } => ArtifactsClient::FsClient(FsClient {}),
        }
    }
}

trait Client: Send + Sync + 'static {
    fn list(
        &self,
        path: PathBuf,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<PathBuf>, CacheError>> + Send + 'static>>;

    fn fetch_uri(
        &self,
        path: PathBuf,
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
struct FsClient {}

impl Default for FsClient {
    fn default() -> Self {
        FsClient {}
    }
}

#[derive(Clone, Debug)]
enum ArtifactsClient {
    FsClient(FsClient),
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
                Box::pin(client.fetch_uri(path, uri).and_then(|uri| async { Ok(uri.into()) }))
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
}

impl Client for ArtifactsClient {
    fn fetch_bytes(
        &self,
        path: PathBuf,
        bytes: BytesMut,
    ) -> Pin<Box<dyn Future<Output = Result<BytesMut, CacheError>> + Send + 'static>> {
        use ArtifactsClient::*;
        match self {
            FsClient(client) => client.fetch_bytes(path, bytes),
        }
    }

    fn list(
        &self,
        path: PathBuf,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<PathBuf>, CacheError>> + Send>> {
        use ArtifactsClient::*;
        match self {
            FsClient(client) => client.list(path),
        }
    }
}

impl Client for FsClient {
    fn list(
        &self,
        path: PathBuf,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<PathBuf>, CacheError>> + Send>> {
        match (|| {
            let items = fs::read_dir(path)?;
            let iter_paths = items.filter_map(|item| Some(item.ok()?.path()));
            Ok(iter_paths.collect())
        })() {
            Ok(paths) => Box::pin(future::ready(Ok(paths))),
            Err(err) => return Box::pin(future::ready(Err(err))),
        }
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
