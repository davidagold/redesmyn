use crate::{
    common::Wrap,
    error::{ServiceError, ServiceResult},
};
use bytes::{Buf, BufMut, BytesMut};
use chrono::{DateTime, Duration, Utc};
use core::fmt;
use cron;
use futures::TryFutureExt;
use lru::LruCache;
use pyo3::{
    exceptions::PyValueError, pyclass, pymethods, types::PyFunction, FromPyObject, Py, PyAny,
    PyResult,
};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    fs,
    future::{self, Future},
    num::NonZeroUsize,
    path::PathBuf,
    pin::Pin,
    str::FromStr,
    sync::Arc,
};
use tokio::{
    sync::{mpsc, Mutex},
    task::JoinHandle,
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

trait ArtifactSpec {
    type Spec: Serialize;

    fn spec(&self) -> &Self::Spec;
}

// TODO: We will probably want to make the original `Box` a reference type
impl From<Box<dyn ArtifactSpec<Spec = Box<dyn erased_serde::Serialize + Send>> + Send>>
    for Wrap<Result<CacheKey, serde_json::Error>>
{
    fn from(
        dyn_spec: Box<dyn ArtifactSpec<Spec = Box<dyn erased_serde::Serialize + Send>> + Send>,
    ) -> Self {
        {
            let mut writer = BytesMut::new().writer();
            serde_json::to_writer(&mut writer, dyn_spec.spec());
            let spec: BTreeMap<String, String> = {
                match serde_json::from_reader(writer.into_inner().reader()) {
                    Ok(spec) => spec,
                    Err(err) => return Wrap(Err(err)),
                }
            };
            let parts = spec.into_iter().map(|(k, v)| [k, v].join("/")).collect::<Vec<_>>();
            let key = parts.join("/");
            Wrap(Ok(key))
        }
    }
}

impl Serialize for Wrap<BTreeMap<String, String>> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl ArtifactSpec for Wrap<BTreeMap<String, String>> {
    type Spec = BTreeMap<String, String>;

    fn spec(&self) -> &Self::Spec {
        &self.0
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

enum Message {
    FetchLatest(Box<dyn ArtifactSpec<Spec = Box<dyn erased_serde::Serialize + Send>> + Send>),
    UpdateEntry(CacheKey, FetchAs),
    GetEntry(Box<dyn ArtifactSpec<Spec = Box<dyn erased_serde::Serialize + Send>> + Send>),
}

type CacheKey = String;

#[derive(Clone, Debug)]
struct RefreshConfig {
    key: String,
    path: PathBuf,
    last_updated: Option<DateTime<Utc>>,
    tx: mpsc::Sender<Message>,
    client: Arc<ArtifactsClient>,
    fetch_as: FetchAs,
}

impl RefreshConfig {
    fn new(
        key: CacheKey,
        path: PathBuf,
        last_updated: Option<DateTime<Utc>>,
        client: Arc<ArtifactsClient>,
        tx: mpsc::Sender<Message>,
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

struct RefreshTask<T> {
    config: RefreshConfig,
    state: T,
}

impl RefreshTask<PendingFetch> {
    fn new(
        schedule: &Schedule,
        config: RefreshConfig,
        fetch_now: bool,
    ) -> ServiceResult<RefreshTask<PendingFetch>> {
        let RefreshConfig { last_updated, .. } = &config;
        let next_update =
            if fetch_now { UpdateTime::Now } else { schedule.next_update(last_updated) };
        let state = PendingFetch::new(next_update)?;
        Ok(RefreshTask { config, state })
    }
}

trait Refresh {
    type State: Send;
}

impl<T: Send> Refresh for RefreshTask<T> {
    type State = T;
}

trait RefreshState {}

struct PendingFetch {
    next_update: UpdateTime,
    handle: JoinHandle<ServiceResult<()>>,
}
impl RefreshState for PendingFetch {}

impl PendingFetch {
    fn new(next_update: UpdateTime) -> ServiceResult<PendingFetch> {
        let wait_task: Pin<Box<dyn Future<Output = ServiceResult<()>> + Send>> = match next_update {
            UpdateTime::Now => Box::pin(future::ready(Ok(()))),
            UpdateTime::DateTime(dt) => {
                let wait_seconds = (dt - Utc::now()).num_seconds();
                let wait_duration =
                    Duration::seconds(wait_seconds).to_std().map_err(|err| err.to_string())?;

                Box::pin(async move {
                    tokio::time::sleep(wait_duration).await;
                    ServiceResult::Ok(())
                })
            }
        };
        let state = PendingFetch {
            next_update: next_update.clone(),
            handle: tokio::spawn(async move { wait_task.await }),
        };
        Ok(state)
    }
}

struct FetchingData {
    handle: JoinHandle<ServiceResult<FetchAs>>,
}

struct UpdatingCache {
    data: FetchAs,
    handle: JoinHandle<ServiceResult<()>>,
}

trait Transition<U, V> {
    fn into(self) -> impl Future<Output = ServiceResult<(RefreshConfig, V)>>;
}

impl From<RefreshTask<PendingFetch>> for RefreshTask<FetchingData> {
    fn from(task: RefreshTask<PendingFetch>) -> Self {
        let RefreshTask { config, state: _ } = task;
        let RefreshConfig { client, path, .. } = config.clone();

        let container = FetchAs::new_empty_not_none(&config.fetch_as);
        let fut = client.fetch(path, container).into_future();
        let handle = tokio::spawn(async move { fut.await });
        RefreshTask { config, state: FetchingData { handle } }
    }
}

struct Cache {
    // model_cache: Arc<Mutex<LruCache<CacheKey, (RefreshTask, Py<PyAny>)>>>,
    schedule: Schedule,
    load_model: Py<PyFunction>,
    tx: Arc<mpsc::Sender<Message>>,
    task: JoinHandle<ServiceResult<()>>,
}

type RefreshEntry = Box<dyn Refresh<State = dyn RefreshState + Send> + Send>;

impl Cache {
    async fn task(
        model_cache: Arc<Mutex<LruCache<CacheKey, (RefreshEntry, Py<PyAny>)>>>,
        mut rx: mpsc::Receiver<Message>,
    ) -> ServiceResult<()> {
        loop {
            let Some(msg) = rx.recv().await else { return ServiceError::from("").into() };
            match msg {
                Message::FetchLatest(dyn_spec) => {
                    let Wrap(maybe_key): Wrap<Result<CacheKey, serde_json::Error>> =
                        dyn_spec.into();

                    // The following does not refresh the cache entry directly, but rather spawns a new task
                    // to handle data fetching for the given key and tracks the task's association with said key.
                    // self.refresh(key).await;
                }
                Message::UpdateEntry(key, payload) => {}
                Message::GetEntry(key) => {}
            }
        }
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
        let schedule: Schedule = match (cron_expr, max_age_seconds) {
            (Some(expr), None) => cron::Schedule::from_str(expr.as_str())
                .map_err(|err| PyValueError::new_err(err.to_string()))?
                .into(),
            (None, Some(seconds)) => seconds.into(),
            _ => return Err(PyValueError::new_err("Something went terribly wrong.")),
        };

        let model_cache: Arc<Mutex<LruCache<CacheKey, (RefreshEntry, Py<PyAny>)>>> =
            Arc::new(LruCache::new(NonZeroUsize::new(max_size.unwrap_or(128)).unwrap()).into());
        let (tx, rx) = mpsc::channel::<Message>(128);
        let task = tokio::spawn(Cache::task(model_cache.clone(), rx));
        let cache = Cache {
            // model_cache,
            schedule,
            load_model: load_model.into(),
            tx: tx.into(),
            task,
        };
        Ok(PyCache { cache, client: client_spec.to_client() })
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
    fn list(&self, path: PathBuf) -> Pin<Box<dyn Future<Output = ServiceResult<Vec<PathBuf>>>>>;

    fn fetch_uri(
        &self,
        path: PathBuf,
        uri: Uri,
    ) -> Pin<Box<dyn Future<Output = ServiceResult<Uri>> + Send + Sync + 'static>> {
        Box::pin(future::ready(Ok(uri.parse(path))))
    }

    fn fetch_bytes(
        &self,
        path: PathBuf,
        bytes: BytesMut,
    ) -> Pin<Box<dyn Future<Output = ServiceResult<BytesMut>> + Send + 'static>>;
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
    ) -> Pin<Box<dyn Future<Output = ServiceResult<FetchAs>> + Send + 'static>> {
        // TODO: This may be fine for now but we should wrap in an `Arc``
        let client = self.clone();
        match fetch_as {
            FetchAs::Uri(Some(uri)) => {
                Box::pin(client.fetch_uri(path, uri).and_then(|uri| async { Ok(uri.into()) }))
            }
            FetchAs::Bytes(Some(bytes)) => {
                if !bytes.is_empty() {
                    return Box::pin(future::ready(Err(ServiceError::from("Bytes must be empty"))));
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
    ) -> Pin<Box<dyn Future<Output = ServiceResult<BytesMut>> + Send + 'static>> {
        use ArtifactsClient::*;
        match self {
            FsClient(client) => client.fetch_bytes(path, bytes),
        }
    }

    fn list(&self, path: PathBuf) -> Pin<Box<dyn Future<Output = ServiceResult<Vec<PathBuf>>>>> {
        use ArtifactsClient::*;
        match self {
            FsClient(client) => client.list(path),
        }
    }
}

impl Client for FsClient {
    fn list(&self, path: PathBuf) -> Pin<Box<dyn Future<Output = ServiceResult<Vec<PathBuf>>>>> {
        match (|| {
            let items = fs::read_dir(path)?;
            let iter_paths = items.filter_map(|item| Some(item.ok()?.path()));
            ServiceResult::Ok(iter_paths.collect())
        })() {
            Ok(paths) => Box::pin(future::ready(ServiceResult::Ok(paths))),
            Err(err) => return Box::pin(future::ready(Err(err))),
        }
    }

    fn fetch_bytes(
        &self,
        path: PathBuf,
        mut bytes: BytesMut,
    ) -> Pin<Box<dyn Future<Output = ServiceResult<BytesMut>> + Send + 'static>> {
        Box::pin(async move {
            bytes.extend(tokio::fs::read(path).await?);
            Ok(bytes)
        })
    }
}
