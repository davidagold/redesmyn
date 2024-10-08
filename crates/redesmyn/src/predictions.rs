use crate::artifacts::BoxedSpec;
use crate::cache::{Cache, CacheHandle, CacheKey};
use crate::common::{build_runtime, OkOrLogErr, TOKIO_RUNTIME};
use crate::error::ServiceResult;
use crate::handler::Handler;
use crate::server::ResourceFactory;
use crate::{config_methods, metrics, validate_param};
use indexmap::IndexMap;
use redesmyn_macros::metric_instrument;

use super::error::ServiceError;
use super::schema::{Relation, Schema};

use actix_web::{web, HttpRequest, HttpResponse, Responder};
use polars::{frame::DataFrame, prelude::*};
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::{collections::HashMap, iter::repeat, time::Duration};
use tokio::time::sleep;
use tokio::{
    sync::{
        mpsc,
        oneshot::{self, Sender},
    },
    time::Instant,
};
use tracing::{error, info, instrument, warn};

use uuid::Uuid;

#[derive(Serialize)]
pub struct PredictionResponse {
    id: String,
    predictions: Vec<Option<f64>>,
}

pub trait ServiceCore {
    fn start(&mut self) -> ServiceResult<Box<dyn ResourceFactory>>;

    fn path(&self) -> String;
}

impl<T, R> ServiceCore for Endpoint<T, R>
where
    T: Send + Sync + Debug + for<'de> Deserialize<'de> + 'static,
    R: Relation<Serialized = T> + Send + Sync + 'static,
{
    #[instrument(skip_all)]
    fn start(&mut self) -> ServiceResult<Box<dyn ResourceFactory>> {
        let (tx_abort, rx_abort) = oneshot::channel::<()>();

        let Some(rx) = std::mem::take(&mut self.rx) else {
            return ServiceError::Error("Tried to start task from subordinate daemon.".to_string())
                .into();
        };

        let cache_handle = if let Some(cache) = self.cache() {
            cache.run().map_err(|err| ServiceError::from(err.to_string()))?;
            cache.handle().map_err(|err| ServiceError::from(err.to_string())).ok_or_log_err()
        } else {
            None
        };

        let config = self.config.clone();
        // TODO: Keep reference to this `JoinHandle`
        let handle = TOKIO_RUNTIME.get_or_init(build_runtime).spawn(async move {
            tokio::spawn(async move {
                Endpoint::<T, R>::task(rx, rx_abort, config, cache_handle).await
            });
            Endpoint::<T, R>::await_shutdown(tx_abort).await
        });

        self.state = EndpointState::Running;
        Ok(Box::new(self.handle()?))
    }

    fn path(&self) -> String {
        self.config.path.clone()
    }
}

pub trait Service: ServiceCore + Sized {
    type R;
    type T;
    type H;

    fn get_schema(&self) -> Option<&Schema>;

    fn get_path(&self) -> String;

    fn get_handler_fn(&self) -> Self::H;

    fn cache(&self) -> Option<&Cache>;

    fn handle(&self) -> ServiceResult<EndpointHandle<Self::T, Self::R>>
    where
        Self::T: Sync + Send + 'static,
        Self::R: Relation<Serialized = Self::T> + Sync + Send + 'static;
}

#[derive(Clone, Copy, Debug)]
enum EndpointState {
    Ready,
    Running,
}

#[derive(Clone, Debug)]
pub struct ServiceConfig {
    pub schema: Option<Schema>,
    pub path: String,
    pub batch_max_delay_ms: u32,
    pub batch_max_size: usize,
    pub handler: Handler,
    pub validate_artifact_params: bool,
}

#[derive(Debug)]
pub struct EndpointBuilder<T, R> {
    schema: Option<Schema>,
    path: Option<String>,
    batch_max_delay_ms: Option<u32>,
    batch_max_size: Option<usize>,
    handler: Option<Handler>,
    cache: Option<Cache>,
    validate_artifact_params: Option<bool>,
    _phantom: (PhantomData<T>, PhantomData<R>),
}

impl<T, R> Default for EndpointBuilder<T, R> {
    fn default() -> Self {
        EndpointBuilder {
            schema: None,
            path: None,
            batch_max_delay_ms: Some(5),
            batch_max_size: Some(64),
            handler: None,
            cache: None,
            validate_artifact_params: Some(false),
            _phantom: (PhantomData, PhantomData),
        }
    }
}

impl<T, R> EndpointBuilder<T, R>
where
    T: Send,
    R: Relation<Serialized = T> + Send + 'static,
{
    config_methods! {
        schema: Schema,
        path: &str,
        batch_max_delay_ms: u32,
        batch_max_size: usize,
        handler: Handler,
        cache: Cache,
        validate_artifact_params: bool
    }

    pub fn build(self) -> Result<Endpoint<T, R>, ServiceError> {
        let config = ServiceConfig {
            schema: self.schema.clone(),
            path: validate_param!(&self, path),
            batch_max_delay_ms: validate_param!(&self, batch_max_delay_ms),
            batch_max_size: validate_param!(&self, batch_max_size),
            handler: validate_param!(&self, handler),
            validate_artifact_params: validate_param!(&self, validate_artifact_params),
        };
        Ok(Endpoint::<T, R>::new(config, self.cache))
    }
}

#[derive(Debug)]
pub struct Endpoint<T, R>
where
    R: Relation<Serialized = T>,
{
    config: ServiceConfig,
    tx: mpsc::Sender<PredictionJob<T, R>>,
    rx: Option<mpsc::Receiver<PredictionJob<T, R>>>,
    // TODO: Include `RefCell<JoinHandle>` for task
    state: EndpointState,
    cache: Option<Arc<Cache>>,
}

impl<T, R> Endpoint<T, R>
where
    T: Send,
    R: Relation<Serialized = T> + Send + 'static,
{
    pub fn builder() -> EndpointBuilder<T, R> {
        EndpointBuilder::<T, R>::default()
    }

    pub fn set_config(&mut self, config: ServiceConfig) -> &Self {
        self.config = config;
        self
    }

    pub fn new(config: ServiceConfig, cache: Option<Cache>) -> Endpoint<T, R> {
        let (tx, rx) = mpsc::channel(1024);

        Endpoint {
            tx,
            rx: Some(rx),
            config,
            state: EndpointState::Ready,
            cache: cache.map(Arc::from),
        }
    }

    #[instrument(skip_all)]
    async fn task(
        mut rx: mpsc::Receiver<PredictionJob<T, R>>,
        mut rx_abort: oneshot::Receiver<()>,
        config: ServiceConfig,
        cache_handle: Option<CacheHandle>,
    ) {
        println!("Starting predict task with config: {:#?}", &config);
        let ServiceConfig { batch_max_delay_ms, batch_max_size, .. } = config.clone();
        loop {
            if rx_abort.try_recv().is_ok() {
                // TODO: Ensure that outstanding requests are handled gracefully.
                return;
            }

            let start = Instant::now();
            let duration_wait = Duration::new(0, batch_max_delay_ms * 1_000_000);
            // TODO: Find better way to yield
            let mut jobs = Vec::<PredictionJob<T, R>>::with_capacity(batch_max_size);
            sleep(Duration::new(0, 1_000_000)).await;
            while start.elapsed() <= duration_wait {
                if let Ok(job) = rx.try_recv() {
                    jobs.push(job);
                    if jobs.len() == batch_max_size {
                        break;
                    }
                }
            }
            if jobs.is_empty() {
                continue;
            }

            // TODO: Route jobs to key-specific batch-predict task rather than sort within latter
            let batches_by_key = batch_jobs_by_key(jobs);
            for (key, batch) in batches_by_key.into_iter() {
                let config = config.clone();
                let model = if let Some(handle) = &cache_handle {
                    match handle.get(&key).await {
                        Ok(model) => Some(model),
                        Err(err) => {
                            warn!("Failed to load model for spec with key '{}': {}", key, err);
                            continue;
                        }
                    }
                } else {
                    None
                };
                tokio::task::spawn_blocking(move || Self::predict_batch(batch, config, model));
            }
        }
    }

    #[instrument(skip_all)]
    fn predict_batch(
        mut batch: BatchJob<R>,
        config: ServiceConfig,
        model: Option<Py<PyAny>>,
    ) -> Result<(), ServiceError> {
        info!(batch_size = batch.len(), "Running batch predict.");
        let df_batch = batch.swap_df(None)?.unwrap();
        let df_results = match Python::with_gil(|py| {
            config
                .handler
                .invoke(PyDataFrame(df_batch), model)
                .inspect_err(|err| {
                    error!("Failed to call handler function `{}`: {err}", config.handler);
                })?
                .extract::<PyDataFrame>(py)
                .map(|pydf| -> DataFrame { pydf.into() })
        }) {
            Ok(df_results) => df_results,
            Err(err) => {
                // TODO: Handle errors! Because this runs in a separate blocking thread whose result
                //       we don't join, we cannot delegate sending failures to the caller.
                error!("{err}");
                Python::with_gil(|py| match err.traceback_bound(py) {
                    Some(traceback) => println!("{:#?}", traceback),
                    None => println!("No traceback."),
                });
                return Err(err.into());
            }
        };
        batch.send_responses(df_results).inspect_err(|err| error!("{}", err))?;
        Result::<(), ServiceError>::Ok(())
    }

    async fn await_shutdown(tx_abort: oneshot::Sender<()>) {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("Received shutdown signal.");
        if tx_abort.send(()).is_err() {
            tracing::error!("Failed to send cancel signal.");
        }
    }
}

/// An `EndpointHandle` allows access to `Endpoint` functionality
pub struct EndpointHandle<T, R>
where
    T: Send,
    R: Relation<Serialized = T>,
{
    tx: Arc<mpsc::Sender<PredictionJob<T, R>>>,
    schema: Option<Arc<Schema>>,
    cache_handle: Option<CacheHandle>,
    path: String,
    endpoint_config: ServiceConfig,
}

// We include this implementation so as not to impose `T: Clone`, `R: Clone`
impl<T, R> Clone for EndpointHandle<T, R>
where
    T: Send,
    R: Relation<Serialized = T>,
{
    fn clone(&self) -> Self {
        EndpointHandle {
            tx: self.tx.clone(),
            schema: self.schema.clone(),
            cache_handle: self.cache_handle.clone(),
            path: self.path.clone(),
            endpoint_config: self.endpoint_config.clone(),
        }
    }
}

impl<T, R> ResourceFactory for EndpointHandle<T, R>
where
    T: Debug + Send + Sync + for<'de> Deserialize<'de> + 'static,
    R: Send + Sync + Relation<Serialized = T> + 'static,
{
    fn new_resource(&mut self) -> Result<actix_web::Resource, ServiceError> {
        let resource = web::resource(self.path.clone())
            .app_data(web::Data::<EndpointHandle<T, R>>::new(self.clone()))
            .app_data(web::Data::<ServiceConfig>::new(self.endpoint_config.clone()))
            .route(web::post().to(invoke::<T, R>));
        Ok(resource)
    }

    fn clone_boxed(&self) -> Box<dyn ResourceFactory> {
        Box::new(self.clone())
    }
}

impl<T, R> EndpointHandle<T, R>
where
    T: Send,
    R: Relation<Serialized = T>,
{
    async fn submit_job(&self, job: PredictionJob<T, R>) -> ServiceResult<()> {
        self.tx.send(job).await.map_err(ServiceError::from)
    }

    fn schema(&self) -> Option<Arc<Schema>> {
        self.schema.clone()
    }
}

#[metric_instrument(dimensions(FunctionName = "Invoke"))]
pub async fn invoke<T, R>(
    req: HttpRequest,
    records: web::Json<Vec<T>>,
    service_handle: web::Data<EndpointHandle<T, R>>,
    endpoint_config: web::Data<ServiceConfig>,
) -> impl Responder
where
    T: Send + std::fmt::Debug,
    R: Relation<Serialized = T>,
{
    metrics!(RequestCount: Count = 1);

    // TODO: Does `match_info` guarantee preservation of identifier order?
    let spec: IndexMap<String, String> =
        req.match_info().iter().map(|(key, val)| (key.to_string(), val.to_string())).collect();

    if endpoint_config.validate_artifact_params {
        let Some(cache_handle) = &service_handle.cache_handle else {
            error!(
                "Artifact parameter validation is enabled but endpoint has no handle to its cache"
            );
            return HttpResponse::InternalServerError()
                .body("The request parameters could not be validated");
        };
        if let Some(Err(err)) = cache_handle.validate(&spec) {
            info!("Invalid request parameters: {}", err);
            return HttpResponse::UnprocessableEntity().body(err.to_string());
        }
    };
    let (tx, rx) = oneshot::channel();
    let job = PredictionJob::<T, R>::new(
        records.into_inner(),
        tx,
        service_handle.schema(),
        Box::new(spec),
    );

    if let Err(err) = service_handle.submit_job(job).await {
        return HttpResponse::InternalServerError().body(err.to_string());
    }
    match rx.await {
        Ok(Ok(response)) => HttpResponse::Ok().json(response),
        Ok(Err(err)) => HttpResponse::InternalServerError().body(err.to_string()),
        Err(err) => HttpResponse::InternalServerError().body(err.to_string()),
    }
}

pub(crate) type HandlerArgs<R, T> =
    (HttpRequest, web::Json<Vec<T>>, web::Data<EndpointHandle<T, R>>, web::Data<ServiceConfig>);

impl<T, R> Service for Endpoint<T, R>
where
    Self: Sync,
    T: Send + Sync + for<'de> Deserialize<'de> + std::fmt::Debug + 'static,
    R: Relation<Serialized = T> + Send + Sync + 'static,
{
    type R = R;
    type T = T;
    type H =
        impl actix_web::Handler<HandlerArgs<Self::R, Self::T>, Output = impl Responder + 'static>;

    fn get_schema(&self) -> Option<&Schema> {
        self.config.schema.as_ref()
    }

    fn get_path(&self) -> String {
        self.config.path.clone()
    }

    fn get_handler_fn(&self) -> Self::H {
        invoke::<Self::T, Self::R>
    }

    fn cache(&self) -> Option<&Cache> {
        self.cache.as_deref()
    }

    fn handle(&self) -> ServiceResult<EndpointHandle<Self::T, Self::R>>
    where
        Self::T: Sync + Send + 'static,
        Self::R: Relation<Serialized = Self::T> + Sync + Send + 'static,
    {
        Ok(EndpointHandle {
            tx: self.tx.clone().into(),
            schema: self.get_schema().map(|schema| schema.clone().into()),
            cache_handle: self.cache().and_then(|cache| cache.handle().ok_or_log_err()), // TODO: Consider failing fast here
            path: self.get_path(),
            endpoint_config: self.config.clone(),
        })
    }
}

struct BatchJob<R>
where
    R: Relation,
{
    model_key: CacheKey,
    jobs_by_id: HashMap<String, PredictionJob<R::Serialized, R>>,
    df: Option<DataFrame>,
}

impl<R> Debug for BatchJob<R>
where
    R: Relation,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("<BatchJob with model key {}", self.model_key))
    }
}

#[instrument(skip_all)]
fn batch_jobs_by_key<T, R>(
    jobs: impl IntoIterator<Item = PredictionJob<T, R>>,
) -> BTreeMap<CacheKey, BatchJob<R>>
where
    T: Send,
    R: Relation<Serialized = T>,
{
    let jobs_by_spec_key = jobs.into_iter().fold(
        BTreeMap::<CacheKey, Vec<PredictionJob<T, R>>>::default(),
        |mut jobs_by_key, job| {
            let Ok(key) = job.spec.as_key() else {
                return jobs_by_key;
            };
            jobs_by_key.entry(key).or_insert(Vec::<PredictionJob<T, R>>::new()).push(job);
            jobs_by_key
        },
    );
    jobs_by_spec_key
        .into_iter()
        .filter_map(|(key, jobs)| {
            let batch = match BatchJob::from_jobs(jobs, key.clone()) {
                Ok(batch) => batch,
                Err(err) => {
                    warn!("Failed to aggregate jobs: {:#?}", err);
                    return None;
                }
            };
            Some((key.clone(), batch))
        })
        .collect()
}

impl<T, R> BatchJob<R>
where
    T: Send,
    R: Relation<Serialized = T>,
{
    fn from_jobs(
        jobs: Vec<PredictionJob<R::Serialized, R>>,
        key: CacheKey,
    ) -> Result<BatchJob<R>, PolarsError> {
        let mut jobs_by_id = HashMap::<String, PredictionJob<R::Serialized, R>>::new();
        let dfs = jobs
            .into_iter()
            .filter_map(|mut job| match job.take_records_as_df() {
                Ok(df) => {
                    jobs_by_id.insert(job.id.into(), job);
                    Some(df.lazy())
                }
                Err(err) => {
                    job.send_result(Err(err));
                    None
                }
            })
            .collect::<Vec<_>>();
        let df_concatenated = concat::<Vec<_>>(dfs, UnionArgs::default())?;
        match df_concatenated.collect() {
            Ok(df) => Ok(BatchJob { jobs_by_id, df: Some(df), model_key: key }),
            Err(err) => Err(err),
        }
    }

    fn swap_df(&mut self, df: Option<DataFrame>) -> Result<Option<DataFrame>, ServiceError> {
        match (&df, &self.df) {
            (Some(_), None) => Ok(std::mem::replace(&mut self.df, df)),
            (None, Some(_)) => Ok(std::mem::replace(&mut self.df, df)),
            _ => Err(ServiceError::Error(
                "Cannot swap DataFrame when both `None` or both `Some`.".into(),
            )),
        }
    }

    #[instrument(skip_all)]
    fn send_responses(&mut self, df_results: DataFrame) -> Result<(), ServiceError> {
        let results = df_results.partition_by(["job_id"], true)?;
        for df in results {
            let Ok(Some(job_id)) = df.column("job_id").and_then(|s| s.str()).map(|s| s.get(0))
            else {
                error!("Failed to retrieve job ID from results DataFrame.");
                continue;
            };
            let Some(job) = self.jobs_by_id.remove(job_id) else {
                error!("Failed to retrieve job with ID {}", &job_id);
                continue;
            };
            let result = match df.column("prediction").and_then(|s| s.f64()).map(|a| a.to_vec()) {
                Ok(predictions) => Ok(PredictionResponse { id: job.id.to_string(), predictions }),
                Err(err) => Err(err.into()),
            };
            job.send_result(result);
        }
        Ok(())
    }

    fn len(&self) -> usize {
        self.jobs_by_id.len()
    }
}

// Parameters:
//  - T: serialized data
//  - R: Relation (can produce a Schema, can parse records into a DataFrame)
pub struct PredictionJob<T, R>
where
    R: Relation<Serialized = T>,
{
    id: Uuid,
    records: Option<Vec<T>>,
    tx: oneshot::Sender<Result<PredictionResponse, ServiceError>>,
    schema: Option<Arc<Schema>>,
    spec: BoxedSpec,
    phantom: PhantomData<R>,
}

impl<T, R> PredictionJob<T, R>
where
    R: Relation<Serialized = T>,
{
    fn new(
        records: Vec<T>,
        tx: Sender<Result<PredictionResponse, ServiceError>>,
        // TODO: We can make this a reference
        schema: Option<Arc<Schema>>,
        spec: BoxedSpec,
    ) -> PredictionJob<T, R>
    where
        R: Relation,
    {
        let id = Uuid::new_v4();
        PredictionJob {
            id,
            records: Some(records),
            tx,
            schema,
            spec: spec.into(),
            phantom: PhantomData,
        }
    }

    fn take_records_as_df(&mut self) -> Result<DataFrame, ServiceError>
    where
        T: Send,
        R: Relation<Serialized = T>,
    {
        let Some(records) = self.records.take() else {
            let msg = "Tried to take missing records".to_string();
            return Err(ServiceError::Error(msg));
        };
        let n_records = records.len();
        let mut df = R::parse(records, self.schema.as_ref())?;
        let col_job_id =
            Series::from_iter(repeat(self.id.to_string()).take(n_records)).with_name("job_id");
        match df.with_column(col_job_id) {
            Ok(_) => Ok(df),
            Err(err) => Err(err.into()),
        }
    }

    fn send_result(self, result: Result<PredictionResponse, ServiceError>) {
        if self.tx.send(result).is_err() {
            // TODO: Structure logging
            error!("Failed to send result for job with ID {}", self.id);
        }
    }
}
