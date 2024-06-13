use crate::cache::Cache;
use crate::handler::{Handler, HandlerConfig, PyHandler};
use crate::{config_methods, metrics, validate_param};
use redesmyn_macros::metric_instrument;

use super::error::ServiceError;
use super::schema::{Relation, Schema};

use actix_web::{web, HttpResponse, Responder};
use polars::{frame::DataFrame, prelude::*};
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::{collections::HashMap, fmt, iter::repeat, time::Duration};
use tokio::time::sleep;
use tokio::{
    sync::{
        mpsc,
        oneshot::{self, Sender},
    },
    task::JoinHandle,
    time::Instant,
};
use tracing::{error, info, instrument};

use uuid::Uuid;

#[derive(Serialize)]
pub struct PredictionResponse {
    id: String,
    predictions: Vec<Option<f64>>,
}

pub trait Service: Sized {
    type R;
    type T;
    type H;

    fn get_schema(&self) -> Schema;

    fn run(&mut self) -> Result<JoinHandle<()>, ServiceError>;

    fn get_path(&self) -> String;

    fn get_handler_fn(&self) -> Self::H;
}

#[derive(Clone, Copy, Debug)]
enum EndpointState {
    Ready,
    Running,
}

#[derive(Clone, Debug)]
pub struct ServiceConfig {
    pub schema: Schema,
    pub path: String,
    pub batch_max_delay_ms: u32,
    pub batch_max_size: usize,
    pub handler_config: HandlerConfig,
    pub handler: Option<Handler>,
    // pub cache: Cache,
}

impl ServiceConfig {
    fn try_init_handler(&mut self) -> Result<&Self, ServiceError> {
        self.handler = Some(Handler::Python(PyHandler::try_new(&self.handler_config)?));
        Ok(self)
    }
}

#[derive(Debug)]
pub struct EndpointBuilder<T, R> {
    schema: Option<Schema>,
    path: Option<String>,
    batch_max_delay_ms: Option<u32>,
    batch_max_size: Option<usize>,
    handler_config: Option<HandlerConfig>,
    cache: Option<Cache>,
    _phantom: (PhantomData<T>, PhantomData<R>),
}

impl<T, R> Default for EndpointBuilder<T, R> {
    fn default() -> Self {
        EndpointBuilder {
            schema: None,
            path: None,
            batch_max_delay_ms: Some(5),
            batch_max_size: Some(64),
            handler_config: None,
            cache: None,
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
        handler_config: HandlerConfig,
        cache: Cache
    }

    pub fn build(self) -> Result<BatchPredictor<T, R>, ServiceError> {
        let config = ServiceConfig {
            schema: validate_param!(&self, schema),
            path: validate_param!(&self, path),
            batch_max_delay_ms: validate_param!(&self, batch_max_delay_ms),
            batch_max_size: validate_param!(&self, batch_max_size),
            handler_config: validate_param!(&self, handler_config),
            handler: None,
        };
        Ok(BatchPredictor::<T, R>::new(config, self.cache.unwrap()))
    }
}

#[derive(Debug)]
pub struct BatchPredictor<T, R>
where
    R: Relation<Serialized = T>,
{
    config: ServiceConfig,
    tx: mpsc::Sender<PredictionJob<T, R>>,
    rx: Option<mpsc::Receiver<PredictionJob<T, R>>>,
    state: EndpointState,
    cache: Cache,
}

impl<T, R> Clone for BatchPredictor<T, R>
where
    Self: Sync,
    R: Relation<Serialized = T>,
{
    fn clone(&self) -> Self {
        let (tx, rx) = match self.state {
            EndpointState::Ready => {
                let (tx, rx) = mpsc::channel::<PredictionJob<T, R>>(1024);
                (tx, Some(rx))
            }
            EndpointState::Running => (self.tx.clone(), None),
        };
        BatchPredictor {
            config: self.config.clone(),
            tx,
            rx,
            state: self.state,
            cache: self.cache.clone(),
        }
    }
}

impl<T, R> BatchPredictor<T, R>
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

    pub fn new(config: ServiceConfig, cache: Cache) -> BatchPredictor<T, R> {
        let (tx, rx) = mpsc::channel(1024);

        BatchPredictor {
            tx,
            rx: Some(rx),
            config,
            state: EndpointState::Ready,
            cache,
        }
    }

    #[instrument(skip_all)]
    async fn task(
        mut rx: mpsc::Receiver<PredictionJob<T, R>>,
        mut rx_abort: oneshot::Receiver<()>,
        config: ServiceConfig,
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

            let config = config.clone();
            match BatchJob::from_jobs(jobs) {
                Ok(batch) => {
                    tokio::task::spawn_blocking(move || Self::predict_batch(batch, config))
                }
                Err(err) => {
                    error!("Failed to predict against batch: {err}");
                    continue;
                }
            };
        }
    }

    #[instrument(skip_all)]
    fn predict_batch(mut batch: BatchJob<R>, config: ServiceConfig) -> Result<(), ServiceError> {
        info!(batch_size = batch.len(), "Running batch predict.");
        let df_batch = batch.swap_df(None)?.unwrap();
        let df_results = match Python::with_gil(|py| {
            config
                .handler
                .ok_or_else(|| ServiceError::Error("Handler not initialized".to_string()))?
                .invoke(PyDataFrame(df_batch), Some(py))
                .inspect_err(|err| {
                    error!("Failed to call handler function `{:#?}`: {err}", config.handler_config);
                })?
                .extract::<PyDataFrame>(py)
                .map(|pydf| -> DataFrame { pydf.into() })
        }) {
            Ok(df_results) => df_results,
            Err(err) => {
                // TODO: Handle errors! Because this runs in a separate blocking thread whose result
                //       we don't join, we cannot delegate sending failures to the caller.
                error!("{err}");
                Python::with_gil(|py| match err.traceback(py) {
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

#[metric_instrument(dimensions(FunctionName = "Invoke"))]
pub async fn invoke<T, R>(
    records: web::Json<Vec<T>>,
    app_state: web::Data<BatchPredictor<T, R>>,
    schema: web::Data<Schema>,
) -> impl Responder
where
    T: Send + std::fmt::Debug,
    R: Relation<Serialized = T>,
{
    metrics!(RequestCount: Count = 1);

    let (tx, rx) = oneshot::channel();
    let job = PredictionJob::<T, R>::new(records.into_inner(), tx, schema.into_inner());

    if let Err(err) = app_state.tx.send(job).await {
        return HttpResponse::InternalServerError().body(err.to_string());
    }
    match rx.await {
        Ok(Ok(response)) => HttpResponse::Ok().json(response),
        Ok(Err(err)) => HttpResponse::InternalServerError().body(err.to_string()),
        Err(err) => HttpResponse::InternalServerError().body(err.to_string()),
    }
}

pub(crate) type HandlerArgs<R, T> =
    (web::Json<Vec<T>>, web::Data<BatchPredictor<T, R>>, web::Data<Schema>);

impl<T, R> Service for BatchPredictor<T, R>
where
    Self: Sync,
    T: Send + std::fmt::Debug + 'static,
    R: Relation<Serialized = T> + Send + 'static,
{
    type R = R;
    type T = T;
    type H =
        impl actix_web::Handler<HandlerArgs<Self::R, Self::T>, Output = impl Responder + 'static>;

    fn get_schema(&self) -> Schema {
        self.config.schema.clone()
    }

    fn run(&mut self) -> Result<JoinHandle<()>, ServiceError> {
        let (tx_abort, rx_abort) = oneshot::channel::<()>();

        let Some(rx) = std::mem::take(&mut self.rx) else {
            return ServiceError::Error("Tried to start task from subordinate daemon.".to_string())
                .into();
        };

        let mut config = self.config.clone();
        config.try_init_handler()?;
        let handle = tokio::spawn(async move {
            tokio::spawn(async move { BatchPredictor::<T, R>::task(rx, rx_abort, config).await });
            BatchPredictor::<T, R>::await_shutdown(tx_abort).await
        });
        self.state = EndpointState::Running;
        Ok(handle)
    }

    fn get_path(&self) -> String {
        self.config.path.clone()
    }

    fn get_handler_fn(&self) -> Self::H {
        invoke::<Self::T, Self::R>
    }
}

struct BatchJob<R>
where
    R: Relation,
{
    jobs_by_id: HashMap<String, PredictionJob<R::Serialized, R>>,
    df: Option<DataFrame>,
}

impl<T, R> BatchJob<R>
where
    T: Send,
    R: Relation<Serialized = T>,
{
    fn from_jobs(jobs: Vec<PredictionJob<R::Serialized, R>>) -> Result<BatchJob<R>, PolarsError> {
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
            Ok(df) => Ok(BatchJob { jobs_by_id, df: Some(df) }),
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
    schema: Arc<Schema>,
    phantom: PhantomData<R>,
}

impl<T, R> PredictionJob<T, R>
where
    R: Relation<Serialized = T>,
{
    fn new(
        records: Vec<T>,
        tx: Sender<Result<PredictionResponse, ServiceError>>,
        schema: Arc<Schema>,
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
        let mut df = R::parse(records, &self.schema)?;
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
