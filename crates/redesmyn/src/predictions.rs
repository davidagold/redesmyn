use super::error::ServiceError;

use actix_web::{web, Handler, HttpResponse, Responder};
use polars::{frame::DataFrame, prelude::*};
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fmt, iter::repeat, time::Duration};
use tokio::time::sleep;
use tokio::{
    sync::{
        mpsc,
        oneshot::{self, Sender},
        Mutex,
    },
    task::JoinHandle,
    time::Instant,
};
use tracing::{error, info, instrument};
use uuid::Uuid;

pub trait Schema<R> {
    fn to_dataframe(records: Vec<R>) -> PolarsResult<DataFrame>
    where
        Self: Sized;
}

#[derive(Serialize)]
pub struct PredictionResponse {
    id: String,
    predictions: Vec<Option<f64>>,
}

#[derive(Clone, Copy, Debug)]
pub struct ServiceConfig {
    pub(super) path: &'static str,
    pub(super) batch_max_delay_ms: u32,
    pub(super) batch_max_capacity: usize,
    pub(super) handler: &'static str,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        ServiceConfig {
            path: "predictions/{model_name}/{model_version}",
            batch_max_delay_ms: 5,
            batch_max_capacity: 1024,
            handler: "handlers.model:handle",
        }
    }
}

// TODO: macro for this.
pub trait Configurable: Sized {
    fn config(&mut self, config: Option<ServiceConfig>) -> ServiceConfig;

    fn path(mut self, path: &'static str) -> Self {
        let mut config = self.config(None);
        config.path = path;
        self.config(Some(config));
        self
    }

    fn batch_max_delay_ms(mut self, batch_max_delay_ms: u32) -> Self {
        let mut config = self.config(None);
        config.batch_max_delay_ms = batch_max_delay_ms;
        self.config(Some(config));
        self
    }

    fn batch_max_capacity(mut self, batch_max_capacity: usize) -> Self {
        let mut config = self.config(None);
        config.batch_max_capacity = batch_max_capacity;
        self.config(Some(config));
        self
    }

    fn handler(mut self, handler: &'static str) -> Self {
        let mut config = self.config(None);
        config.handler = handler;
        self.config(Some(config));
        self
    }
}

impl Configurable for ServiceConfig {
    fn config(&mut self, config: Option<ServiceConfig>) -> ServiceConfig {
        match config {
            Some(new_config) => new_config,
            None => *self,
        }
    }
}

pub trait Service: Sized + Configurable {
    type R;
    type H;

    fn run(&mut self) -> Result<JoinHandle<()>, ServiceError>;

    fn get_handler(&self) -> Self::H;
}

pub struct BatchPredictor<R>
where
    Self: Sync,
{
    tx: Arc<Mutex<mpsc::Sender<PredictionJob<R>>>>,
    handle: ServiceHandle<R>,
}

impl<R> Clone for BatchPredictor<R>
where
    R: Schema<R> + Sync + Send + 'static + for<'a> Deserialize<'a>,
{
    fn clone(&self) -> Self {
        BatchPredictor {
            tx: self.tx.clone(),
            handle: ServiceHandle {
                rx: None,
                config: self.handle.config,
            },
        }
    }
}

impl<R> Configurable for BatchPredictor<R>
where
    R: Schema<R> + Sync + Send + 'static + for<'a> Deserialize<'a>,
{
    fn config(&mut self, config: Option<ServiceConfig>) -> ServiceConfig {
        match config {
            Some(new_config) => {
                self.handle.config = new_config;
                new_config
            }
            None => self.handle.config,
        }
    }
}

pub(crate) type HandlerArgs<R> =
    (web::Path<ModelSpec>, web::Json<Vec<R>>, web::Data<BatchPredictor<R>>);

impl<R> Service for BatchPredictor<R>
where
    R: Schema<R> + Sync + Send + 'static + for<'a> Deserialize<'a>,
{
    type R = R;
    type H = impl Handler<HandlerArgs<Self::R>, Output = impl Responder + 'static>;

    fn run(&mut self) -> Result<JoinHandle<()>, ServiceError> {
        match self.handle.rx {
            Some(_) => Ok(self.handle.start()),
            None => Err(ServiceError::Error("Cannot start service task from clone".to_string())),
        }
    }

    fn get_handler(&self) -> Self::H {
        invoke::<R>
    }
}

#[derive(Deserialize)]
pub struct ModelSpec {
    model_name: String,
    model_version: String,
}

impl fmt::Display for ModelSpec {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}:{}", self.model_name, self.model_version)
    }
}

pub async fn invoke<'de, R>(
    model_spec: web::Path<ModelSpec>,
    records: web::Json<Vec<R>>,
    app_state: web::Data<BatchPredictor<R>>,
) -> impl Responder
where
    R: Schema<R> + Sync + Send + 'static,
    ModelSpec: serde::Deserialize<'de>,
{
    let ModelSpec { model_name, model_version } = &*model_spec;
    // info!(%model_name, %model_version);

    let (tx, rx) = oneshot::channel();
    let job = PredictionJob::new(records.into_inner(), tx);

    if let Err(err) = app_state.tx.lock().await.send(job).await {
        return HttpResponse::InternalServerError().body(err.to_string());
    }
    match rx.await {
        Ok(Ok(response)) => HttpResponse::Ok().json(response),
        Ok(Err(err)) => HttpResponse::InternalServerError().body(err.to_string()),
        Err(err) => HttpResponse::InternalServerError().body(err.to_string()),
    }
}

struct ServiceHandle<R> {
    rx: Option<mpsc::Receiver<PredictionJob<R>>>,
    config: ServiceConfig,
}

impl<R> ServiceHandle<R>
where
    R: Schema<R> + Sync + Send + 'static + for<'a> Deserialize<'a>,
{
    fn start(&mut self) -> JoinHandle<()> {
        let rx = self.rx.take().unwrap();
        let (tx_abort, rx_abort) = oneshot::channel::<()>();

        let config = self.config;
        tokio::spawn(async move {
            tokio::spawn(async move { BatchPredictor::<R>::task(rx, rx_abort, config).await });
            BatchPredictor::<R>::await_shutdown(tx_abort).await
        })
    }
}

impl<R> Default for BatchPredictor<R>
where
    R: Schema<R> + Sync + Send + 'static + for<'a> Deserialize<'a>,
 {
    fn default() -> Self {
        Self::new()
    }
}

impl<R> BatchPredictor<R>
where
    R: Schema<R> + Sync + Send + 'static + for<'a> Deserialize<'a>,
{
    pub fn new() -> BatchPredictor<R> {
        let (tx, rx) = mpsc::channel(1024);

        BatchPredictor {
            tx: Arc::new(tx.into()),
            handle: ServiceHandle {
                rx: Some(rx),
                config: ServiceConfig::default(),
            },
        }
    }

    #[instrument(skip_all)]
    async fn task(
        mut rx: mpsc::Receiver<PredictionJob<R>>,
        mut rx_abort: oneshot::Receiver<()>,
        config: ServiceConfig,
    ) {
        println!("Starting predict task with config: {:?}", config);
        loop {
            if rx_abort.try_recv().is_ok() {
                // TODO: Ensure that outstanding requests are handled gracefully.
                return;
            }

            let start = Instant::now();
            let duration_wait = Duration::new(0, config.batch_max_delay_ms * 1_000_000);
            let mut jobs = Vec::<PredictionJob<R>>::with_capacity(config.batch_max_capacity);
            // TODO: Find better way to yield
            sleep(Duration::new(0, 1_000_000)).await;
            while start.elapsed() <= duration_wait {
                if let Ok(job) = rx.try_recv() {
                    info!("{}", jobs.len());
                    jobs.push(job);
                    if jobs.len() == config.batch_max_capacity {
                        break;
                    }
                }
            }
            if jobs.is_empty() {
                continue;
            }

            match BatchJob::from_jobs(jobs) {
                Ok(batch) => {
                    tokio::task::spawn_blocking(move || Self::predict_batch(batch, &config.clone()))
                }
                Err(err) => {
                    error!("Failed to {err}");
                    continue;
                }
            };
        }
    }

    #[instrument(skip_all)]
    fn predict_batch(mut batch: BatchJob<R>, config: &ServiceConfig) -> Result<(), ServiceError> {
        info!("Running batch predict for {} jobs.", batch.len());

        let Some((handler_module, handler_fn)) = config.handler.split_once(':') else {
            return Err(ServiceError::Error(format!(
                "Failed to parse `handler` specification: {}",
                config.handler
            )));
        };
        let df_batch = batch.swap_df(None)?.unwrap();
        let df_results = match Python::with_gil(|py| {
            py.import(handler_module)?
                .getattr(handler_fn)?
                .call((PyDataFrame(df_batch),), None)?
                .extract::<PyDataFrame>()
                .map(|pydf| -> DataFrame { pydf.into() })
        }) {
            Ok(df_results) => df_results,
            Err(err) => {
                // TODO: Handle errors
                error!("{err}");
                return Err(err.into());
            }
        };
        match batch.send_responses(df_results) {
            Ok(_) => (),
            Err(err) => error!("{}", err),
        };
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

struct BatchJob<R>
where
    R: Schema<R> + Sync + Send + 'static,
{
    jobs_by_id: HashMap<String, PredictionJob<R>>,
    df: Option<DataFrame>,
}

impl<R> BatchJob<R>
where
    R: Schema<R> + Sync + Send + 'static,
{
    fn from_jobs(jobs: Vec<PredictionJob<R>>) -> Result<BatchJob<R>, PolarsError> {
        let mut jobs_by_id = HashMap::<String, PredictionJob<R>>::new();
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

pub struct PredictionJob<R> {
    id: Uuid,
    records: Option<Vec<R>>,
    tx: oneshot::Sender<Result<PredictionResponse, ServiceError>>,
}

impl<R> PredictionJob<R>
where
    R: Schema<R> + Sync + Send + 'static,
{
    fn new(
        records: Vec<R>,
        tx: Sender<Result<PredictionResponse, ServiceError>>,
    ) -> PredictionJob<R> {
        let id = Uuid::new_v4();
        PredictionJob { id, records: Some(records), tx }
    }

    fn take_records_as_df(&mut self) -> Result<DataFrame, ServiceError> {
        let Some(records) = self.records.take() else {
            let msg = "Tried to take missing records".to_string();
            return Err(ServiceError::Error(msg));
        };
        let n_records = records.len();
        let mut df = R::to_dataframe(records)?;
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
