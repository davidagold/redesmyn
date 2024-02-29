use super::config_methods;
use super::error::ServiceError;
use super::schema::{Relation, Schema};

use actix_web::{web, Handler, HttpResponse, Responder};
use polars::{frame::DataFrame, prelude::*};
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::{collections::HashMap, fmt, iter::repeat, time::Duration};
use tokio::sync::mpsc::Receiver;
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

#[derive(Serialize)]
pub struct PredictionResponse {
    id: String,
    predictions: Vec<Option<f64>>,
}

#[derive(Clone, Debug)]
pub struct ServiceConfig {
    pub(super) path: String,
    pub(super) batch_max_delay_ms: u32,
    pub(super) batch_max_capacity: usize,
    pub(super) py_handler: String,
}

impl<'c> Default for ServiceConfig {
    fn default() -> Self {
        ServiceConfig {
            path: "predictions/{model_name}/{model_version}".to_string(),
            batch_max_delay_ms: 5,
            batch_max_capacity: 1024,
            py_handler: "handlers.model:handle".to_string(),
        }
    }
}

pub trait Configurable: Sized {
    fn get_config(&self) -> Result<ServiceConfig, ServiceError>;

    fn set_config(
        &mut self,
        new_config: Option<ServiceConfig>,
    ) -> Result<ServiceConfig, ServiceError>;

    config_methods! {
        path: String,
        batch_max_delay_ms: u32,
        batch_max_capacity: usize,
        py_handler: String
    }
}

impl Configurable for ServiceConfig {
    fn get_config(&self) -> Result<ServiceConfig, ServiceError> {
        return Ok(self.clone());
    }

    fn set_config(
        &mut self,
        new_config: Option<ServiceConfig>,
    ) -> Result<ServiceConfig, ServiceError> {
        match new_config {
            Some(new_config) => Ok(new_config),
            None => Ok(self.clone()),
        }
    }
}

pub trait Service: Sized {
    type R;
    type T;
    type H;

    fn run(&mut self) -> Result<JoinHandle<()>, ServiceError>;

    fn get_handler_fn(&self) -> Self::H;
}

// struct TaskDaemon<R>
// where
//     R: Relation + Sync + Send + 'static,
// {
//     rx: Option<mpsc::Receiver<PredictionJob<R>>>,
//     config: Option<ServiceConfig>,
//     schema: Schema,
// }

// impl<R> TaskDaemon<R>
// where
//     R: Relation + Sync + Send + 'static,
// {
//     fn start_task(&mut self) -> Result<JoinHandle<()>, ServiceError>
//     where
//         R: Relation + Sync + Send + 'static + for<'a> Deserialize<'a>,
//     {
//         let err = ServiceError::Error(
//             "Tried to start task from subordinate daemon.".to_string(),
//         );
//         let (tx_abort, rx_abort) = oneshot::channel::<()>();

//         let config = self.config.clone().ok_or_else(|| err)?;
//         let mut rx = std::mem::take(&mut self.rx)
//             .ok_or_else(|| err)?;

//         let handle = tokio::spawn(async move {
//             tokio::spawn(async move { BatchPredictor::<R>::task(rx, rx_abort, config).await });
//             // BatchPredictor::<R>::await_shutdown(tx_abort).await
//         });
//         Ok(handle)
//     }
// }

pub struct BatchPredictor<R, T>
where
    Self: Sync + Configurable,
    R: Relation + Sync + Send + 'static,
{
    config: Option<ServiceConfig>,
    schema: Schema,
    tx: Arc<Mutex<mpsc::Sender<PredictionJob<R>>>>,
    rx: Option<mpsc::Receiver<PredictionJob<R>>>,
    phantom: PhantomData<T>,
}

impl<R, T> Clone for BatchPredictor<R, T>
where
    Self: Sync + Configurable,
    R: Relation + Sync + Send + 'static,
{
    fn clone(&self) -> Self {
        BatchPredictor {
            config: self.get_config().ok().clone(),
            schema: self.schema.clone(),
            tx: self.tx.clone(),
            rx: None,
            phantom: PhantomData,
        }
    }
}

impl<R, T> Configurable for BatchPredictor<R, T>
where
    Self: Sync,
    R: Relation + Sync + Send + 'static,
{
    fn get_config(&self) -> Result<ServiceConfig, ServiceError> {
        match &self.config {
            Some(config) => Ok(config.clone()),
            None => Err(ServiceError::Error("Tried to take missing config.".to_string())),
        }
    }

    fn set_config(
        &mut self,
        new_config: Option<ServiceConfig>,
    ) -> Result<ServiceConfig, ServiceError> {
        let old_config = &self.config;
        match (old_config, new_config) {
            (None, Some(new_config)) => {
                self.config = Some(new_config.clone());
                Ok(new_config)
            }
            (Some(config), None) => Ok(config.clone()),
            _ => {
                Err(ServiceError::Error("Can only exchange distinct config variants.".to_string()))
            }
        }
    }
}

pub(crate) type HandlerArgs<R, T> =
    (web::Path<ModelSpec>, web::Json<Vec<T>>, web::Data<BatchPredictor<R, T>>, web::Data<Schema>);

impl<R, T> Service for BatchPredictor<R, T>
where
    R: Relation + Sync + Send + for<'de> Deserialize<'de> + 'static,
    T: Sync + Send + for<'de> Deserialize<'de> + 'static,
    Self: Sync + Configurable,
{
    type R = R;
    type T = T;
    type H =
        impl for<'rec> Handler<HandlerArgs<Self::R, Self::T>, Output = impl Responder + 'static>;

    fn run(&mut self) -> Result<JoinHandle<()>, ServiceError> {
        let err = ServiceError::Error("Tried to start task from subordinate daemon.".to_string());
        let (tx_abort, rx_abort) = oneshot::channel::<()>();

        let config = self.get_config()?;
        let rx = std::mem::take(&mut self.rx).ok_or_else(|| err)?;

        let handle = tokio::spawn(async move {
            tokio::spawn(async move { BatchPredictor::<R, T>::task(rx, rx_abort, config).await });
            BatchPredictor::<R, T>::await_shutdown(tx_abort).await
        });
        Ok(handle)
    }

    fn get_handler_fn(&self) -> Self::H {
        invoke::<Self::R, Self::T>
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

pub async fn invoke<R, T>(
    model_spec: web::Path<ModelSpec>,
    records: web::Json<Vec<T>>,
    app_state: web::Data<BatchPredictor<R, T>>,
    schema: web::Data<Schema>,
) -> impl Responder
where
    R: Relation + Sync + Send + for<'de> Deserialize<'de> + 'static,
    T: Sync + Send + for<'de> Deserialize<'de> + 'static,
    ModelSpec: for<'de> serde::Deserialize<'de>,
{
    // let ModelSpec { model_name, model_version } = &*model_spec;
    // info!(%model_name, %model_version);

    let (tx, rx) = oneshot::channel();
    let job = PredictionJob::<R>::new(records.into_inner(), tx, schema);

    if let Err(err) = app_state.tx.lock().await.send(job).await {
        return HttpResponse::InternalServerError().body(err.to_string());
    }
    match rx.await {
        Ok(Ok(response)) => HttpResponse::Ok().json(response),
        Ok(Err(err)) => HttpResponse::InternalServerError().body(err.to_string()),
        Err(err) => HttpResponse::InternalServerError().body(err.to_string()),
    }
}

// impl<'r, 'rec, R> Default for BatchPredictor<'r, 'rec>
// where
//     R: Relation + Sync + Send + 'static + for<'a> Deserialize<'a>,
// {
//     fn default() -> Self {
//         let schema = <R as Relation>::schema(None);
//         Self::new()
//     }
// }

impl<R, T> BatchPredictor<R, T>
where
    R: Relation + Sync + Send + 'static + for<'a> Deserialize<'a>,
    T: Sync + Send + for<'de> Deserialize<'de> + 'static,
{
    pub fn new(schema: Schema) -> BatchPredictor<R, T> {
        let (tx, rx) = mpsc::channel(1024);

        BatchPredictor {
            tx: Arc::new(tx.into()),
            rx: Some(rx),
            config: Some(ServiceConfig::default()),
            schema: schema.clone(),
            phantom: PhantomData
        }
    }

    #[instrument(skip_all)]
    async fn task(
        mut rx: mpsc::Receiver<PredictionJob<R>>,
        mut rx_abort: oneshot::Receiver<()>,
        config: ServiceConfig,
    ) {
        println!("Starting predict task with config: {:#?}", config);
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
    fn predict_batch(
        mut batch: BatchJob<R>,
        config: &ServiceConfig,
    ) -> Result<(), ServiceError> {
        info!("Running batch predict for {} jobs.", batch.len());

        let Some((handler_module, handler_fn)) = config.py_handler.split_once(':') else {
            return Err(ServiceError::Error(format!(
                "Failed to parse `handler` specification: {}",
                config.py_handler
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
    R: Relation + Sync + Send + 'static,
{
    jobs_by_id: HashMap<String, PredictionJob<R>>,
    df: Option<DataFrame>,
}

impl<R> BatchJob<R>
where
    R: Relation + Sync + Send + 'static,
{
    fn from_jobs(jobs: Vec<PredictionJob<R>>) -> Result<BatchJob<R>, PolarsError> {
        let mut jobs_by_id = HashMap::<String, PredictionJob<T>>::new();
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

pub struct PredictionJob<R>
where
    R: Relation + Sync + Send + 'static,
{
    id: Uuid,
    records: Option<Vec<R>>,
    tx: oneshot::Sender<Result<PredictionResponse, ServiceError>>,
    schema: Schema,
}

impl<R> PredictionJob<R>
where
    R: Relation + Sync + Send + 'static,
{
    fn new(
        records: Vec<R>,
        tx: Sender<Result<PredictionResponse, ServiceError>>,
        schema: Schema,
    ) -> PredictionJob<R>
    where
        R: Relation + Sync + Send + 'static,
    {
        let id = Uuid::new_v4();
        PredictionJob { id, records: Some(records), tx, schema }
    }

    fn take_records_as_df(&mut self) -> Result<DataFrame, ServiceError>
    where
        R: Relation + Sync + Send + 'static,
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
