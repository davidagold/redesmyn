use super::error::ServiceError;
use super::schema::{Relation, Schema};

use actix_web::{web, Handler, HttpResponse, Responder};
use polars::{frame::DataFrame, prelude::*};
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::sync::Mutex;
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

#[derive(Clone, Copy)]
enum EndpointState {
    Ready,
    Running,
}

#[derive(Clone, Debug)]
pub struct ServiceConfig {
    pub path: String,
    pub batch_max_delay_ms: u32,
    pub batch_max_size: usize,
    pub py_handler: String,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        ServiceConfig {
            path: "predictions/{model_name}/{model_version}".to_string(),
            batch_max_delay_ms: 5,
            batch_max_size: 1024,
            py_handler: "handlers.model:handle".to_string(),
        }
    }
}

pub struct BatchPredictor<T, R>
where
    R: Relation<Serialized = T>,
{
    config: ServiceConfig,
    schema: Schema,
    tx: Arc<Mutex<mpsc::Sender<PredictionJob<T, R>>>>,
    rx: Option<mpsc::Receiver<PredictionJob<T, R>>>,
    state: EndpointState,
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
                (Arc::new(Mutex::new(tx)), Some(rx))
            }
            EndpointState::Running => (self.tx.clone(), None),
        };
        BatchPredictor {
            config: self.config.clone(),
            schema: self.schema.clone(),
            tx,
            rx,
            state: self.state,
        }
    }
}

macro_rules! config_methods {
    ($($name:ident : $type:ty),*) => {
        $(
            pub fn $name(mut self, $name: $type) -> Self {
                self.config.$name = $name;
                self
            }
        )*
    }
}

impl<T, R> BatchPredictor<T, R>
where
    T: Send,
    R: Relation<Serialized = T> + Send + 'static,
{
    config_methods! {
        path: String,
        batch_max_delay_ms: u32,
        batch_max_size: usize,
        py_handler: String
    }

    pub fn new(schema: Schema) -> BatchPredictor<T, R> {
        let (tx, rx) = mpsc::channel(1024);

        BatchPredictor {
            tx: Arc::new(tx.into()),
            rx: Some(rx),
            config: ServiceConfig::default(),
            schema,
            state: EndpointState::Ready,
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
                    error!("Failed to {err}");
                    continue;
                }
            };
        }
    }

    #[instrument(skip_all)]
    fn predict_batch(mut batch: BatchJob<R>, config: ServiceConfig) -> Result<(), ServiceError> {
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

pub async fn invoke<T, R>(
    _model_spec: web::Path<ModelSpec>,
    records: web::Json<Vec<T>>,
    app_state: web::Data<BatchPredictor<T, R>>,
    schema: web::Data<Schema>,
) -> impl Responder
where
    T: Send + std::fmt::Debug,
    R: Relation<Serialized = T>,
    ModelSpec: for<'de> serde::Deserialize<'de>,
{
    // let ModelSpec { model_name, model_version } = &*model_spec;
    // info!(%model_name, %model_version);

    let (tx, rx) = oneshot::channel();
    let job = PredictionJob::<T, R>::new(records.into_inner(), tx, schema.into_inner());

    if let Err(err) = app_state.tx.lock().expect("Whoops").send(job).await {
        return HttpResponse::InternalServerError().body(err.to_string());
    }
    match rx.await {
        Ok(Ok(response)) => HttpResponse::Ok().json(response),
        Ok(Err(err)) => HttpResponse::InternalServerError().body(err.to_string()),
        Err(err) => HttpResponse::InternalServerError().body(err.to_string()),
    }
}

pub(crate) type HandlerArgs<R, T> =
    (web::Path<ModelSpec>, web::Json<Vec<T>>, web::Data<BatchPredictor<T, R>>, web::Data<Schema>);

impl<T, R> Service for BatchPredictor<T, R>
where
    Self: Sync,
    T: Send + std::fmt::Debug + 'static,
    R: Relation<Serialized = T> + Send + 'static,
{
    type R = R;
    type T = T;
    type H = impl Handler<HandlerArgs<Self::R, Self::T>, Output = impl Responder + 'static>;

    fn get_schema(&self) -> Schema {
        self.schema.clone()
    }

    fn run(&mut self) -> Result<JoinHandle<()>, ServiceError> {
        let (tx_abort, rx_abort) = oneshot::channel::<()>();

        let rx = std::mem::take(&mut self.rx).ok_or_else(|| {
            ServiceError::Error("Tried to start task from subordinate daemon.".to_string())
        })?;

        let config = self.config.clone();
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

