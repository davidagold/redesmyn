use actix_web::{post, web, HttpResponse, Responder};
use polars::{frame::DataFrame, prelude::*};
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    env::VarError,
    fmt, io,
    iter::repeat,
    time::Duration,
};
use thiserror::Error;
use tokio::{
    sync::{
        mpsc::{self, error::SendError},
        oneshot::{self, error::RecvError, Sender},
        Mutex,
    },
    task::{JoinError, JoinHandle},
    time::Instant,
};
use tracing::{self, event, instrument, Level};
use uuid::Uuid;

#[derive(Error, Debug)]
pub enum ServiceError {
    #[error(transparent)]
    PredictionError(#[from] PredictionError),
    #[error("Environment variable not found: {0}.")]
    VarError(#[from] VarError),
    #[error(transparent)]
    IoError(#[from] io::Error),
    #[error(transparent)]
    JoinError(#[from] JoinError),
    #[error("{0}")]
    Error(String),
    #[error("Failed to forward request to prediction service: {0}")]
    SendError(String),
    #[error("Failed to received result: {0}")]
    ReceiveError(#[from] RecvError),
    #[error("Polars operation failed: {0}")]
    ParseError(#[from] PolarsError),
    #[error("Failed to serialize result: {0}")]
    JsonError(#[from] serde_json::Error),
}

impl<T> From<SendError<T>> for ServiceError {
    fn from(err: SendError<T>) -> Self {
        Self::SendError(err.to_string())
    }
}

#[derive(Error, Debug)]
pub enum PredictionError {
    #[error("Prediction failed: {0}.")]
    Error(String),
    #[error("Prediction failed from Polars operation: {0}.")]
    PolarsError(#[from] polars::prelude::PolarsError),
    #[error("Prediction failed from PyO3 operation: {0}.")]
    PyError(#[from] pyo3::prelude::PyErr),
    #[error("Prediction failed during IO: {0}.")]
    IoError(#[from] io::Error),
}

pub trait Record<R> {
    fn to_dataframe(records: Vec<R>) -> PolarsResult<DataFrame>;
}

#[derive(Debug, Deserialize)]
pub struct ToyRecord {
    a: f64,
    b: f64,
}

impl Record<ToyRecord> for ToyRecord {
    fn to_dataframe(records: Vec<ToyRecord>) -> PolarsResult<DataFrame> {
        let (mut a, mut b) = (Vec::<f64>::new(), Vec::<f64>::new());
        for record in records {
            a.push(record.a);
            b.push(record.b);
        }

        let columns: Vec<Series> = vec![Series::new("a", a), Series::new("b", b)];
        DataFrame::new(columns)
    }
}

pub struct PredictionJob<R>
where
    R: Record<R>,
{
    id: Uuid,
    records: Option<Vec<R>>,
    tx: oneshot::Sender<Result<PredictionResponse, ServiceError>>,
}

impl<R> PredictionJob<R>
where
    R: Record<R>,
{
    fn new(
        records: Vec<R>,
        tx: Sender<Result<PredictionResponse, ServiceError>>,
    ) -> PredictionJob<R> {
        let id = Uuid::new_v4();
        return PredictionJob { id, records: Some(records), tx };
    }

    fn take_records_as_df(&mut self) -> Result<DataFrame, ServiceError> {
        let records = match self.records.take() {
            Some(records) => records,
            None => {
                return Err(ServiceError::Error(
                    "Tried to take missing records".to_string(),
                ))
            }
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
        match self.tx.send(result) {
            Err(_) => {
                // TODO: Structure logging
                let message = format!("Failed to send result for job with ID {}", self.id);
                event!(Level::ERROR, message)
            }
            Ok(_) => (),
        }
    }
}

#[derive(Serialize)]
pub struct PredictionResponse {
    predictions: Vec<Option<f64>>,
}

#[instrument(skip_all)]
pub async fn batch_predict_loop<R>(
    mut rx: mpsc::UnboundedReceiver<PredictionJob<R>>,
    mut rx_abort: oneshot::Receiver<()>,
) where
    R: Record<R> + Send + 'static,
{
    loop {
        if let Ok(_) = rx_abort.try_recv() {
            // TODO: Ensure that outstanding requests are handled gracefully.
            return;
        }
        let mut jobs = VecDeque::<PredictionJob<R>>::new();
        let start = Instant::now();
        let duration_wait = Duration::new(0, 10 * 1_000_000);
        while Instant::now() < start + duration_wait {
            if let Ok(job) = rx.try_recv() {
                jobs.push_back(job);
            }
        }
        if jobs.len() == 0 {
            continue;
        };
        let handle_send = tokio::spawn(async move {
            match predict_and_send(jobs).await {
                Ok(()) => (),
                Err(err) => tracing::error!("{}", err),
            }
        });
        tokio::spawn(await_send(handle_send));
    }
}

async fn await_send(handle_send: JoinHandle<()>) {
    match handle_send.await {
        Ok(_) => (),
        Err(err) => event!(Level::ERROR, "{}", err),
    };
}

struct BatchJob<R>
where
    R: Record<R>,
{
    jobs_by_id: HashMap<String, PredictionJob<R>>,
    df: DataFrame,
}

impl<R> BatchJob<R>
where
    R: Record<R>,
{
    fn from_jobs(jobs: VecDeque<PredictionJob<R>>) -> Result<BatchJob<R>, PolarsError> {
        let (jobs_by_id, dfs) = jobs.into_iter().fold(
            (
                HashMap::<String, PredictionJob<R>>::new(),
                Vec::<DataFrame>::new(),
            ),
            |(mut jobs_by_id, mut dfs), mut job| {
                match job.take_records_as_df() {
                    Ok(df) => {
                        dfs.push(df);
                        jobs_by_id.insert(job.id.into(), job);
                    }
                    Err(err) => job.send_result(Err(err)),
                }
                (jobs_by_id, dfs)
            },
        );
        match concat::<Vec<_>>(
            dfs.into_iter().map(DataFrame::lazy).collect(),
            UnionArgs::default(),
        )
        .and_then(LazyFrame::collect)
        {
            Ok(df) => Ok(BatchJob { jobs_by_id, df }),
            Err(err) => Err(err),
        }
    }

    fn take_df(&mut self) -> DataFrame {
        std::mem::replace(&mut self.df, DataFrame::empty())
    }

    #[instrument(skip_all)]
    fn include_predictions(&mut self) -> Result<(), PredictionError> {
        Python::with_gil(|py| {
            py.import("handlers.model")?
                .getattr("handle")?
                .call((PyDataFrame(self.take_df()),), None)?
                .extract::<PyDataFrame>()
                .map(|py_df| {
                    self.df = py_df.into();
                })
                .map_err(PyErr::into)
        })
    }

    #[instrument(skip_all)]
    fn send_responses(mut self) -> Result<(), ServiceError> {
        let dfs = self.df.partition_by(["job_id"], true)?;
        let results = dfs
            .into_iter()
            .filter_map(|df| {
                if let Ok(Some(job_id)) =
                    df.column("job_id").and_then(|s| s.str()).map(|s| s.get(0))
                {
                    match self.jobs_by_id.remove(&job_id.to_string()) {
                        Some(job) => Some((job, df)),
                        None => {
                            tracing::error!("Failed to retrieve job with ID {}", job_id);
                            None
                        }
                    }
                } else {
                    tracing::error!("Failed to retrieve job ID from results DataFrame.");
                    None
                }
            })
            .collect::<Vec<_>>();

        for (job, df) in results {
            match df
                .column("prediction")
                .and_then(|s| s.f64())
                .map(|a| a.to_vec())
            {
                Ok(predictions) => job.send_result(Ok(PredictionResponse { predictions })),
                Err(err) => tracing::error!("{}", err),
            }
        }
        Ok(())
    }
}

#[instrument(skip_all)]
async fn predict_and_send<R>(jobs: VecDeque<PredictionJob<R>>) -> Result<(), ServiceError>
where
    R: Record<R>,
{
    event!(
        Level::INFO,
        "Running batch predict for {} jobs.",
        jobs.len()
    );

    let mut batch = BatchJob::from_jobs(jobs)?;
    batch.include_predictions()?;
    batch.send_responses()
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

pub struct PredictionService<R>
where
    R: Record<R>,
{
    tx: Mutex<mpsc::UnboundedSender<PredictionJob<R>>>,
}

impl<R> PredictionService<R>
where
    R: Record<R>,
{
    pub fn new(tx: mpsc::UnboundedSender<PredictionJob<R>>) -> PredictionService<R> {
        PredictionService { tx: tx.into() }
    }
}

#[instrument(skip_all)]
#[post("/predictions/{model_name}/{model_version}")]
pub async fn submit_prediction_request(
    model_spec: web::Path<ModelSpec>,
    records: web::Json<Vec<ToyRecord>>,
    app_state: web::Data<PredictionService<ToyRecord>>,
) -> impl Responder {
    let ModelSpec { model_name, model_version } = &*model_spec;
    event!(Level::INFO, %model_name, %model_version);

    let (tx, rx) = oneshot::channel();
    let job = PredictionJob::new(records.into_inner(), tx);

    let result = async {
        match app_state.tx.lock().await.send(job) {
            Ok(_) => rx.await?,
            Err(err) => Err(err.into()),
        }
    }
    .await;

    match result {
        Ok(response) => HttpResponse::Ok().json(response),
        Err(err) => HttpResponse::InternalServerError().body(err.to_string()),
    }
}
