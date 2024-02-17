use actix_web::{post, web, HttpResponse, Responder};
use polars::{frame::DataFrame, prelude::*};
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use rayon::prelude::*;
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
        mpsc,
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
    #[error("{0}")]
    PredictionError(#[from] PredictionError),
    #[error("Environment variable not found: {0}.")]
    VarError(#[from] VarError),
    #[error("{0}")]
    IoError(#[from] io::Error),
    #[error("{0}")]
    JoinError(#[from] JoinError),
    #[error("{0}")]
    Error(String),
    #[error("Failed to received result: {0}")]
    ReceiveError(#[from] RecvError),
    #[error("Polars operation failed: {0}")]
    ParseError(#[from] PolarsError),
    #[error("Failed to serialize result: {0}")]
    JsonError(#[from] serde_json::Error),
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

trait Push {
    fn push(&self, columns: &mut HashMap<String, Vec<f64>>);
}

#[derive(Debug, Deserialize)]
enum Record {
    ToyRecord(ToyRecord),
}

impl Record {
    fn push(&self, columns: &mut HashMap<String, Vec<f64>>) {
        match self {
            Record::ToyRecord(r) => r.push(columns),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ToyRecord {
    a: f64,
    b: f64,
}

impl Push for ToyRecord {
    fn push(&self, columns: &mut HashMap<String, Vec<f64>>) {
        columns
            .entry("a".to_string())
            .or_insert_with(Vec::new)
            .push(self.a);
        columns
            .entry("b".to_string())
            .or_insert_with(Vec::new)
            .push(self.b);
    }
}

#[derive(Deserialize)]
pub struct PredictionRequest {
    records: Vec<Record>,
}

pub struct PredictionJob {
    id: Uuid,
    request: PredictionRequest,
    sender: oneshot::Sender<Result<PredictionResponse, ServiceError>>,
}

impl PredictionJob {
    fn new(
        request: PredictionRequest,
        tx: Sender<Result<PredictionResponse, ServiceError>>,
    ) -> PredictionJob {
        let id = Uuid::new_v4();
        return PredictionJob { id, request, sender: tx };
    }

    fn take_records_as_df(&mut self) -> Result<DataFrame, ServiceError> {
        let mut columns = HashMap::<String, Vec<f64>>::new();
        let n_records = self.request.records.len();

        (*self.request.records)
            .into_iter()
            .for_each(|record| record.push(&mut columns));

        let series: Vec<Series> = columns
            .into_iter()
            .map(|(field, values)| Series::new(&field, values))
            .collect();

        let mut df = DataFrame::new(series)?;
        match df.with_column(
            Series::from_iter(repeat(self.id.to_string()).take(n_records)).with_name("job_id"),
        ) {
            Ok(_) => Ok(df),
            Err(err) => Err(err.into()),
        }
    }

    fn send(self, result: Result<PredictionResponse, ServiceError>) {
        match self.sender.send(result) {
            Err(_) => {
                // TODO: Structure logging
                let message = format!("Failed to send result for job with ID {}", self.id);
                event!(Level::WARN, message)
            }
            Ok(_) => (),
        }
    }
}

#[derive(Serialize)]
pub struct PredictionResponse {
    predictions: Vec<Option<f64>>,
}

pub struct AppState {
    job_sender: Mutex<mpsc::UnboundedSender<PredictionJob>>,
}

impl AppState {
    pub fn new(sender: mpsc::UnboundedSender<PredictionJob>) -> AppState {
        AppState { job_sender: sender.into() }
    }
}

#[instrument(skip_all)]
pub async fn batch_predict_loop<'a>(
    mut rx: mpsc::UnboundedReceiver<PredictionJob>,
    mut rx_abort: oneshot::Receiver<()>,
) {
    loop {
        if let Ok(_) = rx_abort.try_recv() {
            // TODO: Ensure that outstanding requests are handled gracefully.
            return;
        }
        let mut jobs = VecDeque::<PredictionJob>::new();
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
        // tokio::spawn(predict_and_send(jobs));
        let handle_send = tokio::spawn(async move { predict_and_send(jobs).await });
        tokio::spawn(await_send(handle_send));
    }
}

async fn await_send(handle_send: JoinHandle<Result<(), ServiceError>>) -> Result<(), ServiceError> {
    match handle_send.await {
        Ok(_) => (),
        Err(err) => event!(Level::ERROR, "{}", err),
    };
    Ok(())
}

struct BatchJob {
    jobs_by_id: HashMap<String, PredictionJob>,
    df: DataFrame,
}

impl BatchJob {
    fn from_jobs(jobs: VecDeque<PredictionJob>) -> Result<BatchJob, PolarsError> {
        let (jobs_by_id, dfs) = jobs.into_iter().fold(
            (
                HashMap::<String, PredictionJob>::new(),
                Vec::<DataFrame>::new(),
            ),
            |(mut jobs_by_id, mut dfs), mut job| {
                match job.take_records_as_df() {
                    Ok(df) => {
                        dfs.push(df);
                        jobs_by_id.insert(job.id.into(), job);
                    }
                    Err(err) => job.send(Err(err)),
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
}

#[instrument(skip_all)]
async fn predict_and_send(jobs: VecDeque<PredictionJob>) -> Result<(), ServiceError> {
    event!(
        Level::INFO,
        "Running batch predict for {} jobs.",
        jobs.len()
    );

    BatchJob::from_jobs(jobs)
        .map_err(PolarsError::into)
        .and_then(include_predictions)
        .map_err(PredictionError::into)
        .and_then(send_responses)
}

#[instrument(skip_all)]
fn include_predictions(mut batch: BatchJob) -> Result<BatchJob, PredictionError> {
    Python::with_gil(move |py| {
        py.import("handlers.model")?
            .getattr("handle")?
            .call((PyDataFrame(batch.take_df()),), None)?
            .extract::<PyDataFrame>()
            .map(|py_df| {
                batch.df = py_df.into();
                batch
            })
            .map_err(PyErr::into)
    })
}

fn send_responses(mut batch: BatchJob) -> Result<(), ServiceError> {
    batch
        .df
        .partition_by(["job_id"], true)?
        .iter()
        .map(|df| {
            let job_id = df
                .column("job_id")?
                .str()?
                .get(0)
                .ok_or_else(|| ServiceError::Error("Failed to get job ID.".to_string()))?;

            match batch.jobs_by_id.remove(&job_id.to_string()).ok_or_else(|| {
                PredictionError::Error(format!("Failed to retrieve job with ID {}", job_id))
            }) {
                Ok(job) => Ok((job, df)).into(),
                Err(err) => Result::<_, ServiceError>::Err(err.into()),
            }
        })
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|result| match result {
            Ok((job, df)) => {
                let predictions = df.column("prediction").map(|s| s.f64())??.to_vec();
                Ok(job.send(Ok(PredictionResponse { predictions })))
            }
            Err(err) => {
                let message_err = err.to_string();
                event!(Level::ERROR, "{}", message_err);
                Err(ServiceError::Error(message_err))
            }
        })
        .collect::<Vec<_>>();
    Ok(())
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

#[instrument(skip_all)]
#[post("/predictions/{model_name}/{model_version}")]
pub async fn submit_prediction_request(
    model_spec: web::Path<ModelSpec>,
    request: web::Json<PredictionRequest>,
    app_state: web::Data<AppState>,
) -> impl Responder {
    let ModelSpec { model_name, model_version } = &*model_spec;
    event!(Level::INFO, %model_name, %model_version);

    let (tx, rx) = oneshot::channel();
    let job = PredictionJob::new(request.into_inner(), tx);
    let result = async {
        match app_state.job_sender.lock().await.send(job) {
            Ok(_) => rx.await?,
            Err(_) => Err(ServiceError::Error("Failed to send job".to_string())),
        }
    }
    .await;

    match result
        .and_then(|response| serde_json::to_string(&response).map_err(serde_json::Error::into))
    {
        Ok(serialized_response) => HttpResponse::Ok().body(serialized_response),
        Err(err) => HttpResponse::InternalServerError().body(err.to_string()),
    }
}
