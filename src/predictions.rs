use actix_web::{post, web, HttpResponse, Responder};
use futures::channel::oneshot;
use polars::{frame::DataFrame, prelude::*};
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque}, env::VarError, fmt, iter::repeat, time::Duration, io
};
use tokio::{
    sync::{mpsc, Mutex},
    time::Instant,
};
use tracing::{self, event, instrument, Level};
use uuid::Uuid;

#[derive(Deserialize)]
pub struct PredictionRequest {
    records: Vec<Record>,
}

#[derive(Debug)]
pub enum ServiceError {
    PredictionError(PredictionError),
    VarError(VarError),
    IoError(io::Error),
}

impl From<PredictionError> for ServiceError {
    fn from(err: PredictionError) -> Self {
        ServiceError::PredictionError(err)
    }
}

impl From<VarError> for ServiceError {
    fn from(err: VarError) -> Self {
        ServiceError::VarError(err)
    }
}

impl From<io::Error> for ServiceError {
    fn from(err: io::Error) -> Self {
        ServiceError::IoError(err)
    }
}

impl fmt::Display for ServiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ServiceError::PredictionError(err) => write!(f, "{}", err),
            ServiceError::VarError(err) => write!(f, "{}", err),
            ServiceError::IoError(err) => write!(f, "{}", err),
        }
    }
}

#[derive(Debug)]
pub enum PredictionError {
    Error(String),
    PolarsError(polars::prelude::PolarsError),
    PyError(pyo3::prelude::PyErr),
    IoError(io::Error),
}

impl From<PolarsError> for PredictionError {
    fn from(err: PolarsError) -> Self {
        PredictionError::PolarsError(err)
    }
}

impl From<serde_json::Error> for PredictionError {
    fn from(err: serde_json::Error) -> Self {
        PredictionError::Error(err.to_string())
    }
}

impl From<pyo3::prelude::PyErr> for PredictionError {
    fn from(err: pyo3::prelude::PyErr) -> Self {
        PredictionError::PyError(err)
    }
}

impl From<io::Error> for PredictionError {
    fn from(err: io::Error) -> Self {
        PredictionError::IoError(err)
    }
}

impl fmt::Display for PredictionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PredictionError::Error(err) => write!(f, "{}", err),
            PredictionError::PolarsError(err) => write!(f, "Polars failure: {}", err),
            PredictionError::PyError(err) => write!(f, "Python failure: {}", err),
            PredictionError::IoError(err) => write!(f, "IO error: {}", err)
        }
    }
}

impl std::error::Error for PredictionError {}

pub struct PredictionJob {
    id: Uuid,
    request: PredictionRequest,
    sender: oneshot::Sender<Result<PredictionResponse, PredictionError>>,
}

impl PredictionJob {
    fn take_records_as_dataframe(&mut self) -> Result<DataFrame, PolarsError> {
        let mut columns = HashMap::<String, Vec<f64>>::new();
        let n_records = self.request.records.len();

        (*self.request.records).into_iter().for_each(|record| {
            columns
                .entry("a".to_string())
                .or_insert_with(Vec::new)
                .push(record.a);
            columns
                .entry("b".to_string())
                .or_insert_with(Vec::new)
                .push(record.b);
        });

        let series: Vec<Series> = columns
            .into_iter()
            .map(|(field, values)| Series::new(&field, values))
            .collect();

        let mut df = match DataFrame::new(series) {
            Ok(df) => df,
            Err(err) => return Err(err.into()),
        };
        match df.with_column(
            Series::from_iter(repeat(self.id.to_string()).take(n_records)).with_name("job_id"),
        ) {
            Ok(_) => Ok(df),
            Err(err) => Err(err),
        }
    }

    fn send(self, result: Result<PredictionResponse, PredictionError>) {
        if let Err(_) = self.sender.send(Err(result.err().unwrap().into())) {
            // TODO: Structure logging
            let message = format!("Failed to send result for job with ID {}", self.id);
            event!(Level::WARN, message)
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Record {
    a: f64,
    b: f64,
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

#[derive(Serialize)]
pub struct PredictionResponse {
    // model_inputs: Vec<Record>,
    predictions: Vec<Option<f64>>,
}

pub struct AppState {
    job_sender: Mutex<mpsc::Sender<PredictionJob>>,
}

impl AppState {
    pub fn new(sender: mpsc::Sender<PredictionJob>) -> AppState {
        AppState { job_sender: sender.into() }
    }
}

#[instrument(skip_all)]
pub async fn batch_predict_loop(mut receiver: mpsc::Receiver<PredictionJob>) {
    loop {
        let mut jobs = VecDeque::<PredictionJob>::new();
        let start = Instant::now();
        let duration_wait = Duration::new(0, 1000);
        while Instant::now() < start + duration_wait {
            if let Some(job) = receiver.recv().await {
                jobs.push_back(job);
                event!(Level::INFO, message = "Added one job to queue.");
            }
        }

        // TODO: Handle cases.
        match predict_and_send(jobs) {
            Ok(_) => (),
            Err(_) => (),
        };
    }
}

#[instrument(skip_all)]
fn predict_and_send(jobs: VecDeque<PredictionJob>) -> Result<(), PredictionError> {
    let mut jobs_by_id = HashMap::<String, PredictionJob>::new();
    let mut dfs = Vec::<LazyFrame>::new();

    jobs.into_iter()
        .for_each(|mut job| match job.take_records_as_dataframe() {
            Ok(df) => {
                dfs.push(df.lazy());
                jobs_by_id.insert(job.id.into(), job);
            }
            Err(err) => job.send(Err(err.into())),
        });

    concat(dfs, UnionArgs::default())
        .and_then(LazyFrame::collect)
        .map_err(PolarsError::into)
        .and_then(include_predictions)
        .and_then(|results_df| {
            event!(Level::INFO, "Sending results");
            send_responses(&results_df, &mut jobs_by_id)
        })
        .map_err(|err| {
            let message_err = err.to_string();
            jobs_by_id.into_values().for_each(|job| {
                job.send(Err(PredictionError::Error(message_err.clone())));
            });
            PredictionError::Error(message_err)
        })
}

#[instrument(skip_all)]
fn include_predictions(df: DataFrame) -> Result<DataFrame, PredictionError> {
    Python::with_gil(|py| {
        py.import("handlers.model")?
            .getattr("handle")?
            .call((PyDataFrame(df),), None)?
            .extract::<PyDataFrame>()
    })
    .map(|py_df| py_df.into())
    .map_err(|err| err.into())
}

fn send_responses(
    df: &DataFrame,
    jobs_by_id: &mut HashMap<String, PredictionJob>,
) -> Result<(), PredictionError> {
    for df in df.partition_by(["job_id"], true)? {
        let job_id = df
            .column("job_id")?
            .str()?
            .get(0)
            .ok_or_else(|| PredictionError::Error("Failed to get job ID.".to_string()))?;

        if let Some(job) = jobs_by_id.remove(&job_id.to_string()) {
            let predictions = df.column("prediction").map(|s| s.f64())??.to_vec();
            job.send(Ok(PredictionResponse { predictions }));
        } else {
            event!(Level::ERROR, "Failed to get job with ID {}", job_id)
        }
    }
    Ok(())
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

    let result = async {
        match app_state
            .job_sender
            .lock()
            .await
            .send(PredictionJob {
                id: Uuid::new_v4(),
                request: request.into_inner(),
                sender: tx,
            })
            .await
        {
            Ok(_) => rx.await.map_err(|err| {
                PredictionError::Error(format!("Failed to receive result: {}", err))
            })?,
            Err(_) => Err(PredictionError::Error("Failed to send job".to_string())),
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
