use actix_web::{post, web, HttpResponse, Responder};
use futures::channel::oneshot;
use polars::{frame::DataFrame, prelude::*};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    fmt,
    iter::repeat,
    time::Duration,
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
pub enum PredictionError {
    Error(String),
    PolarsError(polars::prelude::PolarsError),
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

impl fmt::Display for PredictionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PredictionError::Error(e) => write!(f, "General failure: {}", e),
            PredictionError::PolarsError(e) => write!(f, "Polars failure: {}", e),
        }
    }
}

impl std::error::Error for PredictionError {}

impl PredictionRequest {
    fn take_records_into_dataframe(&mut self, job_id: &String) -> Result<DataFrame, PolarsError> {
        let mut columns = HashMap::<String, Vec<f64>>::new();
        let n_records = self.records.len();

        (*self.records).into_iter().for_each(|record| {
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
            Series::from_iter(repeat(job_id.clone()).take(n_records)).with_name("job_id"),
        ) {
            Ok(_) => Ok(df),
            Err(err) => Err(err),
        }
    }
}

pub struct PredictionJob {
    id: Uuid,
    request: PredictionRequest,
    sender: oneshot::Sender<Result<PredictionResponse, PredictionError>>,
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
        return AppState { job_sender: sender.into() };
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
                event!(Level::INFO, msg = "Added one job to queue.");
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
    let mut senders_by_id =
        HashMap::<String, oneshot::Sender<Result<PredictionResponse, PredictionError>>>::new();

    let (successes, failures): (Vec<_>, Vec<_>) = jobs
        .into_iter()
        .map(|mut job| {
            (
                job.request.take_records_into_dataframe(&job.id.to_string()),
                job,
            )
        })
        .partition(|(result, _)| result.is_ok());

    failures.into_iter().for_each(|(result, job)| {
        match job.sender.send(Err(result.err().unwrap().into())) {
            Ok(_) => (),
            Err(_) => event!(
                Level::WARN,
                "Failed to send result for job with ID {}",
                job.id
            ),
        }
    });

    let dfs: Vec<LazyFrame> = successes
        .into_iter()
        .map(|(result, job)| {
            senders_by_id.insert(job.id.to_string(), job.sender);
            result.unwrap().lazy()
        })
        .collect();

    return concat(dfs, UnionArgs::default())
        .map(|df| df.collect())?
        .map(|mut df| {
            include_predictions(&mut df)?;
            return send_responses(&df, &mut senders_by_id);
        })?;
}

fn send_responses(
    df: &DataFrame,
    senders_by_id: &mut HashMap<
        String,
        oneshot::Sender<Result<PredictionResponse, PredictionError>>,
    >,
) -> Result<(), PredictionError> {
    for df in df.partition_by(["job_id"], true)? {
        let job_id = df
            .column("job_id")?
            .str()?
            .get(0)
            .ok_or_else(|| PredictionError::Error("Failed to get job ID.".to_string()))?;

        if let Some(sender) = senders_by_id.remove(&job_id.to_string()) {
            let predictions = df.column("prediction").map(|s| s.f64())??.to_vec();
            match sender.send(Ok(PredictionResponse { predictions })) {
                Ok(_) => (),
                Err(_) => event!(Level::ERROR, "Failed to send response"),
            }
        } else {
            event!(
                Level::ERROR,
                "Failed to get sender for job with ID {}",
                &job_id.to_string()
            )
        }
    }
    return Ok(());
}

fn include_predictions(df: &mut DataFrame) -> Result<(), PredictionError> {
    let predictions = Series::from_iter(repeat(1.0).take(df.shape().0));
    return df
        .with_column(predictions.with_name("prediction"))
        .map(|_| Ok(()))?
        .map_err(|err: PolarsError| err.into());
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
            Ok(_) => rx
                .await
                .map_err(|_| PredictionError::Error("Failed to receive".to_string()))?,
            Err(_) => Err(PredictionError::Error("Failed to send job".to_string())),
        }
    }
    .await;

    match result
        .and_then(|response| serde_json::to_string(&response).map_err(serde_json::Error::into))
    {
        Ok(serialized_response) => HttpResponse::Ok().body(serialized_response),
        Err(e) => HttpResponse::InternalServerError().body(e.to_string()),
    }
}
