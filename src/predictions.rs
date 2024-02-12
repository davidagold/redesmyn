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
}

impl fmt::Display for PredictionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PredictionError::Error(e) => write!(f, "Prediction failed: {}", e),
        }
    }
}

impl std::error::Error for PredictionError {}

impl PredictionRequest {
    fn take_records_into_dataframe(&mut self, job_id: &Uuid) -> Result<DataFrame, PolarsError> {
        let mut columns = HashMap::<String, Vec<f64>>::new();

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

        let mut df = DataFrame::new(series).expect("Failed to create DataFrame");
        let mut column_job_id = Series::from_iter(repeat(job_id.to_string()).take(df.shape().0));
        column_job_id.rename("job_id");
        match df.with_column(column_job_id) {
            Ok(_) => Ok(df),
            Err(e) => Err(e),
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
    model_inputs: Vec<Record>,
    predictions: Vec<f64>,
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

        predict_and_send(jobs);
    }
}

#[instrument(skip_all)]
fn predict_and_send(jobs: VecDeque<PredictionJob>) -> () {
    let mut senders_by_id =
        HashMap::<Uuid, oneshot::Sender<Result<PredictionResponse, PredictionError>>>::new();
    let dfs = jobs
        .into_iter()
        .fold(Vec::<LazyFrame>::new(), |mut dfs, mut job| {
            match job.request.take_records_into_dataframe(&job.id) {
                Ok(df) => dfs.push(df.lazy()),
                Err(e) => {
                    match job.sender.send(Err(PredictionError::Error(format!(
                        "Failed to get DataFrame: {}",
                        e
                    )))) {
                        Ok(_) => (),
                        Err(_) => {
                            event!(
                                Level::WARN,
                                "Failed to send result for job with ID {}",
                                job.id
                            )
                        }
                    }
                    return dfs;
                }
            };
            senders_by_id.insert(job.id, job.sender);
            dfs
            // let n_predictions = &model_inputs.len();
            // let predictions = repeat(1.0).take(*n_predictions).collect();
            // return (PredictionResponse { model_inputs, predictions }, tx);
        });
    let df = concat(dfs, UnionArgs::default());
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
        app_state
            .job_sender
            .lock()
            .await
            .send(PredictionJob {
                id: Uuid::new_v4(),
                request: request.into_inner(),
                sender: tx,
            })
            .await
            .map_err(|_| "Send failed")?;

        rx.await.map_err(|_| "Receive failed")
    }
    .await;

    match result {
        Ok(result) => match result {
            Ok(response) => match serde_json::to_string(&response) {
                Ok(serialized_response) => HttpResponse::Ok().body(serialized_response),
                Err(e) => HttpResponse::InternalServerError().body(e.to_string()),
            },
            Err(e) => HttpResponse::InternalServerError().body(e.to_string()),
        },
        Err(e) => HttpResponse::InternalServerError().body(e.to_string()),
    }
}
