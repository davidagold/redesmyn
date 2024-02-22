use super::error::ServiceError;
use actix_web::{post, web, HttpResponse, Responder};
use polars::{frame::DataFrame, prelude::*};
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use redesmyn_macros;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fmt, iter::repeat, time::Duration};
use tokio::{
    sync::{
        mpsc,
        oneshot::{self, Sender},
        Mutex,
    },
    time::Instant,
};
use tracing::{error, info, instrument};
use uuid::Uuid;

pub trait Schema<R> {
    fn to_dataframe(records: Vec<R>) -> PolarsResult<DataFrame>;
}

#[derive(Debug, Deserialize, redesmyn_macros::Schema)]
pub struct ToyRecord {
    a: f64,
    b: f64,
}

pub struct PredictionJob<R>
where
    R: Schema<R>,
{
    id: Uuid,
    records: Option<Vec<R>>,
    tx: oneshot::Sender<Result<PredictionResponse, ServiceError>>,
}

impl<R> PredictionJob<R>
where
    R: Schema<R>,
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

#[derive(Serialize)]
pub struct PredictionResponse {
    id: String,
    predictions: Vec<Option<f64>>,
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
    R: Schema<R>,
{
    tx: Mutex<mpsc::UnboundedSender<PredictionJob<R>>>,
}

impl<R> PredictionService<R>
where
    R: Schema<R>,
{
    pub fn new(tx: mpsc::UnboundedSender<PredictionJob<R>>) -> PredictionService<R> {
        PredictionService { tx: tx.into() }
    }
}

#[instrument(skip_all)]
pub async fn batch_predict_loop<R>(
    mut rx: mpsc::UnboundedReceiver<PredictionJob<R>>,
    mut rx_abort: oneshot::Receiver<()>,
) where
    R: Schema<R> + std::marker::Sync + Send + 'static,
{
    loop {
        if rx_abort.try_recv().is_ok() {
            // TODO: Ensure that outstanding requests are handled gracefully.
            return;
        }
        let start = Instant::now();
        let duration_wait = Duration::new(0, 5 * 1_000_000);
        let capacity = 1024;
        let mut jobs = Vec::<PredictionJob<R>>::with_capacity(capacity);
        while Instant::now() < start + duration_wait {
            if jobs.len() == capacity {
                break;
            };
            if let Ok(job) = rx.try_recv() {
                jobs.push(job);
            }
        }
        if jobs.is_empty() {
            continue;
        };

        // Process batch in separate blocking thread.
        tokio::task::spawn_blocking(move || {
            info!("Running batch predict for {} jobs.", jobs.len());
            let mut batch = BatchJob::from_jobs(jobs)?;
            let df_batch = (&mut batch).swap_df(None)?.unwrap();
            let df_results = match Python::with_gil(|py| {
                py.import("handlers.model")?
                    .getattr("handle")?
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
        });
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
    info!(%model_name, %model_version);

    let (tx, rx) = oneshot::channel();
    let job = PredictionJob::new(records.into_inner(), tx);

    if let Err(err) = app_state.tx.lock().await.send(job) {
        return HttpResponse::InternalServerError().body(err.to_string());
    }
    match rx.await {
        Ok(Ok(response)) => HttpResponse::Ok().json(response),
        Ok(Err(err)) => HttpResponse::InternalServerError().body(err.to_string()),
        Err(err) => HttpResponse::InternalServerError().body(err.to_string()),
    }
}

struct BatchJob<R>
where
    R: Schema<R>,
{
    jobs_by_id: HashMap<String, PredictionJob<R>>,
    df: Option<DataFrame>,
}

impl<R> BatchJob<R>
where
    R: Schema<R> + Send + 'static,
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
            let result = match df
                .column("prediction")
                .and_then(|s| s.f64())
                .map(|a| a.to_vec())
            {
                Ok(predictions) => Ok(PredictionResponse { id: job.id.to_string(), predictions }),
                Err(err) => Err(err.into()),
            };
            job.send_result(result);
        }
        Ok(())
    }
}
