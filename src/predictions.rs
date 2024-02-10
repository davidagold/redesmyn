use actix_web::{post, web, HttpResponse, Responder};
use futures::channel::oneshot;
use serde::Deserialize;
use std::{
    collections::VecDeque,
    fmt,
    iter::{repeat, zip},
    time::Duration,
};
use tokio::{
    sync::{mpsc, Mutex},
    time::Instant,
};
use tracing::{self, event, instrument, Level};

#[derive(Deserialize)]
pub struct PredictionRequest {
    records: Vec<ModelInput>,
}

impl PredictionRequest {
    fn take_records(&mut self) -> Vec<ModelInput> {
        std::mem::take(&mut self.records)
    }
}

#[derive(Debug, Deserialize)]
pub struct ModelInput {
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

pub struct PredictionResponse {
    model_inputs: Vec<ModelInput>,
    predictions: Vec<f64>,
}

pub struct AppState {
    job_sender: Mutex<mpsc::Sender<(Vec<ModelInput>, oneshot::Sender<PredictionResponse>)>>,
}

impl AppState {
    pub fn new(sender: mpsc::Sender<(Vec<ModelInput>, oneshot::Sender<PredictionResponse>)>) -> AppState {
        return AppState {
            job_sender: sender.into(),
        };
    }
}

#[instrument(skip_all)]
pub async fn batch_predict_loop(
    mut receiver: mpsc::Receiver<(Vec<ModelInput>, oneshot::Sender<PredictionResponse>)>,
) {
    loop {
        let mut queue = VecDeque::<(Vec<ModelInput>, oneshot::Sender<PredictionResponse>)>::new();
        let start = Instant::now();
        let duration_wait = Duration::new(0, 1000);
        while Instant::now() < start + duration_wait {
            if let Some((model_input, tx)) = receiver.recv().await {
                queue.push_back((model_input, tx));
                event!(Level::INFO, msg = "Added one job to queue.");
            }
        }
        let jobs_by_sender: Vec<(Vec<ModelInput>, oneshot::Sender<PredictionResponse>)> =
            queue.into_iter().collect();

        let model_responses = predict(
            jobs_by_sender
                .iter()
                .map(|job_by_sender| &job_by_sender.0)
                .collect(),
        );

        zip(jobs_by_sender, model_responses)
            .for_each(|((_, tx), responses)| {
                match tx.send(responses) {
                    Ok(()) => (),
                    Err(_) => event!(Level::ERROR, "Prediction failed."),
                }
        });
    }
}

#[instrument]
fn predict(model_inputs: Vec<&Vec<ModelInput>>) -> Vec<PredictionResponse> {
    let responses = Vec::from_iter(
        model_inputs
            .iter()
            .map(|model_input| Vec::from_iter(model_input.iter().map(|_| 1.0)))
            .zip(model_inputs)
            .map(|(predictions, model_inputs)| PredictionResponse {
                model_inputs: *model_inputs,
                predictions,
            }),
    );
    responses
}

#[instrument(skip_all)]
#[post("/predictions/{model_name}/{model_version}")]
pub async fn submit_prediction_request(
    model_spec: web::Path<ModelSpec>,
    mut model_input: web::Json<PredictionRequest>,
    app_state: web::Data<AppState>,
) -> impl Responder {
    let ModelSpec {
        model_name,
        model_version,
    } = &*model_spec;
    event!(Level::INFO, %model_name, %model_version);

    let (tx, rx) = oneshot::channel();

    let result = async {
        app_state
            .job_sender
            .lock()
            .await
            .send((model_input.take_records(), tx))
            .await
            .map_err(|_| "Send failed")?;

        rx.await.map_err(|_| "Receive failed")
    }
    .await;

    match result {
        Ok(response) => HttpResponse::Ok().body(serde:: response),
        Err(e) => HttpResponse::InternalServerError().body(e),
    }
}
