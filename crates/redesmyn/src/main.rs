use std::env;

use actix_web::{dev::ServerHandle, web, App, HttpServer};
use pyo3::{PyResult, Python};
use redesmyn::predictions::{self, PredictionService};
use tokio::{
    signal,
    sync::{mpsc, oneshot},
};
use tracing::instrument;
use tracing_subscriber::{self, layer::SubscriberExt, prelude::*, EnvFilter};

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
#[instrument]
async fn main() -> () {
    

}

async fn await_shutdown(server_handle: ServerHandle, tx_abort: oneshot::Sender<()>) {
    let _ = signal::ctrl_c().await;
    tracing::info!("Received shutdown signal.");
    if tx_abort.send(()).is_err() {
        tracing::error!("Failed to send cancel signal.");
    }
    server_handle.stop(true).await;
}
