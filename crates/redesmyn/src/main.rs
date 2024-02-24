use std::env;

use actix_web::{dev::ServerHandle, web, App, HttpServer};
use pyo3::{PyResult, Python};
use redesmyn::predictions::{self, BatchPredictor};
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


