use tokio::sync::mpsc;
use rs_model_server::predictions;
use actix_web::{web, App, HttpServer};
use tracing_subscriber::{self, EnvFilter, fmt::format::FmtSpan, layer::SubscriberExt, prelude::*};


#[tokio::main]
async fn main() -> std::io::Result<()> {
    let tracing_subscriber = tracing_subscriber::fmt::layer()
        .json()
        .with_span_events(FmtSpan::CLOSE);
    
    tracing_subscriber::registry()
        .with(EnvFilter::from_default_env())
        .with(tracing_subscriber)
        .init();

    let (tx, rx) = mpsc::channel(512);

    tokio::spawn(predictions::batch_predict_loop(rx));

    HttpServer::new(move || {
        let app_state = predictions::AppState::new(tx.clone());
        App::new()
            .app_data(web::Data::new(app_state))
            .service(predictions::submit_prediction_request)
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
