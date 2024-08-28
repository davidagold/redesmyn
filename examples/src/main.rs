use cron;
use polars::datatypes::DataType;
use pyo3::{prelude::*, Python};
use redesmyn::{
    cache::{Cache, FsClient, Schedule},
    common::{consume_and_log_err, include_python_paths},
    do_in,
    error::ServiceError,
    handler::Handler,
    logging::{LogConfig, LogOutput},
    metrics::EmfOutput,
    predictions::Endpoint,
    schema::Schema,
    server::Server,
};
use std::{env::current_exe, path::PathBuf, str::FromStr};
use tracing::info;

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<(), ServiceError> {
    let Some(base_dir_str) = do_in!(|| {
        let exe_dir = current_exe().ok()?;
        exe_dir.parent()?.parent()?.parent()?.parent()?.to_path_buf().to_str()?.to_string()
    }) else {
        return Ok(());
    };
    info!("base_dir = {:#?}", base_dir_str);
    let example_dir: PathBuf = [base_dir_str.as_str(), "examples"].iter().collect();

    do_in!(|| {
        LogConfig::new(
            LogOutput::Stdout,
            Some(EmfOutput::new([example_dir.to_str()?, "logs/metrics.log"].iter().collect())),
        )
        .init();
    });

    let mut schema = Schema::default();
    schema.add_field("sepal_width", DataType::Float64);
    schema.add_field("petal_length", DataType::Float64);
    schema.add_field("petal_width", DataType::Float64);

    do_in!(|| {
        consume_and_log_err(include_python_paths([example_dir.to_str()?]));
    });

    let load_model = Python::with_gil(|py| {
        py.import_bound("model")?
            .getattr("SepalLengthPredictor")?
            .getattr("load_model")
            .map_err(|err| ServiceError::from(err.to_string()))
            .map(|obj| obj.unbind())
    })?;

    let Some(afs_client) = do_in!(|| {
        let models_dir: PathBuf =
            [base_dir_str.as_str(), "py-redesmyn/examples/iris"].iter().collect();
        FsClient::new(models_dir, "/models/mlflow/iris/{run_id}/{model_id}/artifacts/model".into())
    }) else {
        return Ok(());
    };
    let cache = Cache::new(
        afs_client,
        Some(64),
        Some(Schedule::Cron(
            cron::Schedule::from_str("0 * * * * * *")
                .map_err(|err| ServiceError::from(err.to_string()))?,
        )),
        Some(true),
        load_model,
        None,
    );

    let handler = Handler::try_from(
        Python::with_gil(|py| PyResult::Ok(py.import_bound("model")?.getattr("handle")?.unbind()))
            .map_err(ServiceError::from)?,
    )?;
    let endpoint = Endpoint::<String, Schema>::builder()
        .schema(schema)
        .path("/predictions/{run_id}/{model_id}")
        .cache(cache)
        .batch_max_size(100)
        .batch_max_delay_ms(5)
        .handler(handler)
        .build()?;

    let mut server = Server::new(None);
    server.register(endpoint);
    server.push_pythonpath("./py-redesmyn/examples/iris");
    server.serve()?.await?;

    Ok(())
}
