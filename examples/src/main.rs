use cron;
use polars::datatypes::DataType;
use pyo3::{types::PyFunction, Py, Python};
use redesmyn::{
    cache::{ArtifactsClient, Cache, FsClient, Schedule},
    do_in,
    error::ServiceError,
    handler::PySpec,
    predictions::BatchPredictor,
    schema::{Relation, Schema},
    server::Server,
};
use std::{env::current_exe, str::FromStr};

// #[derive(Debug, Deserialize, Relation)]
// pub struct ToyRecord {
//     a: f64,
//     b: f64,
// }

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<(), ServiceError> {
    // let schema = <ToyRecord as Relation>::schema(None).unwrap();

    let mut schema = Schema::default();
    schema.add_field("a", DataType::Float64);
    schema.add_field("b", DataType::Float64);

    // let service = BatchPredictor::<String, ToyRecord>::new(schema)
    let load_model: Py<PyFunction> = Python::with_gil(|py| {
        py.import("mlflow")?
            .getattr("sklearn")?
            .getattr("load_model")?
            .extract::<Py<PyFunction>>()
            .map_err(|err| ServiceError::from(err.to_string()))
    })?;

    let Some(afs_client) = do_in!(|| {
        let exe_dir = current_exe().ok()?;
        let mut models_dir = exe_dir.parent()?.parent()?.parent()?.to_path_buf();
        models_dir.push("models");
        ArtifactsClient::FsClient {
            client: FsClient::new(models_dir),
            load_model,
        }
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
    );

    let endpoint = BatchPredictor::<String, Schema>::builder()
        .schema(schema)
        .path("/predictions/{model_version}")
        .cache(cache)
        .batch_max_size(100)
        .batch_max_delay_ms(5)
        .handler_config(PySpec::new().module("tests.test_server").method("handler").into())
        .build()?;

    let mut server = Server::default();
    // server.log_config(LogConfig::Stdout);
    server.register(endpoint);
    server.push_pythonpath("./py-redesmyn");
    server.serve()?.await?;

    Ok(())
}
