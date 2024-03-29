use polars::datatypes::DataType;
use redesmyn::{
    error::ServiceError, handler::PySpec, predictions::BatchPredictor, schema::{Relation, Schema}, server::{Serve, Server}
};




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

    let service = BatchPredictor::<String, Schema>::builder()
        .schema(schema)
        .path("predictions/{model_name}/{model_version}")
        .batch_max_size(100)
        .batch_max_delay_ms(5)
        .handler_config(PySpec::new().module("tests.test_server").method("handler").into())
        .build()?;

    let mut server = Server::default();
    // server.log_config(LogConfig::Stdout);
    server.register(service);
    server.push_pythonpath("./py-redesmyn");
    server.serve()?.await?;

    Ok(())
}
