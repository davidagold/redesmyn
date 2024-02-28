use std::iter::repeat;
use std::{collections::HashMap, time::Instant};


use polars::prelude::*;
use redesmyn::{
    error::ServiceError,
    predictions::{BatchPredictor, Configurable},
    schema::{self, dataframe_from_records},
    server::{Serve, Server},
};
// use redesmyn_macros::Schema;
use serde::de::{DeserializeSeed, IntoDeserializer};
use serde::Deserialize;
use serde_json::Deserializer;

// #[derive(Debug, Deserialize, Schema)]
pub struct ToyRecord {
    a: f64,
    b: f64,
}

// #[tokio::main(flavor = "multi_thread", worker_threads = 4)]
fn main() -> Result<(), ServiceError> {
    // let service = BatchPredictor::<ToyRecord>::new()
    //     .path("predictions/{model_name}/{model_version}")
    //     .batch_max_capacity(50)
    //     .batch_max_delay_ms(10)
    //     .handler("handlers.model:handle");

    // let server = Server::default().register(service);
    // server.serve()?.await?.map_err(Into::into)

    let json = r#"
        {
            "a": 1.0,
            "b": null,
            "c": "Foo",
            "d": 13321312312
        }
    "#;

    for _ in 0..10 {

        let n_records: usize = 1000;
    
        let start_0 = Instant::now();
        let schema = schema::Schema::default()
            .add_field("a", DataType::Float64)
            .add_field("b", DataType::Float64)
            .add_field("c", DataType::String)
            .add_field("d", DataType::Int64);
    
        let records = repeat(json).take(n_records).collect::<Vec<_>>();
        let df = dataframe_from_records(&schema, records)?;
        println!("Elapsed: {:#?}", start_0.elapsed());
        // println!("{:#?}", df);
        
        // let start_1 = Instant::now();

        // #[derive(Deserialize)]
        // struct ToyRecord {
        //     a: f64,
        //     b: Option<f64>,
        //     c: String,
        //     d: i64,
        // }
        // let (a, b, c, d) = repeat(json).take(n_records).map(|r| serde_json::from_str(r).unwrap()).fold(
        //     (Vec::<f64>::new(), Vec::<Option<f64>>::new(), Vec::<String>::new(), Vec::<i64>::new()),
        //     |(mut a, mut b, mut c, mut d), r: ToyRecord| {
        //         a.push(r.a);
        //         b.push(r.b);
        //         c.push(r.c);
        //         d.push(r.d);
        //         (a, b, c, d)
        //     },
        // );
        // let df = DataFrame::new(vec![
        //     Series::new("a", a),
        //     Series::new("b", b),
        //     Series::new("c", c),
        //     Series::new("d", d),
        // ])?;
    
        // println!("{:#?}", start_1.elapsed());
        // println!("{df}");
    }


    Ok(())
}
