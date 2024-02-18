> redes-man -- An adviser, a counselor. [[quod.lib.umich.edu]](https://quod.lib.umich.edu/m/middle-english-dictionary/dictionary/MED36316)

# Redesmyn: serve ML model predictions with Python and Rust

Redesmyn (/ˈreɪd.smɪn/, REEDZ-min) helps you build real-time ML inference HTTP servers in Python and Rust.
* **Written in Rust**: The core Redesmyn server functionality is written in Rust for safety and performance.
* **Extensible through both Rust and Python**: Implement your application in Rust, Python, or whichever combination suits you:
    * Instantiate a server and specify endpoints and handler chains in either language.
    * Implement request handlers in either language:
        * Declare handler schemas via Rust structs that derive the `Record` trait and take advantage of [strongly-typed Serde parsing](https://docs.rs/serde_json/latest/serde_json/#parsing-json-as-strongly-typed-data-structures) parsing, or 
        * Declare handler schemas via [Pydantic](https://docs.pydantic.dev/latest/) models and use [untyped Serde parsing](https://docs.rs/serde_json/latest/serde_json/#operating-on-untyped-json-values).
    * Redesmyn by default passes model input parsed using either strategy above to your Python handlers as [Polars](https://pola.rs) DataFrames with zero-copy thanks to [Arrow](https://arrow.apache.org).
* **Flexible**: Redesmyn aims to expose as much control over your HTTP server as possible:
    * Specify your endpoint paths and parameters.
    * Apply custom header, request, and response logic.
* **Obervable**: Redesmyn aims to help you monitor the behavior of your models in real-time and post-hoc:
    * Logging is structured by default and surfaces dimensioned metrics for latency, availability, and input distribution summaries.
    * Write input to and output from any request handler to supported data sinks.
* **Optional integrated model store**: Integrate your endpoints with local or remote filesystem as your model stores: 
    * Map path and query parameters to model specifications that determine the location of your models.
    * Configure your service to automatically fetch updated models asynchronously without service interruption.
