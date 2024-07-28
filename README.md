## Redesmyn: Build ML inference servers with Python and Rust

Redesmyn (/ˈreɪd.smɪn/, REEDZ-min) helps you build services for real-time ML inference in Python and Rust:
* **Dual language**: Core Redesmyn functionality is written in Rust for safety and performance and exposed through interoperable Python and Rust APIs.
* **Predict in Python**: Seamlessly integrate prediction handlers written in Python with Rust server frameworks.
* **Serde deserialization**: Declare inference handler schemas via the Rust `Schema` trait or [Pydantic](https://docs.pydantic.dev/latest/) Python subclasses to use
    [strongly-typed Serde parsing](https://docs.rs/serde_json/latest/serde_json/#parsing-json-as-strongly-typed-data-structures) or
    [untyped Serde parsing](https://docs.rs/serde_json/latest/serde_json/#operating-on-untyped-json-values), respectively.
* **Built on Polars**: Request payloads are parsed into [Polars](https://pola.rs) DataFrames that can be passed to Python inference handlers with zero copy.
* **Ergonomic API**: Declare service endpoints with customizable paths and parameters and conduct request validation with Pydantic models.
* **Asynchronous model cache**: Manage model caching and async updating via an integrated cache that maps URL parameters to model variants.
* **Observability via Embedded Metrics Format logging**: Redesmyn applications can be configured to emit collated EMF metrics log messages.
