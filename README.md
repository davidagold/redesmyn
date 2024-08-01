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
* **Observability**: Redesmyn applications can be configured to emit collated AWS EMF metrics log messages.

### Example

To illustrate, the snippet below instantiates and runs a Redesmyn `Server` whose single `Endpoint` is managed by an inference handler that receives
batched inference requests as a Polars DataFrame and accesses a cached `sklearn` model parametrized by `run_id` and `model_id`.
```python
import asyncio

import mlflow
import redesmyn.artifacts as afs
import redesmyn.service as svc
from sklearn.pipeline import Pipeline


@svc.endpoint(
    path="/predictions/iris/{run_id}/{model_id}",
    batch_max_delay_ms=10,
    batch_max_size=64,
    cache_config=afs.CacheConfig(
        client=afs.FsClient(
            base_path=Path(__file__).parent,
            path_template="/models/mlflow/iris/{run_id}/{model_id}/artifacts/model",
        ),
        load_model=mlflow.sklearn.load_model,
    ),
)
def handle(model: Pipeline, records_df: DataFrame) -> DataFrame:
    return model.predict(X=records_df)


async def main():
    server = svc.Server()
    server.register(endpoint=handle)
    await server.serve()


asyncio.run(main())
```


## `Endpoint`s

A Redesmyn server is just an [Actix](https://actix.rs/docs/) HTTP server with `Endpoint`s that serve `POST` requests containing records against which to conduct inference.
Just like a regular HTTP server, each such `Endpoint` is associated with a path, which can be configured in the specification of the `Endpoint` handler:

```python

model = mlflow.sklearn.load_model(model_uri=...)

@svc.endpoint(path="/predictions/iris/{run_id}/{model_id}")
def handle(records_df: DataFrame) -> DataFrame:
    return model.predict(X=records_df)

```

The handler function itself is just a Python function that expects a Polars `DataFrame` argument, which contains the present batch of inference requests.
Redesmyn takes care of deserializing incoming requests into Polars rows and batching the latter into a `DataFrame`.
Thanks to Polars' use of Arrow and PyO3, the Rust-based server functionality and Python-based inference functionality are interoperable with zero IPC or copying of data.

You can modify the batching behavior with the following parameters:

```python
@svc.endpoint(
    path="/predictions/iris/{run_id}/{model_id}",
    batch_max_delay_ms=10,
    batch_max_size=64,
)
def handle(model: Pipeline, records_df: DataFrame) -> DataFrame:
    ...
```
