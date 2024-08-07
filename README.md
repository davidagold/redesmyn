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

To illustrate, the following snippet, which is simplified from [this example](https://github.com/davidagold/redesmyn/tree/main/py-redesmyn/examples/iris), instantiates and runs a Redesmyn `Server` whose single `Endpoint` is managed by an inference handler that receives batched inference requests as a Polars DataFrame and accesses a cached `sklearn` model parametrized by `run_id` and `model_id`:

```python
import asyncio

import mlflow
import polars as pl
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
def handle(model: Pipeline, records_df: pl.DataFrame) -> pl.DataFrame:
    return model.predict(X=records_df)


async def main():
    server = svc.Server()
    server.register(endpoint=handle)
    await server.serve()


asyncio.run(main())
```

If we run the above, we can make requests against the endpoint as follows:

```
$ curl -X POST -d '["{\"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}"]' \
    -H 'Content-Type: application/json' \
    http://localhost:8080/predictions/903683212157180428/000449a650df4e36844626e647d15664
{"id":"0e1ae8ba-f1fe-42fb-956e-882f222f503f","predictions":[5.014526282601766]}%
```


## `Endpoint`

To handle incoming inference requests, we must first register an `Endpoint`.
As a Redesmyn server is just an [Actix](https://actix.rs/docs/) HTTP server, each such `Endpoint` is associated with a path that can be configured in the specification of the `Endpoint` handler:

```python
model = mlflow.sklearn.load_model(model_uri=...)

@svc.endpoint(path="/predictions/iris")
def handle(records_df: pl.DataFrame) -> pl.DataFrame:
    return model.predict(X=records_df)
```

The `path` parameter is customizable.
As demonstrated in the introductory [example](#example) above, paths also support URL parameters, which designate model parametrizations.
We'll discuss how to use such functionality in the [model parametrizations and cache section](#model-parametrizations-and-cache) below.

The handler function itself is just a Python function that expects a Polars `DataFrame` argument, which contains the present batch of inference requests.
Redesmyn takes care of deserializing incoming requests into Polars rows and batching the latter into a `DataFrame`.
Thanks to Polars' use of Arrow and PyO3, the Rust-based server functionality and Python-based inference functionality are interoperable with zero IPC or copying of data.

You can modify the batching behavior with the following parameters:

```python
@svc.endpoint(
    path="/predictions/iris",
    batch_max_delay_ms=10,
    batch_max_size=64,
)
def handle(records_df: pl.DataFrame) -> pl.DataFrame:
    ...
```



## `Schema`




## Model parametrizations and `Cache`

Often we wish to deploy many models indexed by some set of parametrizations.
For instance, we may train a different model for a subset of ISO 3166-2 codes and a general fallback model for the parent ISO 3166-1 code.
You can configure a Redesmyn endpoint to accept URL parameters that correspond to those that index distinct models and to pass its respective handler the appropriate model from a model `Cache`.
The `Cache` itself is in turn configured to retrieve models -- for instance, from a local filestore or a remote object store -- according to such parametrizations.

URL-based model parametrizations and model `Cache` functionality go hand in hand, so we'll explore them simultaneously.
In the following example, we specify both an `Endpoint` whose path contains URL parameters `ios_3166_1` and `iso_3166_2`
as well as an `FsClient` (file system client) whose `path_template` contains the same parameters.

```python
@svc.endpoint(
    path="/predictions/transaction/{iso_3166_1}/{iso_3166_2}/",
    cache_config=afs.CacheConfig(
        client=afs.FsClient(
            base_path=Path(__file__).parent / "models/mlflow/transaction",
            path_template="/{iso_3166_1}/{iso_3166_2}/artifacts/model",
        ),
        load_model=mlflow.sklearn.load_model,
    ),
)
def handle(model: Pipeline, records_df: pl.DataFrame) -> pl.DataFrame:
    return model.predict(X=records_df)
```

The `Endpoint` coordinates with its respective `Cache`, whose configuration is specified by the `CacheConfig`, to pass the appropriate `Pipeline` model to the handler given the requested values of `iso_3166_1` and `iso_3166_2`.


### `Cache` updates



## `ArtifactSpec`



## `Server`
