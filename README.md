## Redesmyn: Build ML inference servers with Python and Rust

Redesmyn (/ˈreɪd.smɪn/, REEDZ-min) helps you build services for real-time ML inference in Python and Rust:
* **Dual language**: Core Redesmyn functionality is written in Rust for safety and performance and exposed through interoperable Python and Rust APIs.
* **Predict in Python**: Seamlessly integrate prediction handlers written in Python with Rust server frameworks.
* **Serde deserialization**: Declare inference handler schemas via the Rust `Schema` trait or [Pydantic](https://docs.pydantic.dev/latest/) Python subclasses to use
    [strongly-typed Serde parsing](https://docs.rs/serde_json/latest/serde_json/#parsing-json-as-strongly-typed-data-structures) or
    [untyped Serde parsing](https://docs.rs/serde_json/latest/serde_json/#operating-on-untyped-json-values), respectively.
* **Built on Polars**: Request payloads are parsed into [Polars](https://pola.rs) DataFrames that can be passed to Python inference handlers with zero copy.
* **Ergonomic API**: Declare service endpoints with custom, parametrizable paths and conduct request validation with Pydantic models.
* **Asynchronous model cache**: Manage model caching and async updating via an integrated cache that maps URL parameters to model variants.
* **Observability**: Redesmyn applications can be configured to emit collated AWS EMF metrics log messages.

**NOTE**: Redesmyn is currently in active development targeting a v0.1 release, which is intended as the first iteration officially suitable for public use.
Some features described in the present README are aspirational and are included to give a sense of our intended direction for Redesmyn.
Such aspirational features are indicated by a "†" linking to the corresponding GitHub issue.
You can also follow our progress towards v0.1 on the [v0.1 Project Roadmap](https://github.com/users/davidagold/projects/7/views/1).

### Example

The following snippet, which is simplified from [this example](https://github.com/davidagold/redesmyn/tree/main/py-redesmyn/examples/iris), instantiates and runs a Redesmyn `Server` whose single `Endpoint` is managed by an inference handler that receives batched inference requests as a Polars DataFrame and accesses a cached `sklearn` model parametrized by `run_id` and `model_id`:

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

To handle incoming inference requests, we must create an `Endpoint`.
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

The handler function itself is just a Python function that expects a Polars `DataFrame` argument.
The `DataFrame` contains records from the present batch of inference requests, which Redesmyn deserializes and aggregates for you.
Thanks to Polars' use of Arrow and PyO3, the Rust-based server functionality and Python-based inference functionality interoperate with zero IPC or copying of data.
You can also customize or opt out of Redesmyn's automatic deserialization in favor of receiving the request object directly. [†](https://github.com/davidagold/redesmyn/issues/89)

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

You can declare input and output schemas for an `Endpoint` handler function by subclassing the `Schema` class:

```python
class Input(Schema):
    sepal_width: pl.Float64
    petal_length: pl.Float64
    petal_width: pl.Float64


class Output(Schema):
    sepal_length: pl.Float64


@endpoint(path="/predictions/iris")
def handle(records_df: Input.DataFrame) -> Output.DataFrame:
    return records_df.select(sepal_length=pl.Series(model.predict(X=records_df)))

```
`Schema`, and therefore any descendant, is a subclass of Pydantic's `BaseModel`.
To indicate that a handler argument or return type annotation is a Polars `DataFrame` expected to conform to a given `Schema` subclass, simply type the object using `Schema.DataFrame` class property as above.
This property of `Schema`'s metaclass is equivalent to `Annotated[polars.DataFrame, cls]`, where `cls` is the present `Schema` subclass.
Thus, annotating a parameter or return type with `Schema.DataFrame` both indicates to type checkers that the object itself is expected to be of type `polars.DataFrame` and enables dynamic inspection of the annotated `DataFrame`'s expected fields.

There are two primary uses for `Schema.DataFrame` annotations as above:
1. Hinting which fields are expected during request deserialization:
If `Schema.DataFrame` annotations such as above are present in the inference handler's signature, Redesmyn will deserialize only those fields specified in the input `Schema` and ignore all others.
2. Validating incoming prediction requests[†](https://github.com/davidagold/redesmyn/issues/90): You can configure a Redesmyn `Endpoint` to return an HTTP 422 response if either
(i) an expected field from the `Schema.DataFrame` annotation is missing in a record, or
(ii) an unexpected field is present.


## Model parametrizations and `Cache`

Often we wish to deploy many models indexed by some set of parametrizations.
For instance, we may train a different model for each of a subset of ISO 3166-2 codes and a general fallback model for the parent ISO 3166-1 code.
You can configure a Redesmyn endpoint to accept URL parameters that correspond to the parameters that index distinct models and to pass its respective handler the appropriate model from a model `Cache`.
The `Cache` itself is in turn configured to retrieve models from a local filestore or a remote object store according to such parametrizations.

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

The above `Endpoint` coordinates with its respective `Cache`, whose configuration is specified by the `CacheConfig`, to pass the appropriate `Pipeline` model to the handler given the requested values of `iso_3166_1` and `iso_3166_2`.


### `Cache` updates



## `ArtifactSpec`



## `Server`
