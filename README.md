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
