import asyncio
import re
from typing import Annotated, Any

import polars as pl
from annotated_types import Predicate
from handlers.model import Model
from pydantic import BaseModel

from redesmyn import artifacts as afs
from redesmyn import service as svc


class Input(svc.Schema):
    a = pl.Float64()
    b = pl.Float64()


class Output(svc.Schema):
    prediction = pl.Float64()


@afs.artifact_spec(
    load_fn=lambda run_id: Model().load(run_id=run_id),
    cache_path="s3://model-bucket/{model_name}/{model_version}/",
)
class ModelArtifact(BaseModel):
    @staticmethod
    def validate_version(v: Any) -> bool:
        version_match = re.match(r"v^\d\.\d\.\d", v)
        return True if version_match is not None else False

    model_name: str
    model_version: Annotated[str, Predicate(validate_version)]
    run_id: Annotated[str, afs.LatestKey]


@svc.endpoint(
    path="/predictions/{model_name}/{model_version}",
    cache=afs.ModelCache[ModelArtifact, Model](client=afs.FsClient()),
    batch_max_delay_ms=10,
    batch_max_size=64,
)
def handler(model: Model, records_df: Input.DataFrame) -> Output.DataFrame:
    return model.predict(records_df=records_df)


async def main():
    server = svc.Server()
    server.register(handler)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
