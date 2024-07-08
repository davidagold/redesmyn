import asyncio
from pathlib import Path
import re
from typing import Annotated, Self, TypeVar

import mlflow
import polars as pl
from handlers.model import Model
from pydantic import model_validator
from redesmyn import artifacts as afs
from redesmyn import service as svc


class Input(svc.Schema):
    a = pl.Float64()
    b = pl.Float64()


class Output(svc.Schema):
    prediction = pl.Float64()


M = TypeVar("M")

class VersionedModelSpec(afs.ArtifactSpec[M]):
    model_version: str
    run_id: Annotated[str, afs.LatestKey]

    @model_validator(mode="after")
    def check_version(self) -> Self:
        if re.match(r"v^\d\.\d\.\d", self.model_version) is None:
            raise ValueError(f"'{self.model_version}' is not a valid version.")

        return self


@svc.endpoint(
    path="/predictions/{model_version}",
    cache=afs.ModelCache(
        client=afs.FsClient(base_path=Path("."), path_template="s3://model-bucket/{model_version}/"),
        # path=afs.path("s3://model-bucket/{model_version}/"),
        spec=VersionedModelSpec,
        # refresh=afs.Cron(schedule="0 * * * * *"),
        load_model=mlflow.sklearn.load_model,
    ),
    batch_max_delay_ms=10,
    batch_max_size=64,
)
def handler(model: Model, records: Input.DataFrame) -> Output.DataFrame:
    return model.predict(records_df=records)


async def main():
    server = svc.Server().register(handler)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
