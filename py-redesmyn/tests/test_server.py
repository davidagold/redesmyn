import asyncio
from typing import Optional

import polars as pl
from handlers.model import Model
from redesmyn.endpoint import endpoint
from redesmyn.schema import Schema
from redesmyn.server import Server


class Input(Schema):
    a = pl.Float64()
    b = pl.Float64()


class Output(Schema):
    prediction = pl.Float64()


@endpoint(
    path="predictions/{model_name}/{model_version}",
    batch_max_delay_ms=10,
    batch_max_size=64,
)
def handler(records_df: Input.DataFrame, run_id: Optional[str] = None) -> Output.DataFrame:
    return Model().load(run_id=run_id).predict(records_df=records_df)


async def main():
    server = Server()
    server.register(handler)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
