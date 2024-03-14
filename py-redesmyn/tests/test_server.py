import asyncio
import polars as pl
from redesmyn.schema import Schema
from redesmyn.py_redesmyn import Endpoint
from redesmyn.server import Server

async def main():
    schema_in = pl.Struct({
        "a": pl.Float64(),
        "b": pl.Float64(),
    })
    schema_out = pl.Struct({
        "prediction": pl.Float64()
    })

    endpoint = Endpoint(
        signature=(schema_in, schema_out),
        path="predictions/{model_name}/{model_version}",
        handler="handlers.model:handle",
    )

    server = Server()
    server.register(endpoint)

    server.serve()