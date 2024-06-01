import asyncio
import sys
from argparse import ArgumentParser
from pathlib import Path

import mlflow
from common import project_dir
from redesmyn.service import Server, endpoint
from sklearn.pipeline import Pipeline
from train import Input, Output, SepalLengthPredictor

mlflow.set_tracking_uri(project_dir() / "models/mlflow/iris")
arg_parser = ArgumentParser()
arg_parser.add_argument("run_id", type=str)
args = arg_parser.parse_args()
pipeline = SepalLengthPredictor(run_id=args.run_id)


@endpoint(
    path="/predictions/{model_name}/{model_version}",
    batch_max_delay_ms=10,
    batch_max_size=64,
)
def handle(records_df: Input.DataFrame) -> Output.DataFrame:
    return pipeline.include_predictions(records_df=records_df)


async def main():
    server = Server()
    server.register(handle)
    await server.serve()


asyncio.run(main())
