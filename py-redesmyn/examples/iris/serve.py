import asyncio
import sys
from argparse import ArgumentParser
from pathlib import Path

import mlflow
import redesmyn.artifacts as afs
from common import project_dir
from redesmyn.service import Server, endpoint
from sklearn.pipeline import Pipeline
from tests.test_server import VersionedModelSpec
from train import Input, Output, SepalLengthPredictor

mlflow.set_tracking_uri(project_dir() / "models/mlflow/iris")
arg_parser = ArgumentParser()
arg_parser.add_argument("run_id", type=str)
args = arg_parser.parse_args()
pipeline = SepalLengthPredictor(run_id=args.run_id)


@endpoint(
    path="/predictions/{model_version}",
    batch_max_delay_ms=10,
    batch_max_size=64,
    cache=afs.ModelCache(
        client=afs.FsClient(fetch_as=afs.FetchAs.Uri),
        # TODO: Conceptually, this field makes more sense as part of the client (also, this shouldn't be an S3 path)
        path=afs.path("s3://model-bucket/{model_version}/"),
        spec=VersionedModelSpec,
        refresh=afs.Cron(schedule="0 * * * * *"),
    ),
)
def handle(model: SepalLengthPredictor, records_df: Input.DataFrame) -> Output.DataFrame:
    return pipeline.include_predictions(records_df=records_df)


async def main():
    server = Server()
    server.register(handle)
    await server.serve()


asyncio.run(main())
