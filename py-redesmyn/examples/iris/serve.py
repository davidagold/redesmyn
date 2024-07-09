import asyncio
import sys
from argparse import ArgumentParser
from pathlib import Path

import mlflow
import redesmyn.artifacts as afs
from sklearn.linear_model import Lasso
from common import project_dir
from redesmyn.service import Server, endpoint
from sklearn.pipeline import Pipeline
from train import Input, Output, SepalLengthPredictor, SepalLengthPredictorSpec
from redesmyn.py_redesmyn import LogConfig


LogConfig(path=Path("./logs/run.txt")).init()
mlflow.set_tracking_uri(project_dir() / "models/mlflow/iris")
# arg_parser = ArgumentParser()
# arg_parser.add_argument("run_id", type=str)
# args = arg_parser.parse_args()


@endpoint(
    path="/predictions/{run_id}/{model_id}",
    batch_max_delay_ms=10,
    batch_max_size=64,
    cache_config=afs.CacheConfig(
        client=afs.FsClient(base_path=Path(__file__).parent, path_template="/models/mlflow/iris/{run_id}/{model_id}/artifacts/model"),
        load_model=lambda *args: SepalLengthPredictor(*args),
        spec=SepalLengthPredictorSpec,
        # client=afs.FsClient(fetch_as=afs.FetchAs.Uri),
        # TODO: Conceptually, this field makes more sense as part of the client (also, this shouldn't be an S3 path)
        # path=afs.path("s3://model-bucket/{model_version}/"),
        # spec=VersionedModelSpec,
        # refresh=afs.Cron(schedule="0 * * * * *"),
    ),
)
def handle(model: SepalLengthPredictor, records_df: Input.DataFrame) -> Output.DataFrame:
    return model.include_predictions(records_df=records_df)


async def main():
    server = Server()
    server.register(handle)
    await server.serve()


asyncio.run(main())
