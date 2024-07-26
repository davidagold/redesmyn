import asyncio
import sys
from argparse import ArgumentParser
from datetime import timedelta
from pathlib import Path

import mlflow
import redesmyn.artifacts as afs
from common import project_dir
from redesmyn.py_redesmyn import LogConfig
from redesmyn.service import Server, endpoint
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from train import Input, Output, SepalLengthPredictor, SepalLengthPredictorSpec

LogConfig(path=Path("./logs/run.txt"), emf_path=Path("./logs/metrics.txt")).init()
mlflow.set_tracking_uri(project_dir() / "models/mlflow/iris")


@endpoint(
    path="/predictions/{run_id}/{model_id}",
    batch_max_delay_ms=10,
    batch_max_size=64,
    cache_config=afs.CacheConfig(
        client=afs.FsClient(
            base_path=Path(__file__).parent,
            path_template="/models/mlflow/iris/{run_id}/{model_id}/artifacts/model",
        ),
        load_model=lambda *args: SepalLengthPredictor(*args),
        spec=SepalLengthPredictorSpec,
        interval=timedelta(minutes=1),
    ),
)
def handle(model: SepalLengthPredictor, records_df: Input.DataFrame) -> Output.DataFrame:
    return model.include_predictions(records_df=records_df)


async def main():
    server = Server()
    server.register(handle)
    await server.serve()


asyncio.run(main())
