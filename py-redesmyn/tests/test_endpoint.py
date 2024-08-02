import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Callable, Coroutine, Dict, List, Optional
from unittest import mock

import aiohttp
import mlflow
import polars as pl
import pytest
from more_itertools import first
from redesmyn import artifacts as afs
from redesmyn.py_redesmyn import LogConfig
from redesmyn.service import Server, endpoint
from sklearn.utils.discovery import itemgetter

from tests.fixtures.iris_model import SepalLengthPredictor, get_handle

PROJECT_DIR = Path(__file__).parent.parent


@pytest.fixture()
def irises() -> pl.DataFrame:
    return pl.read_csv(PROJECT_DIR / "examples/iris/data/iris.csv")


async def request_prediction(
    session: aiohttp.ClientSession,
    run_id: str,
    data: Dict,
    response_by_run_id: Dict[str, Dict],
    callback: Optional[Callable[..., Coroutine]] = None,
):
    url = f"http://localhost:8080/predictions/{run_id}/000449a650df4e36844626e647d15664"
    async with session.post(url, json=[json.dumps(data)]) as resp:
        record = {"http_status_code": resp.status, "run_id": run_id, "body": await resp.text()}
        response_by_run_id[run_id] = record

    if callback:
        await callback()


async def serve_and_predict(
    server: Server, irises: pl.DataFrame, response_by_run_id: Dict[str, Dict]
):
    field_names = [
        "sepal_width",
        "petal_length",
        "petal_width",
    ]
    data = first(list(irises.iter_rows(named=True)))
    print(data)
    async with aiohttp.ClientSession() as session:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(server.serve())
            tg.create_task(
                request_prediction(
                    session=session,
                    run_id="903683212157180428",
                    data=data,
                    response_by_run_id=response_by_run_id,
                )
            )
            tg.create_task(
                request_prediction(
                    session=session,
                    run_id="invalid_run_id",
                    data=data,
                    response_by_run_id=response_by_run_id,
                    callback=server.stop,
                )
            )


class TestEndpoint:
    @mock.patch.dict(os.environ, {"RUST_LOG": "debug"})
    def test_validates_artifact_params(self, irises: pl.DataFrame):
        dir_logs = Path(__file__).parent / "logs"
        dir_logs.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=dir_logs, delete=False) as f:
            LogConfig(path=Path(f.name)).init()

        server = Server()
        server.register(get_handle(validate_artifact_params=True))
        response_by_run_id = {}
        coro = serve_and_predict(server=server, irises=irises, response_by_run_id=response_by_run_id)
        asyncio.run(coro)
        assert response_by_run_id["903683212157180428"]["http_status_code"] == 200
        assert response_by_run_id["invalid_run_id"]["http_status_code"] == 422
        assert response_by_run_id["invalid_run_id"]["body"].startswith(
            "Python Error: ValidationError"
        )

    @mock.patch.dict(os.environ, {"RUST_LOG": "debug"})
    def test_unparametrized_handler(self, irises: pl.DataFrame):
        dir_logs = Path(__file__).parent / "logs"
        dir_logs.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=dir_logs, delete=False) as f:
            LogConfig(path=Path(f.name)).init()

        model = SepalLengthPredictor(
            model_uri=(
                PROJECT_DIR
                / "examples/iris/models/mlflow/iris/903683212157180428/000449a650df4e36844626e647d15664/artifacts/model"
            ).as_posix()
        )

        @endpoint(path="/predictions/iris")
        def handle(records_df: pl.DataFrame) -> pl.DataFrame:
            return model.include_predictions(records_df=records_df)

        # server = Server()
        # server.register(get_handle(validate_artifact_params=True))
        # response_by_run_id = {}
        # coro = serve_and_predict(server=server, irises=irises, response_by_run_id=response_by_run_id)
        # asyncio.run(coro)
        # assert response_by_run_id["903683212157180428"]["http_status_code"] == 200
        # assert response_by_run_id["invalid_run_id"]["http_status_code"] == 422
        # assert response_by_run_id["invalid_run_id"]["body"].startswith(
        #     "Python Error: ValidationError"
        # )
