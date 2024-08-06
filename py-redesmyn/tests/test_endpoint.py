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

from tests.common import DIR_PYREDESMYN, DIR_TESTS, request_prediction, serve_and_predict
from tests.fixtures.common import irises
from tests.fixtures.iris_model import Input, Output, SepalLengthPredictor, get_handle


def test_undertyped_handler_signature(irises: pl.DataFrame):
    # https://github.com/davidagold/redesmyn/issues/77
    @endpoint(path="/path/to/endpoint")
    def handle(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(prediction=1.0)

    server = Server()
    server.register(handle)
    response_by_run_id = {}

    async def callback(record, run_id: int):
        record["run_id"] = run_id
        response_by_run_id[run_id] = record

    def tasks(session: aiohttp.ClientSession, data: Dict) -> List[Callable[[], Coroutine]]:
        return [
            lambda: request_prediction(
                url="http://localhost:8080/path/to/endpoint",
                session=session,
                data=data,
                callback=lambda record: callback(record=record, run_id=i),
            )
            for i in range(10)
        ]

    coro = serve_and_predict(
        server=server, irises=irises, tasks=tasks, response_by_id=response_by_run_id
    )
    asyncio.run(coro)
    assert all(r["http_status_code"] == 200 for r in response_by_run_id.values())


class TestEndpoint:
    def test_validates_artifact_params(self, irises: pl.DataFrame):
        server = Server()
        server.register(get_handle(validate_artifact_params=True))
        response_by_run_id = {}

        async def callback(record, run_id: str):
            record["run_id"] = run_id
            response_by_run_id[run_id] = record

        def tasks(session: aiohttp.ClientSession, data: Dict) -> List[Callable[[], Coroutine]]:
            def task(run_id, stop: bool = False):
                return request_prediction(
                    session=session,
                    url=f"http://localhost:8080/predictions/{run_id}/000449a650df4e36844626e647d15664",
                    data=data,
                    callback=lambda record: callback(record, run_id=run_id),
                )

            return [
                lambda: task(run_id="903683212157180428"),
                lambda: task(run_id="invalid_run_id", stop=True),
            ]

        coro = serve_and_predict(
            server=server, irises=irises, tasks=tasks, response_by_id=response_by_run_id
        )
        asyncio.run(coro)
        assert response_by_run_id["903683212157180428"]["http_status_code"] == 200
        assert response_by_run_id["invalid_run_id"]["http_status_code"] == 422
        assert response_by_run_id["invalid_run_id"]["body"].startswith(
            "Python Error: ValidationError"
        )

    def test_unparametrized_handler(self, irises: pl.DataFrame):
        model = SepalLengthPredictor(
            model_uri=(
                DIR_TESTS
                / "fixtures/models/mlflow/iris/903683212157180428/000449a650df4e36844626e647d15664/artifacts/model"
            ).as_posix()
        )

        @endpoint(path="/predictions/iris")
        def handle(records_df: Input.DataFrame) -> Output.DataFrame:
            return model.include_predictions(records_df=records_df)

        server = Server()
        server.register(handle)
        response_by_id = {}

        def tasks(session: aiohttp.ClientSession, data: Dict) -> List[Callable[..., Coroutine]]:
            async def callback(record: Dict, req_id: int):
                record["request_id"] = req_id
                response_by_id[req_id] = record

            return [
                lambda: request_prediction(
                    url="http://localhost:8080/predictions/iris",
                    session=session,
                    data=data,
                    callback=lambda record: callback(record=record, req_id=i),
                )
                for i in range(10)
            ]

        coro = serve_and_predict(
            server=server, tasks=tasks, irises=irises, response_by_id=response_by_id
        )
        asyncio.run(coro)
        assert all(r["http_status_code"] == 200 for r in response_by_id.values())
        assert all(
            all(type(pred) == float for pred in body["predictions"])
            for body in (json.loads(r["body"]) for r in response_by_id.values())
        )
