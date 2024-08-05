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
from redesmyn import service as svc
from sklearn.utils.discovery import itemgetter

from tests.common import DIR_PYREDESMYN, DIR_TESTS, request_prediction, serve_and_predict
from tests.fixtures.common import irises
from tests.fixtures.iris_model import (
    Input,
    Output,
    SepalLengthPredictor,
    SepalLengthPredictorSpec,
    get_handle,
)


def test_handles_invalid_cache(irises: pl.DataFrame):
    @svc.endpoint(
        path="/invalid/predictions/{run_id}/{model_id}",
        batch_max_delay_ms=10,
        batch_max_size=64,
        cache_config=afs.CacheConfig(
            client=afs.FsClient(
                base_path=Path("/this/path/is/not/valid"),
                path_template="/{run_id}/{model_id}/artifacts/model",
            ),
            load_model=lambda *args: SepalLengthPredictor(*args),
            spec=SepalLengthPredictorSpec,
        ),
    )
    def handle_with_broken_cache(
        model: SepalLengthPredictor, df: Input.DataFrame
    ) -> Output.DataFrame:
        return model.include_predictions(records_df=df)

    server = svc.Server()
    # Register both broken and working handlers
    server.register(handle_with_broken_cache)
    server.register(get_handle())

    response_by_id = {}

    def tasks(session: aiohttp.ClientSession, data: Dict) -> List[Callable[[], Coroutine]]:
        async def callback(record: Dict, req_id: int):
            record["request_id"] = req_id
            response_by_id[req_id] = record

        return [
            lambda: request_prediction(
                url="http://localhost:8080/predictions/903683212157180428/000449a650df4e36844626e647d15664",
                session=session,
                data=data,
                callback=lambda record: callback(record=record, req_id=i),
            )
            for i in range(10)
        ]

    coro = serve_and_predict(server=server, tasks=tasks, irises=irises, response_by_id=response_by_id)
    asyncio.run(coro)
    print(response_by_id)
    assert all(r["http_status_code"] == 200 for r in response_by_id.values())
    assert all(
        all(type(pred) == float for pred in body["predictions"])
        for body in (json.loads(r["body"]) for r in response_by_id.values())
    )
