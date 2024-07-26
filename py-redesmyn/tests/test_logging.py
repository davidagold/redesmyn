import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Dict
from unittest import mock

import aiohttp
import mlflow
import polars as pl
import pytest
from redesmyn import artifacts as afs
from redesmyn.py_redesmyn import LogConfig
from redesmyn.service import Server, endpoint

from tests.fixtures.iris_model import get_handle

PROJECT_DIR = Path(__file__).parent.parent


@pytest.fixture()
def irises() -> pl.DataFrame:
    return pl.read_csv(PROJECT_DIR / "examples/iris/data/iris.csv")


async def send_post_request(
    task_id: int,
    session: aiohttp.ClientSession,
    url: str,
    data: Dict,
    records_by_task_id: Dict[int, Dict],
):
    async with session.post(url, json=[json.dumps(data)]) as response:
        message = await response.text()
        record_request = {"http_status_code": response.status, **json.loads(message)}
        records_by_task_id[task_id] = record_request


async def predict_then_stop(server: Server, irises: pl.DataFrame):
    url = "http://localhost:8080/predictions/903683212157180428/000449a650df4e36844626e647d15664"
    field_names = [
        "sepal_width",
        "petal_length",
        "petal_width",
    ]
    async with aiohttp.ClientSession() as session:
        records_by_task_id: Dict[int, Dict] = {}
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(
                    send_post_request(
                        task_id=idx,
                        session=session,
                        url=url,
                        data=dict(zip(field_names, row)),
                        records_by_task_id=records_by_task_id,
                    )
                )
                for idx, row in enumerate(irises.iter_rows())
            ]

    await asyncio.sleep(2)
    await server.stop()
    results = pl.DataFrame(data=list(records_by_task_id.values()))
    assert all(code == 200 for code in results["http_status_code"])


class TestLogging:
    async def _serve_and_predict(self, server: Server, irises: pl.DataFrame):
        loop = asyncio.get_running_loop()
        async with asyncio.TaskGroup() as tg:
            tg.create_task(server.serve())
            tg.create_task(predict_then_stop(server, irises))

    @mock.patch.dict(os.environ, {"RUST_LOG": "info"})
    def test_init(self, irises: pl.DataFrame):
        dir_logs = Path(__file__).parent / "logs"
        dir_logs.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=dir_logs, delete=True) as f:
            LogConfig(path=Path(f.name)).init()

            server = Server()
            server.register(get_handle())
            asyncio.run(self._serve_and_predict(server=server, irises=irises))

            traces = [json.loads(line.decode("utf8")) for line in f.readlines()]
            assert len(traces) > 0
            assert all("timestamp" in trace for trace in traces)
