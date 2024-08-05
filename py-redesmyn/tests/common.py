import asyncio
import json
from pathlib import Path
from typing import Callable, Coroutine, Dict, List, Optional

import aiohttp
import polars as pl
from more_itertools import first
from redesmyn import service as svc

DIR_TESTS = Path(__file__).parent
DIR_PYREDESMYN = DIR_TESTS.parent
DIR_PROJECT = DIR_PYREDESMYN.parent


async def request_prediction(
    session: aiohttp.ClientSession,
    url: str,
    data: Dict,
    callback: Optional[Callable[..., Coroutine]] = None,
):
    resp = await session.post(url, json=[json.dumps(data)])
    record = {"http_status_code": resp.status, "body": await resp.text()}
    if callback:
        await callback(record)


async def serve_and_predict(
    server: svc.Server,
    tasks: Callable[..., List[Coroutine]],
    irises: pl.DataFrame,
    response_by_id: Dict[str, Dict],
):
    data = first(list(irises.iter_rows(named=True)))
    async with aiohttp.ClientSession() as session, asyncio.TaskGroup() as server_tg:
        server_tg.create_task(server.serve())
        async with asyncio.TaskGroup() as inner_tg:
            [inner_tg.create_task(t) for t in tasks(session=session, data=data)]

        server_tg.create_task(server.stop())
