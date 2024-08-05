import asyncio
import json
import logging
from pathlib import Path
from typing import Callable, Coroutine, Dict, List, Optional, Type

import aiohttp
import polars as pl
from more_itertools import first
from redesmyn import service as svc

DIR_TESTS = Path(__file__).parent
DIR_PYREDESMYN = DIR_TESTS.parent
DIR_PROJECT = DIR_PYREDESMYN.parent


async def retry(
    coro_fn: Callable[[], Coroutine],
    retryable: List[Type[Exception]],
    exp_backoff_coef_ms: int = 1000,
    max_num_attempts: int = 3,
):
    num_attempts = 0
    while num_attempts <= max_num_attempts:
        num_attempts += 1
        coro = coro_fn()
        try:
            return await coro
        except Exception as e:
            if type(e) in retryable:
                wait_ms = exp_backoff_coef_ms * 2 ** (num_attempts - 1)
                num_attempts_remaining = max_num_attempts - num_attempts
                logging.warn(f"Encountered exception {e} while trying {coro}")
                await asyncio.sleep(wait_ms / 1000)
            else:
                raise e


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
    tasks: Callable[..., List[Callable[[], Coroutine]]],
    irises: pl.DataFrame,
    response_by_id: Dict[str, Dict],
):
    data = first(list(irises.iter_rows(named=True)))
    async with aiohttp.ClientSession() as session, asyncio.TaskGroup() as server_tg:
        server_tg.create_task(server.serve())
        async with asyncio.TaskGroup() as inner_tg:
            [
                inner_tg.create_task(retry(t, retryable=[aiohttp.client.ClientConnectorError]))
                for t in tasks(session=session, data=data)
            ]

        server_tg.create_task(server.stop())
