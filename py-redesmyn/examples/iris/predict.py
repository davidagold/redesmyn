import asyncio
import json
import time
from typing import Dict

import aiohttp
import polars as pl
from common import load_irises
from train import Input


async def send_post_request(
    task_id: int,
    session: aiohttp.ClientSession,
    url: str,
    data: Dict,
    records_by_task_id: Dict[int, Dict],
):
    start = time.perf_counter_ns()
    async with session.post(url, json=[json.dumps(data)]) as response:
        if response.status != 200:
            print(response)
        message = await response.text()
        timing_ms = (time.perf_counter_ns() - start) / 1e9
        record = json.loads(message)
        record_request = {"timing": timing_ms, **record}
        records_by_task_id[task_id] = record_request


async def main():
    url = "http://localhost:8080/predictions/903683212157180428/000449a650df4e36844626e647d15664"
    records_by_task_id: Dict[int, Dict] = {}
    irises = load_irises()

    async with aiohttp.ClientSession() as session:
        async with asyncio.TaskGroup() as tg:
            for idx, row in enumerate(irises.iter_rows()):
                data = dict(zip(Input.field_names(), row))
                coro = send_post_request(
                    task_id=idx,
                    session=session,
                    url=url,
                    data=data,
                    records_by_task_id=records_by_task_id,
                )
                tg.create_task(coro)

    df = pl.DataFrame(list(records_by_task_id.values()))
    print(df.describe())
    print(df.head())


if __name__ == "__main__":
    asyncio.run(main())
