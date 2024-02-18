import asyncio
from itertools import repeat
import time
from typing import Dict
import aiohttp
import polars as pl


model_name = "model"
model_version = "1.0.0"
url = f"http://localhost:8080/predictions/{model_name}/{model_version}"


record = {"a": 1, "b": 2}
data = list(repeat(record, 4))

async def send_post_request(
    task_id: int,
    session: aiohttp.ClientSession,
    url: str,
    data: Dict,
    records_by_task_id: Dict[int, Dict],
):
    start = time.perf_counter_ns()
    async with session.post(url, json=data) as response:
        response_data = await response.text()
        if isinstance(response_data, str):
            message = response_data
        else: 
            message = "success"
        timing_ms = (time.perf_counter_ns() - start) / 1e9
        record_request = {
            "status_code": str(response.status), "timing": timing_ms, "id": task_id, "message": message
        }
        records_by_task_id[task_id] = record_request


async def main():
    n_tasks = 10000
    records_by_task_id: Dict[int, Dict] = {}
    async with aiohttp.ClientSession() as session:
        async with asyncio.TaskGroup() as tg:
            for task_id in range(n_tasks):
                coroutine = send_post_request(
                    task_id=task_id,
                    session=session,
                    url=url,
                    data=data,
                    records_by_task_id=records_by_task_id
                )
                tg.create_task(coroutine)
    
    df = pl.DataFrame(list(records_by_task_id.values())).cast({"status_code": pl.Categorical})
    print(df.select("timing").describe())
    print(df.group_by("status_code", "message").len())


if __name__ == '__main__':
    asyncio.run(main())

