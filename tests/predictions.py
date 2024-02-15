import asyncio
import time
from typing import Dict
import aiohttp


model_name = "model"
model_version = "1.0.0"
url = f"http://localhost:8080/predictions/{model_name}/{model_version}"


data = {
    "records": [
        {"a": 1, "b": 2},
        {"a": 1, "b": 2},
        {"a": 1, "b": 2},
        {"a": 1, "b": 2},
    ]
}

async def send_post_request(
    task_id: int,
    session: aiohttp.ClientSession,
    url: str,
    data: Dict,
    timings_ms_by_task_id: Dict[int, float],
):
    start = time.perf_counter_ns()
    async with session.post(url, json=data) as response:
        response_data = await response.text()
        timing_ms = (time.perf_counter_ns() - start) / 1e9
        timings_ms_by_task_id[task_id] = timing_ms


async def main():
    n_tasks = 1000
    timings_ms_by_task_id: Dict[int, float] = {}
    async with aiohttp.ClientSession() as session:
        async with asyncio.TaskGroup() as tg:
            for task_id in range(n_tasks):
                coroutine = send_post_request(
                    task_id=task_id,
                    session=session,
                    url=url,
                    data=data,
                    timings_ms_by_task_id=timings_ms_by_task_id
                )
                tg.create_task(coroutine)
    
    print(f"{len(timings_ms_by_task_id)=}")
    print(timings_ms_by_task_id)


if __name__ == '__main__':
    asyncio.run(main())

