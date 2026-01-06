from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

import structlog


class BackgroundTaskManager:
    def __init__(self, *, logger: structlog.BoundLogger | None = None) -> None:
        self._log = logger or structlog.get_logger("redesmyn.background_tasks")
        self._tasks: set[asyncio.Task[object]] = set()

    @property
    def tasks(self) -> set[asyncio.Task[object]]:
        return self._tasks

    def spawn(
        self, coro: Coroutine[Any, Any, object], *, name: str
    ) -> asyncio.Task[object]:
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)

        def _done(t: asyncio.Task[object]) -> None:
            self._tasks.discard(t)
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                self._log.error(
                    "background_task.failed",
                    task=name,
                    error=str(exc),
                )

        task.add_done_callback(_done)
        return task

    async def cancel_and_await(self) -> None:
        tasks = list(self._tasks)
        if not tasks:
            return
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
