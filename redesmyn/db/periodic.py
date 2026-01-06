from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from redesmyn.db.sqlite_lock import is_sqlite_database_locked_error, sqlite_lock_backoff_s


async def run_periodic_db_task(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    interval_s: float,
    once: bool,
    name: str,
    step: Callable[[AsyncSession], Awaitable[Any]],
    logger: structlog.BoundLogger | None = None,
) -> None:
    log = logger or structlog.get_logger(f"redesmyn.{name}")
    attempt = 0
    while True:
        try:
            async with sessionmaker() as session:
                try:
                    await step(session)
                except Exception:
                    await session.rollback()
                    raise
            attempt = 0
        except asyncio.CancelledError:
            raise
        except OperationalError as e:
            if is_sqlite_database_locked_error(e):
                log.warning(
                    f"{name}.db_locked",
                    attempt=attempt,
                    interval_s=interval_s,
                )
                await asyncio.sleep(sqlite_lock_backoff_s(attempt))
                attempt += 1
                continue
            raise
        except Exception:
            log.exception(f"{name}.loop_failed")
            await asyncio.sleep(1.0)

        if once:
            return
        await asyncio.sleep(interval_s)

