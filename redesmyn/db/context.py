from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from redesmyn.db.session import create_engine, create_sessionmaker, init_db


@dataclass(frozen=True, slots=True)
class DbContext:
    engine: AsyncEngine
    sessionmaker: async_sessionmaker[AsyncSession]


@asynccontextmanager
async def open_db(
    *,
    db_path: Path,
    migrate: bool,
) -> AsyncIterator[DbContext]:
    engine = create_engine(db_path)
    try:
        await init_db(engine, migrate=migrate)
        sessionmaker = create_sessionmaker(engine)
        yield DbContext(engine=engine, sessionmaker=sessionmaker)
    finally:
        await engine.dispose()
