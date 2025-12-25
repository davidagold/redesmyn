from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import desc, select

from redesmyn.context import RepoContext
from redesmyn.db import Event, Pause, create_engine, create_sessionmaker
from redesmyn.domain.enums import PauseMode


class NotInitializedError(RuntimeError):
    pass


def _ensure_initialized(ctx: RepoContext) -> None:
    if not ctx.db_path.exists():
        raise NotInitializedError("Redesmyn is not initialized in this repo. Run `rn init`.")


async def get_effective_pause(ctx: RepoContext, *, branch: str | None) -> Pause | None:
    if branch:
        pause = await get_active_pause(ctx, scope=f"branch:{branch}")
        if pause is not None:
            return pause
    return await get_active_pause(ctx, scope="repo")


async def get_active_pause(ctx: RepoContext, *, scope: str) -> Pause | None:
    _ensure_initialized(ctx)

    engine = create_engine(ctx.db_path)
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            stmt = (
                select(Pause)
                .where(Pause.scope == scope, Pause.cleared_at.is_(None))
                .order_by(desc(Pause.id))
                .limit(1)
            )
            return await session.scalar(stmt)
    finally:
        await engine.dispose()


def _normalize_mode(mode: str | PauseMode) -> PauseMode:
    if isinstance(mode, PauseMode):
        return mode
    return PauseMode(mode)


async def set_pause(ctx: RepoContext, *, scope: str, mode: str | PauseMode, reason: str | None) -> Pause:
    _ensure_initialized(ctx)

    normalized = _normalize_mode(mode)

    engine = create_engine(ctx.db_path)
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            active = await session.scalar(
                select(Pause)
                .where(Pause.scope == scope, Pause.cleared_at.is_(None))
                .order_by(desc(Pause.id))
                .limit(1)
            )
            if active is not None:
                active.cleared_at = datetime.now(UTC)
                active.cleared_reason = "superseded"

            pause = Pause(scope=scope, mode=normalized, reason=reason, created_at=datetime.now(UTC))
            session.add(pause)
            session.add(
                Event(
                    event_type="pause.set",
                    payload={"scope": scope, "mode": normalized.value, "reason": reason},
                    created_at=datetime.now(UTC),
                )
            )
            await session.commit()
            await session.refresh(pause)
            return pause
    finally:
        await engine.dispose()


async def clear_pause(ctx: RepoContext, *, scope: str, reason: str | None) -> Pause | None:
    _ensure_initialized(ctx)

    engine = create_engine(ctx.db_path)
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            pause = await session.scalar(
                select(Pause)
                .where(Pause.scope == scope, Pause.cleared_at.is_(None))
                .order_by(desc(Pause.id))
                .limit(1)
            )
            if pause is None:
                return None

            pause.cleared_at = datetime.now(UTC)
            pause.cleared_reason = reason
            session.add(
                Event(
                    event_type="pause.cleared",
                    payload={"scope": scope, "reason": reason},
                    created_at=datetime.now(UTC),
                )
            )
            await session.commit()
            await session.refresh(pause)
            return pause
    finally:
        await engine.dispose()


async def list_pauses(ctx: RepoContext, *, scope: str) -> list[Pause]:
    _ensure_initialized(ctx)

    engine = create_engine(ctx.db_path)
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            rows = await session.scalars(select(Pause).where(Pause.scope == scope).order_by(desc(Pause.id)))
            return list(rows)
    finally:
        await engine.dispose()
