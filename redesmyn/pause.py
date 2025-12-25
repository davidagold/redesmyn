from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import desc, select

from redesmyn.context import RepoContext
from redesmyn.db import Event, Pause, PauseScope, create_engine, create_sessionmaker
from redesmyn.domain.enums import PauseMode


class NotInitializedError(RuntimeError):
    pass


def _ensure_initialized(ctx: RepoContext) -> None:
    if not ctx.db_path.exists():
        raise NotInitializedError("Redesmyn is not initialized in this repo. Run `rn init`.")


def _normalize_scope(scope: str | PauseScope) -> PauseScope:
    if isinstance(scope, PauseScope):
        return scope
    if scope == "repo":
        return PauseScope.for_repo()
    if scope.startswith("branch:"):
        return PauseScope.for_branch(scope.removeprefix("branch:"))
    raise ValueError(f"Unknown pause scope: {scope}")


async def get_effective_pause(ctx: RepoContext, *, branch: str | None) -> Pause | None:
    if branch:
        pause = await get_active_pause(ctx, scope=PauseScope.for_branch(branch))
        if pause is not None:
            return pause
    return await get_active_pause(ctx, scope=PauseScope.for_repo())


async def get_active_pause(ctx: RepoContext, *, scope: str | PauseScope) -> Pause | None:
    _ensure_initialized(ctx)
    normalized_scope = _normalize_scope(scope)

    engine = create_engine(ctx.db_path)
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            stmt = (
                select(Pause)
                .where(Pause.scope == normalized_scope, Pause.cleared_at.is_(None))
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


async def set_pause(ctx: RepoContext, *, scope: str | PauseScope, mode: str | PauseMode, reason: str | None) -> Pause:
    _ensure_initialized(ctx)

    normalized_scope = _normalize_scope(scope)
    normalized = _normalize_mode(mode)

    engine = create_engine(ctx.db_path)
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            active = await session.scalar(
                select(Pause)
                .where(Pause.scope == normalized_scope, Pause.cleared_at.is_(None))
                .order_by(desc(Pause.id))
                .limit(1)
            )
            if active is not None:
                active.cleared_at = datetime.now(UTC)
                active.cleared_reason = "superseded"

            pause = Pause(scope=normalized_scope, mode=normalized, reason=reason, created_at=datetime.now(UTC))
            session.add(pause)
            session.add(
                Event(
                    event_type="pause.set",
                    data={"scope": normalized_scope.to_dict(), "mode": normalized.value, "reason": reason},
                    created_at=datetime.now(UTC),
                )
            )
            await session.commit()
            await session.refresh(pause)
            return pause
    finally:
        await engine.dispose()


async def clear_pause(ctx: RepoContext, *, scope: str | PauseScope, reason: str | None) -> Pause | None:
    _ensure_initialized(ctx)

    normalized_scope = _normalize_scope(scope)
    engine = create_engine(ctx.db_path)
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            pause = await session.scalar(
                select(Pause)
                .where(Pause.scope == normalized_scope, Pause.cleared_at.is_(None))
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
                    data={"scope": normalized_scope.to_dict(), "reason": reason},
                    created_at=datetime.now(UTC),
                )
            )
            await session.commit()
            await session.refresh(pause)
            return pause
    finally:
        await engine.dispose()


async def list_pauses(ctx: RepoContext, *, scope: str | PauseScope) -> list[Pause]:
    _ensure_initialized(ctx)

    normalized_scope = _normalize_scope(scope)
    engine = create_engine(ctx.db_path)
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            rows = await session.scalars(
                select(Pause).where(Pause.scope == normalized_scope).order_by(desc(Pause.id))
            )
            return list(rows)
    finally:
        await engine.dispose()
