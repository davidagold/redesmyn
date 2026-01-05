from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import desc, select

from redesmyn.context import RepoContext
from redesmyn.db import (
    Block,
    BlockAck,
    BlockScope,
    Event,
    create_engine,
    create_sessionmaker,
)
from redesmyn.db.models import AckRelease, ManualRelease
from redesmyn.domain.enums import BlockMode, BlockPolicy


class NotInitializedError(RuntimeError):
    pass


def _ensure_initialized(ctx: RepoContext) -> None:
    if not ctx.db_path.exists():
        raise NotInitializedError(
            "Redesmyn is not initialized in this repo. Run `rn init`."
        )


def _normalize_scope(scope: str | BlockScope) -> BlockScope:
    if isinstance(scope, BlockScope):
        return scope
    if scope == "repo":
        return BlockScope.for_repo()
    if scope.startswith("branch:"):
        return BlockScope.for_branch(scope.removeprefix("branch:"))
    raise ValueError(f"Unknown block scope: {scope}")


def _normalize_policy(policy: str | BlockPolicy) -> BlockPolicy:
    if isinstance(policy, BlockPolicy):
        return policy
    return BlockPolicy(policy)


def _normalize_mode(mode: str | BlockMode) -> BlockMode:
    if isinstance(mode, BlockMode):
        return mode
    return BlockMode(mode)


async def get_effective_block(
    ctx: RepoContext, *, branch: str | None, policy: str | BlockPolicy
) -> Block | None:
    normalized_policy = _normalize_policy(policy)
    if branch:
        block = await get_active_block(
            ctx, scope=BlockScope.for_branch(branch), policy=normalized_policy
        )
        if block is not None:
            return block
    return await get_active_block(
        ctx, scope=BlockScope.for_repo(), policy=normalized_policy
    )


async def get_active_block(
    ctx: RepoContext, *, scope: str | BlockScope, policy: str | BlockPolicy
) -> Block | None:
    _ensure_initialized(ctx)
    normalized_scope = _normalize_scope(scope)
    normalized_policy = _normalize_policy(policy)

    engine = create_engine(ctx.db_path)
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            stmt = (
                select(Block)
                .where(
                    Block.scope == normalized_scope,
                    Block.policy == normalized_policy,
                    Block.cleared_at.is_(None),
                )
                .order_by(desc(Block.id))
                .limit(1)
            )
            return await session.scalar(stmt)
    finally:
        await engine.dispose()


async def set_manual_block(
    ctx: RepoContext,
    *,
    scope: str | BlockScope,
    policy: str | BlockPolicy,
    mode: str | BlockMode,
    reason: str | None,
) -> Block:
    _ensure_initialized(ctx)

    normalized_scope = _normalize_scope(scope)
    normalized_policy = _normalize_policy(policy)
    normalized_mode = _normalize_mode(mode)
    release = ManualRelease().model_dump(mode="python")

    engine = create_engine(ctx.db_path)
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            active = await session.scalar(
                select(Block)
                .where(
                    Block.scope == normalized_scope,
                    Block.policy == normalized_policy,
                    Block.cleared_at.is_(None),
                )
                .order_by(desc(Block.id))
                .limit(1)
            )
            if active is not None:
                active.cleared_at = datetime.now(UTC)
                active.cleared_reason = "superseded"

            block = Block(
                scope=normalized_scope,
                policy=normalized_policy,
                mode=normalized_mode,
                release=release,
                reason=reason,
                created_at=datetime.now(UTC),
            )
            session.add(block)
            session.add(
                Event(
                    event_type="block.set",
                    data={
                        "scope": normalized_scope.to_dict(),
                        "policy": normalized_policy.value,
                        "mode": normalized_mode.value,
                        "release": release,
                        "reason": reason,
                    },
                    created_at=datetime.now(UTC),
                )
            )
            await session.commit()
            await session.refresh(block)
            return block
    finally:
        await engine.dispose()


async def clear_block(
    ctx: RepoContext,
    *,
    scope: str | BlockScope,
    policy: str | BlockPolicy,
    reason: str | None,
) -> Block | None:
    _ensure_initialized(ctx)

    normalized_scope = _normalize_scope(scope)
    normalized_policy = _normalize_policy(policy)

    engine = create_engine(ctx.db_path)
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            block = await session.scalar(
                select(Block)
                .where(
                    Block.scope == normalized_scope,
                    Block.policy == normalized_policy,
                    Block.cleared_at.is_(None),
                )
                .order_by(desc(Block.id))
                .limit(1)
            )
            if block is None:
                return None

            block.cleared_at = datetime.now(UTC)
            block.cleared_reason = reason
            session.add(
                Event(
                    event_type="block.cleared",
                    data={
                        "scope": normalized_scope.to_dict(),
                        "policy": normalized_policy.value,
                        "reason": reason,
                    },
                    created_at=datetime.now(UTC),
                )
            )
            await session.commit()
            await session.refresh(block)
            return block
    finally:
        await engine.dispose()


async def list_blocks(
    ctx: RepoContext, *, scope: str | BlockScope, policy: str | BlockPolicy
) -> list[Block]:
    _ensure_initialized(ctx)

    normalized_scope = _normalize_scope(scope)
    normalized_policy = _normalize_policy(policy)

    engine = create_engine(ctx.db_path)
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            rows = await session.scalars(
                select(Block)
                .where(
                    Block.scope == normalized_scope, Block.policy == normalized_policy
                )
                .order_by(desc(Block.id))
            )
            return list(rows)
    finally:
        await engine.dispose()


async def ack_block(ctx: RepoContext, *, block_id: int, task_id: int) -> Block | None:
    _ensure_initialized(ctx)

    engine = create_engine(ctx.db_path)
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            block = await session.get(Block, block_id)
            if block is None or block.cleared_at is not None:
                return None

            session.add(
                BlockAck(
                    block_id=block_id,
                    task_id=task_id,
                    acked_at=datetime.now(UTC),
                )
            )
            session.add(
                Event(
                    event_type="block.ack",
                    data={
                        "block_id": block_id,
                        "task_id": task_id,
                        "scope": block.scope.to_dict(),
                        "policy": block.policy.value,
                    },
                    created_at=datetime.now(UTC),
                )
            )
            await session.commit()

            if block.release.get("type") != "acks":
                return block

            release = AckRelease.model_validate(block.release)
            acked = await session.scalars(
                select(BlockAck.task_id).where(BlockAck.block_id == block_id)
            )
            acked_ids = set(acked)
            if set(release.required_task_ids).issubset(acked_ids):
                block.cleared_at = datetime.now(UTC)
                block.cleared_reason = "acks_satisfied"
                session.add(
                    Event(
                        event_type="block.cleared",
                        data={"block_id": block_id, "reason": "acks_satisfied"},
                        created_at=datetime.now(UTC),
                    )
                )
                await session.commit()
                await session.refresh(block)

            return block
    finally:
        await engine.dispose()
