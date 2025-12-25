from __future__ import annotations

from redesmyn.blocks import (
    NotInitializedError,
    clear_block,
    get_active_block,
    get_effective_block,
    list_blocks,
    set_manual_block,
)
from redesmyn.context import RepoContext
from redesmyn.db import Block, BlockScope
from redesmyn.domain.enums import BlockMode, BlockPolicy

DEFAULT_POLICY = BlockPolicy.GitMutations


async def get_effective_pause(ctx: RepoContext, *, branch: str | None) -> Block | None:
    return await get_effective_block(ctx, branch=branch, policy=DEFAULT_POLICY)


async def get_active_pause(
    ctx: RepoContext, *, scope: str | BlockScope
) -> Block | None:
    return await get_active_block(ctx, scope=scope, policy=DEFAULT_POLICY)


async def set_pause(
    ctx: RepoContext,
    *,
    scope: str | BlockScope,
    mode: str | BlockMode,
    reason: str | None,
) -> Block:
    return await set_manual_block(
        ctx, scope=scope, policy=DEFAULT_POLICY, mode=mode, reason=reason
    )


async def clear_pause(
    ctx: RepoContext, *, scope: str | BlockScope, reason: str | None
) -> Block | None:
    return await clear_block(ctx, scope=scope, policy=DEFAULT_POLICY, reason=reason)


async def list_pauses(ctx: RepoContext, *, scope: str | BlockScope) -> list[Block]:
    return await list_blocks(ctx, scope=scope, policy=DEFAULT_POLICY)


__all__ = [
    "NotInitializedError",
    "DEFAULT_POLICY",
    "get_effective_pause",
    "get_active_pause",
    "set_pause",
    "clear_pause",
    "list_pauses",
]
