from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select

from redesmyn.context import RepoContext
from redesmyn.db import Repository
from redesmyn.db.context import open_db
from redesmyn.git_projections import update_git_projections_in_session
from redesmyn.host_identity import load_or_create_host_identity


async def update_git_projections(ctx: RepoContext) -> None:
    """
    Best-effort git-derived snapshots/events produced locally (repo-executor/daemon side).

    The API/control-plane reads these from the DB and must not execute git.
    """
    ctx.state_dir.mkdir(parents=True, exist_ok=True)

    async with open_db(db_path=ctx.db_path, migrate=False) as db:
        async with db.sessionmaker() as session:
            repo = await session.scalar(
                select(Repository).where(Repository.repo_root == str(ctx.repo_root))
            )
            if repo is None:
                return

            host_key = load_or_create_host_identity(ctx).host_key
            await update_git_projections_in_session(
                ctx=ctx,
                session=session,
                repo=repo,
                host_key=host_key,
                now=datetime.now(UTC),
            )
            await session.commit()
