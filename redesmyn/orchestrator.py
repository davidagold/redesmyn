from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select

from redesmyn.context import RepoContext
from redesmyn.db import Repository, create_engine, create_sessionmaker, init_db
from redesmyn.git_telemetry import update_git_projections
from redesmyn.repo import default_branch


async def init_repo(ctx: RepoContext, *, migrate: bool) -> None:
    ctx.state_dir.mkdir(parents=True, exist_ok=True)

    engine = create_engine(ctx.db_path)
    try:
        await init_db(engine, migrate=migrate)
        sessionmaker = create_sessionmaker(engine)

        async with sessionmaker() as session:
            repo = await session.scalar(
                select(Repository).where(Repository.repo_root == str(ctx.repo_root))
            )
            if repo is None:
                repo = Repository(
                    repo_root=str(ctx.repo_root),
                    default_branch=default_branch(ctx.repo_root),
                    created_at=datetime.now(UTC),
                )
                session.add(repo)
                await session.commit()
    finally:
        await engine.dispose()

    await update_git_projections(ctx)
