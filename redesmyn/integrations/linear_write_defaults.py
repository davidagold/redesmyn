from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from redesmyn.db import LinearEpicDefaults
from redesmyn.integrations.linear import (
    LinearClient,
    resolve_default_team,
    resolve_or_create_label,
)


@dataclass(frozen=True, slots=True)
class LinearWriteDefaults:
    team_id: str
    label_id: str


async def load_linear_write_defaults(
    session: AsyncSession, *, epic_id: int
) -> LinearWriteDefaults | None:
    row = await session.get(LinearEpicDefaults, epic_id)
    if row is None or row.team_id is None or row.label_id is None:
        return None
    return LinearWriteDefaults(team_id=row.team_id, label_id=row.label_id)


async def ensure_linear_write_defaults(
    session: AsyncSession,
    *,
    epic_id: int,
    epic_slug: str,
    project_id: str,
    client: LinearClient,
) -> LinearWriteDefaults:
    row = await session.get(LinearEpicDefaults, epic_id)
    preferred_team_id = row.team_id if row is not None else None

    try:
        team = await resolve_default_team(
            client, project_id=project_id, preferred_team_id=preferred_team_id
        )
    except ValueError:
        team = await resolve_default_team(client, project_id=project_id)

    label_id: str
    if row is not None and row.label_id is not None:
        label_id = row.label_id
    else:
        label_id = (
            await resolve_or_create_label(client, label_name=epic_slug, team_id=team.id)
        ).id

    if row is None:
        row = LinearEpicDefaults(epic_id=epic_id, team_id=team.id, label_id=label_id)
        session.add(row)
    else:
        row.team_id = team.id
        row.label_id = label_id

    await session.flush()
    return LinearWriteDefaults(team_id=team.id, label_id=label_id)
