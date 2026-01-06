from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class LinearToken:
    access_token: str
    refresh_token: str | None
    token_type: str
    scope: str | None
    expires_at: datetime | None


@dataclass(frozen=True, slots=True)
class LinearWorkflowState:
    id: str
    name: str
    type: str
    position: float | None


@dataclass(frozen=True, slots=True)
class LinearIssue:
    id: str
    identifier: str
    title: str
    description: str | None
    state_type: str | None
    state_name: str | None = None
    team_id: str | None = None
    label_ids: tuple[str, ...] = ()
    label_names: tuple[str, ...] = ()
    team_states: tuple[LinearWorkflowState, ...] = ()


@dataclass(frozen=True, slots=True)
class LinearIssueRelation:
    id: str
    type: str
    issue_id: str
    related_issue_id: str


@dataclass(frozen=True, slots=True)
class LinearTeam:
    id: str
    key: str
    name: str


@dataclass(frozen=True, slots=True)
class LinearProject:
    id: str
    name: str
    slug: str | None
    teams: tuple[LinearTeam, ...] = ()


@dataclass(frozen=True, slots=True)
class LinearMilestone:
    id: str
    name: str


@dataclass(frozen=True, slots=True)
class LinearLabel:
    id: str
    name: str
