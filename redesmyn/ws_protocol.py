from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from redesmyn.domain.enums import CommandState


class RepoKey(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    repo_id: str


class DaemonHello(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["hello"] = "hello"
    daemon_id: str
    host: str | None = None
    capabilities: dict[str, Any] = Field(default_factory=dict)
    attached_repos: list[RepoKey] = Field(default_factory=list)


class DaemonPing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["ping"] = "ping"
    id: str | None = None


class DaemonHeartbeat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["heartbeat"] = "heartbeat"
    attached_repos: list[RepoKey] | None = None


class DaemonEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["event"] = "event"
    workspace_id: str
    repo_id: str
    event_type: str
    data: dict[str, Any] = Field(default_factory=dict)


class DaemonCommandAck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["command_ack"] = "command_ack"
    command_id: int
    state: CommandState
    data: dict[str, Any] = Field(default_factory=dict)


DaemonInboundMessage = Annotated[
    DaemonHello | DaemonPing | DaemonHeartbeat | DaemonEvent | DaemonCommandAck,
    Field(discriminator="type"),
]


class ServerCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["command"] = "command"
    command_id: int
    command_type: str
    workspace_id: str | None = None
    repo_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
