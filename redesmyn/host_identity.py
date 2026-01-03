from __future__ import annotations

import platform
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from redesmyn.context import RepoContext


class HostIdentity(BaseModel):
    host_key: str
    display_name: str


def host_identity_path(ctx: RepoContext) -> Path:
    # Back-compat: this file originally existed for the local runner, but host_key is
    # also the stable v1 daemon/executor identity.
    return ctx.state_dir / "runner-host.json"


def load_or_create_host_identity(ctx: RepoContext) -> HostIdentity:
    path = host_identity_path(ctx)
    if path.exists():
        return HostIdentity.model_validate_json(path.read_text(encoding="utf-8"))

    identity = HostIdentity(host_key=str(uuid4()), display_name=platform.node())
    path.write_text(identity.model_dump_json(indent=2), encoding="utf-8")
    return identity
