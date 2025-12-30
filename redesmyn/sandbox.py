from __future__ import annotations

import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Union

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Annotated

SandboxNetworkMode = Literal["allow", "deny"]


class NullSandboxPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["none"] = "none"


class WorktreeSandboxPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["worktree"] = "worktree"
    worktree_path: Path
    shared_state_paths: list[Path]
    network: SandboxNetworkMode = "allow"


SandboxPolicy = Annotated[
    Union[NullSandboxPolicy, WorktreeSandboxPolicy],
    Field(discriminator="type"),
]


@dataclass(frozen=True, slots=True)
class SandboxWrapResult:
    argv: list[str]
    env: dict[str, str]


class SandboxCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    available: bool
    supports_worktree: bool
    supports_network_deny: bool
    unavailable_reason: str | None = None


class SandboxProvider(Protocol):
    name: str

    def capabilities(self) -> SandboxCapabilities: ...

    def wrap(
        self,
        *,
        argv: list[str],
        cwd: Path,
        env: dict[str, str],
        policy: SandboxPolicy,
    ) -> SandboxWrapResult: ...


def _subpath_clause(path: Path) -> str:
    return f'(subpath "{path}")'


def _sandbox_exec_profile(policy: WorktreeSandboxPolicy) -> str:
    clauses = [
        "(version 1)",
        # Start permissive, then clamp writes. This keeps v0 sandboxing simple and
        # avoids having to enumerate every readable system path.
        "(allow default)",
        "(deny file-write*)",
        # PTYs are required for interactive tools (e.g. Codex exec). On macOS,
        # `openpty` touches `/dev/*` devices which counts as a file write under
        # sandbox-exec. Allowing `/dev` keeps PTYs functional without widening
        # normal filesystem writes.
        f"(allow file-write* {_subpath_clause(Path('/dev'))})",
        f"(allow file-write* {_subpath_clause(Path('/private/dev'))})",
        *[
            f"(allow file-write* {_subpath_clause(allowed)})"
            for allowed in [policy.worktree_path, *policy.shared_state_paths]
        ],
    ]
    if policy.network == "deny":
        clauses.append("(deny network*)")
    return "\n".join(clauses) + "\n"


@dataclass(frozen=True, slots=True)
class NoSandboxProvider:
    name: str = "none"

    def capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities(
            provider=self.name,
            available=True,
            supports_worktree=False,
            supports_network_deny=False,
            unavailable_reason=None,
        )

    def wrap(
        self,
        *,
        argv: list[str],
        cwd: Path,
        env: dict[str, str],
        policy: SandboxPolicy,
    ) -> SandboxWrapResult:
        if policy.type != "none":
            raise RuntimeError("Sandbox provider is unavailable on this platform.")
        return SandboxWrapResult(argv=list(argv), env=dict(env))


@dataclass(frozen=True, slots=True)
class SandboxExecProvider:
    name: str = "sandbox-exec"

    def capabilities(self) -> SandboxCapabilities:
        exe = shutil.which("sandbox-exec")
        return SandboxCapabilities(
            provider=self.name,
            available=exe is not None,
            supports_worktree=True,
            supports_network_deny=True,
            unavailable_reason=None if exe else "`sandbox-exec` not found on PATH",
        )

    def wrap(
        self,
        *,
        argv: list[str],
        cwd: Path,
        env: dict[str, str],
        policy: SandboxPolicy,
    ) -> SandboxWrapResult:
        if policy.type == "none":
            return SandboxWrapResult(argv=list(argv), env=dict(env))

        if not isinstance(policy, WorktreeSandboxPolicy):
            raise RuntimeError(f"Unsupported sandbox policy: {policy.type}")

        capabilities = self.capabilities()
        if not capabilities.available:
            raise RuntimeError(
                capabilities.unavailable_reason or "Sandbox is unavailable"
            )

        profile = _sandbox_exec_profile(policy)
        return SandboxWrapResult(
            argv=["sandbox-exec", "-p", profile, *argv],
            env=dict(env),
        )


def make_sandbox_provider() -> SandboxProvider:
    system = platform.system()
    if system == "Darwin":
        return SandboxExecProvider()
    return NoSandboxProvider()
