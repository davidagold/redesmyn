from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from redesmyn.agent_interface.v0 import AgentBackend, ShellAgent
from redesmyn.domain.enums import AgentKind

if TYPE_CHECKING:
    from redesmyn.db.models import AgentSession

AgentBackendBuilder = Callable[["AgentSession"], AgentBackend]


def _shell_backend(_agent_session: "AgentSession") -> AgentBackend:
    return ShellAgent()


# Registry seam: T-2 owns selection; T-3/T-4 provide concrete backends.
#
# Until agent-specific implementations land, non-Generic kinds safely fall back
# to ShellAgent (no agent-specific semantics).
AGENT_BACKEND_BUILDERS: dict[AgentKind, AgentBackendBuilder] = {
    AgentKind.Generic: _shell_backend,
    AgentKind.Codex: _shell_backend,
    AgentKind.ClaudeCode: _shell_backend,
}


def register_agent_backend_builder(
    kind: AgentKind, builder: AgentBackendBuilder
) -> None:
    AGENT_BACKEND_BUILDERS[kind] = builder
