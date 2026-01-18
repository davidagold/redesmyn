---
epic: gpui
branch:
  suggested: rn/gpui/T-32-agent-kind-and-resume-by-id
rn:
  parent: T-2
---

# T-32 Agent kind taxonomy + interface mode inference + resume-by-id turn builder (Domain 4)

## Problem

Agent/session work will be developed in parallel across:

- daemon exec runtime,
- control plane “send message” logic,
- CLI ergonomics (`rn`),
- and the native session viewer.

If we do not centralize:

- **agent kind** taxonomy,
- **interface mode** detection (structured vs interactive),
- and **resume-by-id** argv construction,

we will produce subtly incompatible behavior across clients/daemons and risk regressing the most important “send message” flows.

We also want to fix naming: “Generic” does not communicate capabilities/limitations.

## Goal

Introduce a single, canonical Rust implementation of:

1. Agent kind taxonomy (including a better name than “Generic”).
2. Interface mode inference from the harness argv.
3. Resume-by-id turn argv builder for structured agents.

This is a **pure utilities** ticket intended to unblock massively parallel work in Domain 4.

## Requirements

### 1) Naming: `Shell` (instead of `Generic`)

In the Rust port, name the “unstructured / tmux-first” agent kind **`Shell`**:

- communicates capabilities: interactive shell/TUI
- communicates limitation: no reliable structured semantics without agent-specific signals

We keep “Generic” only as a legacy synonym where needed for backwards compatibility during the split-codebase period.

### 2) Agent kind inference (argv)

Implement `infer_agent_kind_from_argv(argv: &[String]) -> AgentKind` with the same best-effort behavior as today:

- If argv is `codex …`, infer `Codex`.
- If argv is `claude …` or `claude-code …`, infer `ClaudeCode`.
- If argv is a wrapper (`uv run`, `npx`, `npm exec`, etc.), scan the post-wrapper tokens.
- Otherwise infer `Shell`.

This logic must be deterministic and well-tested (no filesystem probing).

### 3) Interface mode inference (argv)

Implement `infer_interface_mode_from_argv(argv: &[String], resolved_kind: AgentKind) -> AgentInterfaceMode`:

- Codex is **structured** iff the *agent invocation* contains `codex exec --json …`.
- Claude Code is **structured** iff the *agent invocation* contains:
  - `--print` (or `-p`), and
  - `--output-format stream-json` (or equivalent).
- Everything else is **interactive**.

The function must operate on the agent invocation argv (i.e., handle wrapper tools like `uv run`).

### 4) Resume-by-id turn builder (structured agents)

Implement `build_resume_by_id_turn(...)` for structured agents:

- Codex:
  - requires `codex exec --json …`
  - produces `... exec --json <options...> resume <thread_id> -`
  - preserves exec options (e.g. `-C`, `--color`, etc.)
- Claude Code:
  - requires `--print` + `--output-format stream-json`
  - inserts `--resume <session_id>` immediately after `--print`/`-p`
  - drops `--continue`
  - replaces any existing `--resume <old>`

Return value includes:

- argv for the resumed turn
- stdin prompt (ensures newline semantics)

### 5) Tests (port behavior precisely)

Add unit tests equivalent to:

- `tests/test_agent_turn_transport.py`

## Acceptance criteria

- The Rust workspace has a canonical agent-kind/mode inference module used by both daemon and control plane code.
- Resume-by-id argv generation matches the Python semantics (tests pass).
- Naming is clarified (`Shell` replaces “Generic” in Rust-facing user and developer surfaces).

## Dependencies / sequencing

- Depends on ID/newtype foundations (T-2) for stable identifiers referenced by later session runtime code.

## Reference implementation (today; for behavior orientation only)

- Agent kind inference (Python today):
  - `redesmyn/agent_kind.py` (`infer_agent_kind_from_argv`, wrapper scanning).
- Interface mode inference (Python today):
  - `redesmyn/agent_runtime.py` (`infer_interface_mode_from_argv`).
- Resume-by-id argv builder (Python today):
  - `redesmyn/agent_turn_transport.py` (`build_resume_by_id_turn`).
- Tests (Python today):
  - `tests/test_agent_turn_transport.py`

