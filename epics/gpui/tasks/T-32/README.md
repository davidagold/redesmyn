---
epic: gpui
branch:
  suggested: rn/gpui/T-32-agent-kind-taxonomy
rn:
  parent: T-2
---

# T-32 Agent taxonomy + interface inference + turn builders (structured exec + app-server) (Domain 4)

## Problem

Agent/session work will be developed in parallel across:

- daemon exec runtime,
- control plane “send message” logic,
- CLI ergonomics (`rn`),
- and the native session viewer.

If we do not centralize:

- **agent taxonomy** (provider vs runtime kind),
- **interface inference** (structured exec vs tmux vs app-server),
- and **turn construction** (resume-by-id turns, and app-server request shapes),

we will produce subtly incompatible behavior across clients/daemons and risk regressing the most important “send message” flows.

We also want to fix naming: “Generic” does not communicate capabilities/limitations.

## Goal

Introduce a single, canonical Rust implementation of:

1. Agent taxonomy (provider vs runtime kind) with clear naming.
2. Deterministic inference of provider + runtime kind from a harness argv (wrapper-aware).
3. Canonical turn builders:
   - resume-by-id **argv** builder for structured exec agents, and
   - request “shape” builders for app-server agents (no network in this ticket).

This is a **pure utilities + types** ticket intended to unblock massively parallel work in Domain 4.

## Requirements

### 1) Terminology + taxonomy (provider vs runtime kind)

We must make “what agent is this?” vs “how do we talk to it?” explicit.

Introduce two independent axes:

**Agent provider** (brand / implementation family)

- `AgentProvider::Codex`
- `AgentProvider::ClaudeCode`
- `AgentProvider::Shell` (generic executable; no structured semantics assumed)

**Agent runtime kind** (interaction contract / transport shape)

- `AgentRuntimeKind::ShellTmux` (interactive; tmux-first; unstructured)
- `AgentRuntimeKind::StructuredExec` (per-turn process spawn; structured stdout)
- `AgentRuntimeKind::AppServer` (long-lived server; request/response + event stream)

`Shell` replaces “Generic” as the default provider name in the Rust port because it communicates both capabilities and limitations.

### 2) Canonical Rust constructs (starting point; consumed across the port)

Define these as shared “building block” types in a small Rust crate/module (names illustrative; exact crate name is implementer choice):

**Invocation parsing / wrapper handling**

- `struct AgentInvocation { raw_argv: Vec<String>, wrapper: Vec<String>, agent_argv: Vec<String> }`
  - `wrapper` is the prefix (e.g., `["uv","run"]`, `["npm","exec"]`) if present.
  - `agent_argv` starts at the actual agent executable token (`codex`, `claude`, `claude-code`, or “unknown”).
  - Provide helper methods:
    - `fn agent_executable_name(&self) -> Option<String>`
    - `fn contains_subcommand(&self, name: &str) -> bool` (wrapper-aware)
    - `fn contains_flag(&self, flag: &str) -> bool`

**Resolved interface**

- `struct ResolvedAgentInterface { provider: AgentProvider, runtime: AgentRuntimeKind }`
- `enum AgentProviderSelection { Auto, Codex, ClaudeCode, Shell }` (mirrors Python selection)

**External conversation handles**

- `enum ExternalSessionRef { None, CodexThread { thread_id: String, turn_id: Option<String> }, ClaudeSession { session_id: String }, Unknown { type_: String, raw: serde_json::Value } }`
  - `None` means “not resumable” / no durable external handle is known.
  - The Rust port should preserve the “hinted provider from external_session_ref” behavior to avoid flapping across restarts.

**Turn transport building blocks**

- `struct ResumeByIdTurn { argv: Vec<String>, stdin_prompt: String }` (exec-based structured turns)
- `enum AppServerTurnIntent { StartNew { prompt: String }, Resume { external: ExternalSessionRef, prompt: String } }`
  - This ticket does **not** implement JSON-RPC/networking; it only defines the “intent” shape needed by later app-server runtime work (T-39/T-68).

Consumption rules (why these exist):

- Daemon runners (T-36..T-39, T-68) must depend on these types rather than rolling their own parsing/building.
- Control plane agent command semantics (T-41) use `ResolvedAgentInterface` to enforce capability checks (e.g. “resume-by-id is only valid when `ExternalSessionRef` is present”) without depending on provider-specific argv parsing.
- UI/CLI surfaces display:
  - resolved provider (`Codex`, `ClaudeCode`, `Shell`),
  - resolved runtime kind (`ShellTmux` / `StructuredExec` / `AppServer`),
  - and whether the current session has a resumable external handle.

### 3) Provider inference (argv)

Implement `infer_agent_provider_from_argv(argv: &[String]) -> AgentProvider` with the same best-effort behavior as today:

- If argv is `codex …`, infer `Codex`.
- If argv is `claude …` or `claude-code …`, infer `ClaudeCode`.
- If argv is a wrapper (`uv run`, `npx`, `npm exec`, `pnpm dlx`, etc.), scan the post-wrapper tokens.
- Otherwise infer `Shell`.

This logic must be deterministic and well-tested (no filesystem probing).

### 4) Runtime kind inference (argv + provider)

Implement `infer_agent_runtime_kind(invocation: &AgentInvocation, provider: AgentProvider) -> AgentRuntimeKind`:

- **Codex**:
  - If the agent invocation contains `app-server`, infer `AppServer`.
  - Else if the agent invocation contains `exec` and structured markers (see below), infer `StructuredExec`.
  - Else default to `ShellTmux` (we do not assume structured semantics).
- **Claude Code**:
  - If structured markers are present (see below), infer `StructuredExec`.
  - Else default to `ShellTmux`.
- **Shell**: always `ShellTmux`.

Structured markers (must be wrapper-aware):

- Codex structured exec iff the invocation contains `codex exec --json …`
- Claude Code structured exec iff the invocation contains:
  - `--print` (or `-p`), and
  - `--output-format stream-json` (or equivalent)

### 5) Provider selection + stability across restarts

Implement `resolve_agent_provider(selection, argv, external_session_ref_hint) -> AgentProvider`:

- explicit selection wins (`Codex`/`ClaudeCode`/`Shell`)
- `Auto` prefers provider implied by `external_session_ref` when present (avoid “flapping”)
- otherwise fall back to argv inference

### 6) Resume-by-id turn builder (structured exec agents)

Implement `build_resume_by_id_exec_turn(...)` (name illustrative) for structured exec providers:

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

### 7) App-server turn intent builder (no networking)

Define (and unit test) a small helper that maps:

- a `prompt` and
- an optional `ExternalSessionRef` (Codex thread id when known)

into an `AppServerTurnIntent` for later runtime code. This is intentionally “dumb plumbing”, but centralizing it prevents each app-server integration from inventing its own state machine.

### 8) Tests (port behavior precisely)

Add unit tests equivalent to:

- `tests/test_agent_turn_transport.py`
  - plus new tests for runtime kind inference:
    - `codex exec --json …` => `StructuredExec`
    - `codex app-server …` => `AppServer`
    - wrapper cases (`uv run`, `npm exec`, etc.)

## Acceptance criteria

- The Rust workspace has a canonical agent provider/runtime inference module used by both daemon and control plane code.
- Resume-by-id argv generation matches the Python semantics (tests pass).
- Naming is clarified (`Shell` replaces “Generic” in Rust-facing user and developer surfaces).
- App-server “turn intent” is represented explicitly (even though the protocol implementation lands later).

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

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
