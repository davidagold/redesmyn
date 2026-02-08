---
epic: gpui
branch:
  suggested: rn/gpui/T-84-agent-selection-at-session-start
rn:
  node:
    branch: rn/gpui/T-84-agent-selection-at-session-start
  parent: null
---

# T-84 Agent selection at session start (remove hardcoded `agent_kind`) (Domain 4/7)

## Problem

The GPUI port still hardcodes `agent_kind: Codex` in multiple start/restart paths, so the UI
cannot accurately represent or execute an explicit agent choice at session start.

Current issues:

- Start/restart actions from task cards and session UI do not take an explicit selected
  `AgentKind`.
- The UI can present settings and model controls, but agent-kind choice is not a first-class,
  durable presentation state for “start session” actions.
- Control plane validation for daemon-supported agent kinds is static, not capability-driven.

## Goal

Make agent kind a real parameter through start/restart flows, with explicit UI state and
capability-aware validation, while preserving current Codex-only execution behavior for unsupported
kinds.

## Scope

1. **UI state + action plumbing**
   - Add a clear source of truth for selected `AgentKind` in session/task presentation state.
   - Plumb selected kind through:
     - task-card `Start agent`,
     - task quick actions (`Start`/`Restart`),
     - session start/restart request helpers.
   - Remove hardcoded `AgentKind::Codex` request construction in these paths.

2. **Capability-gated UX**
   - Resolve daemon-supported kinds from live capability state where available.
   - Disable or hide unsupported kinds in start surfaces.
   - If capability state is unresolved, show explicit loading/unknown state instead of silently
     defaulting to Codex.

3. **Control-plane request validation tightening**
   - Ensure `StartAgent`/`RestartAgent` return actionable errors when requested kind is unsupported
     in current runtime conditions.
   - Keep error category/message consistent with other unavailable capability failures.

4. **Compatibility guardrails**
   - Preserve existing behavior for Codex paths.
   - Keep API/wire compatibility for current clients.

## Non-goals

- Implementing new runtime drivers for non-Codex agents.
- Reworking model-selection semantics beyond the agent-kind selection plumbing.
- Changing durable session-event schema beyond what is strictly needed for phase 1.

## Dependencies / sequencing

- Parent: `T-83` (shared cascading/menu primitives and polished session controls).
- Uses existing control-plane agent orchestration surfaces from `T-41`.
- Prepares phase 2 (`T-85`) where daemon/runtime support expands beyond Codex.

## Acceptance criteria

- No start/restart request path in GPUI UI hardcodes `AgentKind::Codex`.
- User-selected `AgentKind` is the value sent in `StartAgent`/`RestartAgent` requests.
- Unsupported kinds are clearly represented as unavailable (not silently coerced).
- Start/restart failure messaging is actionable and visible in task/session UI.
- Existing Codex flow remains intact and regression-tested.

## Validation

- UI-level tests for start/restart request payload kind selection.
- Control-plane tests for unsupported-kind responses.
- Manual smoke in desktop app:
  - select kind,
  - start/restart,
  - confirm command/session behavior and status messaging.
