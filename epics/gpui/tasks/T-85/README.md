---
epic: gpui
branch:
  suggested: rn/gpui/T-85-multi-agent-runtime-routing
rn:
  node:
    branch: rn/gpui/T-85-multi-agent-runtime-routing
  parent: T-84
---

# T-85 Multi-agent runtime routing (capability-driven dispatch + driver registry) (Domain 3/4)

## Problem

After `T-84`, the UI/control plane can express agent choice, but runtime execution is still
Codex-only in daemon routing/driver wiring. This creates a mismatch between selectable intent and
actual runnable backends.

Current issues:

- Daemon command router rejects non-Codex start/list-model commands.
- Driver wiring is single-driver (`CodexDriver`) rather than kind-routed.
- Daemon capability handshake does not expose agent-kind support as a typed dispatch constraint.
- Control-plane daemon dispatch does not select a daemon based on requested agent-kind support.

## Goal

Make agent execution capability-driven end-to-end: requested `AgentKind` routes only to compatible
daemon runtime support, and unsupported combinations fail early with precise diagnostics.

## Scope

1. **Typed capability expansion**
   - Extend daemon capability representation to include supported agent kinds.
   - Preserve backward compatibility for older capability payloads where feasible.

2. **Control-plane routing by kind**
   - Track daemon capability data in router state.
   - Dispatch commands only to daemon instances that support requested `AgentKind`.
   - Return `unavailable` when no compatible daemon exists for the repo/scope.

3. **Daemon command-router refactor**
   - Replace Codex-only checks with a driver-registry lookup by `AgentKind`.
   - Route start/resume/list-model/set-model/permission actions through the selected driver.
   - Keep explicit failure path when no driver is registered for a requested kind.

4. **Model catalog parity by kind**
   - Ensure `ListAgentModels(agent_kind=...)` is routed through the same capability/driver logic.
   - Keep model-selection responses coherent with active driver support.

5. **Observability + diagnostics**
   - Add structured logs for capability negotiation and kind-based dispatch decisions.
   - Keep user-facing errors concise and actionable.

## Non-goals

- Building full Claude/Shell runtime feature parity in one step if underlying runners are not
  ready.
- UI redesign of session/composer controls (covered by earlier tasks).

## Dependencies / sequencing

- Parent: `T-84` (agent-kind selection plumbing and UX).
- Depends on existing runtime foundations (`T-23`, `T-35`, `T-39`, `T-68`) and agent taxonomy
  (`T-32`).

## Acceptance criteria

- Daemon/control-plane handshake communicates supported agent kinds.
- Control-plane dispatch selects only daemon(s) compatible with requested `AgentKind`.
- Daemon router no longer hardcodes Codex-only logic in generic start/list-model paths.
- Unsupported combinations fail with explicit `unavailable`/`invalid_request` errors (as
  appropriate), with no false-success session starts.
- End-to-end tests cover:
  - compatible dispatch,
  - incompatible dispatch,
  - no-compatible-daemon behavior,
  - model-list routing by kind.

## Validation

- Unit tests for capability parsing/routing decisions.
- Control-plane integration tests for kind-based dispatch and failure semantics.
- Daemon integration tests for driver registry selection and command handling.
- Manual smoke with at least one compatible and one unsupported kind selection path.
