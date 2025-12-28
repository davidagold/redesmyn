# Messages + Commands Epic: Control Doc (Canonical)

This file is the canonical “control doc” for the **Messages + Commands** epic: intent, v0 spec, sequencing, and key decisions. Keep it current.

## Metadata

```yaml
slug: messages-commands
name: Messages + Commands
root_branch: main
linear:
  project_id: null
```

## 1) Vision

Provide a first-class, auditable communication and control loop between humans and agent sessions:

- Persisted message threads (user ⇄ agent session; later: agent ⇄ agent).
- Structured commands with explicit lifecycle (`queued → running → succeeded/failed/canceled`) and durable provenance.
- Delivery modes that work across harnesses:
  - Hook-based (when available) for tighter integration and richer UX.
  - Cooperative (skill/bootstrap + polling) when hooks aren’t available.
  - Manual “bridging” while attached as a last-resort degraded mode (still persisted + auditable).

The UI remains **graph-first**: messaging and commands should feel integrated into the node workflow rather than a detached inbox/table.

## 2) Relationship to Agent Orchestration

This epic depends on the core session/runner building blocks defined in `epics/agent-orchestration/README.md` (agent sessions, event stream plumbing, repo observer, etc.), but it is designed separately so we can:

- keep dogfooding unblocked (agent/git activity first),
- do a dedicated design session for the messaging/commands UX and data model.

## 3) v0 scope

- Message threads per node/session with durable storage and query APIs.
- Command issuance + agent acknowledgment + state transitions.
- WebSocket events for new messages + command state changes (building on the existing event stream).
- Minimal dashboard UI integration: per-node thread + command issuance from selection context.

## 4) Non-goals (v0)

- Multi-user collaboration semantics beyond a single user’s cockpit (shared tenancy).
- Perfect deep integration for every harness; hooks are opportunistic.

