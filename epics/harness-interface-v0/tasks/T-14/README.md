# T-14 Structured tmux log capture race (missing `thread.started` / early events)

## Metadata

```yaml
id: T-14
epic: harness-interface-v0
stacked_on: T-7
must_land_after:
- T-10
branch:
  suggested: rn/harness-interface-v0/T-14-structured-log-capture-race
```

## Problem

For structured agents (e.g. `codex exec --json ...`) we tail the tmux pane using `tmux pipe-pane` and persist the
structured JSON stream to the per-session `output.log`.

Intermittently, the persisted log begins **mid-stream** (e.g. first entry is `item_1` / `item_2`) and is missing the
initial structured events:

- `thread.started` (Codex thread id)
- `turn.started`
- early item events

This breaks downstream functionality that depends on those early events being present in the captured log:

- discovering / persisting `external_session_ref` (resumable session indicator)
- structured “resume-by-id” automation workflows (e.g. merge conflict assist)
- debugging/observability (it’s not clear what actually happened)

### Likely root cause

`tmux pipe-pane` only captures output from the moment the pipe is attached onward. If the pipe is attached late
or temporarily detached (race / tmux-server hiccup / reattach), early output is lost and cannot be recovered from
`output.log`.

We have reproduced cases where:

- Codex itself still emits `thread.started` normally, but
- `output.log` is missing it and begins at later items, implying the pipe attached after Codex started writing.

## Goal

Guarantee that structured session logs reliably include the earliest structured events (especially `thread.started`),
or provide an equally reliable source of truth that does not depend on tmux pipe timing.

## Proposed approaches (pick one)

1) **Structured-mode direct capture (preferred)**
   - For structured exec turns, write the harness stdout/stderr directly to `output.log` from the launcher (`run.sh`),
     and avoid relying on `tmux pipe-pane` for log capture.
   - Keep tmux only as the lifecycle container / attach target if needed, but make logging independent of tmux.

2) **Pipe handshake**
   - Ensure the harness process does not begin writing until `pipe-pane` is confirmed attached.
   - This might require a handshake file/socket or a wrapper that blocks until the pipe is in place.

3) **Augment capture with a secondary source**
   - If the first structured events are missing from `output.log`, recover from the harness’s own session artifacts
     (e.g. `CODEX_HOME/sessions/**/rollout-*.jsonl`) and persist a synthetic “missing” event.
   - This is a mitigation and should not be the only mechanism.

## Acceptance criteria

- A structured `codex exec --json ...` session’s `output.log` always begins with `thread.started` and `turn.started`
  (or equivalent deterministic session/turn identifiers).
- `external_session_ref` is reliably persisted for structured Codex sessions without requiring heuristics.
- We have enough logging/events to debug “did we start a turn?” and “what are we waiting on?” scenarios.

