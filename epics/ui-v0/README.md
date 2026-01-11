# UI v0 Epic: Control Doc (Canonical)

This file is the canonical “control doc” for the **UI v0** epic: intent, scope, v0 readiness criteria, and task map. Keep it current.

## Metadata

```yaml
slug: ui-v0
name: UI v0
root_branch: main
linear:
  project_id: null
```

## 1) Vision

Make the dashboard UI “v0 ready” for daily dogfooding:

- Core workflows are discoverable and hard to misuse.
- The UI reliably reflects backend state (agents, merge/restack runs, conflicts, daemon presence).
- Errors are actionable and do not “fail silent” (or dump noisy internals).
- The graph remains the primary control surface; supporting panels are concise and purposeful.

## 2) v0 readiness criteria (UI)

This is intentionally practical rather than exhaustive. The UI is “v0 ready” when:

- **Merge/restack flows are predictable**
  - “Ready to merge” semantics are clear and aligned with what merge actually requires.
  - Merge/restack plans are visible (at least at a summary level) before execution.
  - Conflicts/blocking states are surfaced at the correct task(s), with a clear next step.
- **Agent lifecycle is reliable and legible**
  - Start/restart/stop are correct and do not desync the UI.
  - Status indicators and tooltips are concise and consistent.
  - Attach/log actions are easy to discover and work from both card + details.
- **Daemon/executor state is easy to interpret**
  - Connected/offline state is visible, non-noisy, and uses display names.
  - The UI communicates “where” execution is happening when multiple hosts exist.
- **Graph interactions feel stable**
  - Selection, focus, and viewport behavior are smooth and deterministic.
  - Layout does not jitter during routine updates.

## 3) Task map

- `epics/ui-v0/tasks/T-1/README.md`: “Ready to merge” convenience behavior (auto-ready unmerged ancestors on the spine).
- `epics/ui-v0/tasks/T-2/README.md`: Modernize query management (TanStack Query, event-driven invalidation, responsive UI updates).
- `epics/ui-v0/tasks/T-3/README.md`: Frontend refactor for v0 (decompose large components, share patterns, reduce duplication).
- `epics/ui-v0/tasks/T-4/README.md`: Remove redundant Linear connection button from the Details panel.
- `epics/ui-v0/tasks/T-5/README.md`: Remove redundant inline error text from task cards (dedupe with callout).
- `epics/ui-v0/tasks/T-6/README.md`: Remove tooltip that duplicates error details already shown in the callout.
- `epics/ui-v0/tasks/T-7/README.md`: Repo executor panel polish (copy buttons, terminology, telemetry/projections messaging).
- `epics/ui-v0/tasks/T-8/README.md`: Simplify graph interactions (remove on-select viewport movement and diagonalization behavior).
- `epics/ui-v0/tasks/T-9/README.md`: Left-align commit titles (keep trunk spacing invariant).
- `epics/ui-v0/tasks/T-10/README.md`: Make epic menu hover background full width.
- `epics/ui-v0/tasks/T-11/README.md`: Animate task card expansion (reflow graph layout for variable node heights).
- `epics/ui-v0/tasks/T-12/README.md`: Show tasteful in-flight progress indicators for all user actions (no silent requests).
- `epics/ui-v0/tasks/T-13/README.md`: Revise git action status indicators to reflect run scope and the true blocker (avoid misattribution to the initiating task).
