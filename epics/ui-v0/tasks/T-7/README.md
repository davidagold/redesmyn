# T-7 Repo executor panel: remove copy buttons, clarify messaging, and consolidate status UI

## Metadata

```yaml
id: T-7
epic: ui-v0
stacked_on: T-6
branch:
  suggested: rn/ui-v0/T-7-repo-executor-panel-polish
```

## Problem

The repo executor status UI is currently confusing and noisy:

- It includes “Copy start” / “Copy dev” buttons that are not core workflow actions.
- Wording like “No primary host” + “Attached: …” is unclear (what is “attached”, and attached to what?).
- When telemetry is unavailable, the UI implies sync projections may be unavailable, but projections should still be shown
  (they may be stale, but should remain useful).
- The “No primary repo executor” pseudo-callout duplicates the status pill (“No primary executor”) and uses confusing copy
  (“Acquire a primary executor lease”) that is not necessarily user-actionable.

Reference screenshot: `codex-clipboard-tflCuX.png`.

## Goal

Make the repo executor panel:

- concise and well-designed (deliberate layout, minimal noise)
- clear about what the system state means
- consistent in terminology (prefer “executor”)
- honest about limitations (e.g. telemetry missing) without hiding useful data

## Requirements

### 1) Remove copy action buttons

- Remove “Copy start” / “Copy dev” and any other copy-only actions from the repo executor panel.

### 2) Clarify “primary” vs “attached” semantics

- Confirm what “attached” represents (daemon connected? repo attached? lease held?).
- Update copy to explain the state in plain language using consistent terminology:
  - primary executor (lease holder) vs connected/attached executors
  - what capabilities are disabled when there is no primary executor (e.g. git mutations)

### 3) Telemetry unavailable: keep projections, label staleness

- Ensure sync projections (e.g. “out of sync with upstream”) continue to render when telemetry is unavailable.
- Update copy to indicate projections may be stale without implying they disappear.

### 4) Consolidate status messaging and improve tone

- Show “no primary executor” information once (single location, single phrasing).
- Avoid telling the user to do something they cannot reasonably interpret; prefer empathetic, actionable guidance when
  possible (or simply explain the state and where to look next).
- Styling should follow dashboard conventions: avoid excessive borders; use spacing and subtle separators.

## Acceptance Criteria

- The repo executor panel no longer includes copy-only action buttons.
- “No primary executor” is displayed once, with consistent terminology and clearer explanation of impact.
- When telemetry is unavailable, projections remain visible and are labeled as potentially stale.
- UI layout is calmer and more deliberate; `just check` remains green.

