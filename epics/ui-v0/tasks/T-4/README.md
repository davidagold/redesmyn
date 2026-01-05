# T-4 Remove redundant Linear connection button from Details panel

## Metadata

```yaml
id: T-4
epic: ui-v0
stacked_on: T-3
branch:
  suggested: rn/ui-v0/T-4-details-linear-button-removal
```

## Problem

The Details panel currently includes a Linear connection/status button that duplicates the global Linear connection control.
This is redundant and makes the Details panel feel busier than it needs to be.

Reference screenshot: `codex-clipboard-QCxxyW.png`.

## Goal

Remove the Linear connection/status control from the Details panel so:

- Linear auth/connect is managed in one obvious global location.
- The Details panel stays focused on task-specific, actionable content.

## Requirements

- Remove the Details panel Linear connection/status button (and any spacing/layout it creates).
- Do not remove the global Linear connection control.
- Ensure Linear sync actions remain discoverable via the intended global UI surface(s).

## Acceptance Criteria

- The Details panel no longer renders a Linear connection/status button.
- Users can still connect/disconnect Linear via the global Linear control.
- `just check` remains green.

