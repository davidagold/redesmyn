# T-6 Remove tooltip that duplicates error details already shown in the callout

## Metadata

```yaml
id: T-6
epic: ui-v0
stacked_on: T-5
branch:
  suggested: rn/ui-v0/T-6-error-tooltip-removal
```

## Problem

The task card error UI currently shows a tooltip containing the error message, even though the same error message is already
available in the expanded callout panel. This creates extra UI chrome without adding information.

Reference screenshot: `codex-clipboard-Qxc2Jt.png`.

## Goal

Remove the tooltip so error details are accessed via the callout expansion only.

## Requirements

- Remove the tooltip wrapper for the error summary/message.
- Keep the callout expansion as the way to view the full error details (including long paths).
- Maintain accessibility: the error summary should remain readable and the expand control should remain discoverable.

## Acceptance Criteria

- Hovering the error summary does not show a tooltip with duplicated error text.
- Expanding the callout continues to reveal the error details.
- `just check` remains green.

