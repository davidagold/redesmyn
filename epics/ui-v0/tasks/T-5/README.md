# T-5 Remove redundant inline red error text from task cards

## Metadata

```yaml
id: T-5
epic: ui-v0
stacked_on: T-4
branch:
  suggested: rn/ui-v0/T-5-task-card-error-dedupe
```

## Problem

When a task hits an error (e.g. restack failure), the task card shows:

- a red inline error message under the title, and
- an error callout panel that repeats the same message.

This is visually noisy and redundant.

Reference screenshot: `codex-clipboard-1h0JPp.png`.

## Goal

Display the error in a single place on the task card (the callout), keeping the card content concise.

## Requirements

- Remove the redundant inline red error message from the task card.
- Keep the error callout as the single source of truth for error display, including the expandable details.
- Preserve the ability to quickly see “something is wrong” at a glance (the callout summary should remain obvious).

## Acceptance Criteria

- The task card does not render a separate red inline error line when the error callout is present.
- The callout continues to show a compact summary and expandable details.
- `just check` remains green.

