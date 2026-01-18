---
epic: gpui
branch:
  suggested: rn/gpui/T-44-gpui-ui-foundations
rn:
  parent: T-43
---

# T-44 GPUI UI foundations (theme, tokens, gpui-component survey, shared widgets) (Domain 5)

## Problem

We want a large, parallel GPUI port that stays maintainable.

Without a shared UI foundation, we will:

- duplicate styling/theme decisions,
- invent inconsistent interaction patterns,
- and create painful merge conflicts in a single “god UI module”.

We also have non-negotiable UX constraints:

- no silent actions,
- no spinner wheels,
- accessible progress affordances,
- and a calm, legible UI.

## Goal

Establish the UI “design system” layer for the GPUI app:

- theme + tokens,
- shared widgets and patterns,
- and an explicit inventory of what GPUI core and `gpui-component` give us vs what we must build.

This ticket is meant to unblock parallel work across graph/session/diff UIs by creating a stable base.

## Requirements

### 1) Theme model

Implement theme preference:

- `dark` / `light` / `system`
- persisted locally (app settings; exact storage mechanism TBD but must be stable)

Theme toggling must be accessible and must not require a persistent navigation sidebar.

### 2) Tokens and typography

Define a small set of tokens used across the app:

- spacing scale,
- radii,
- border/outline policy (avoid “busy border soup”),
- text styles (body, caption, mono for code-ish surfaces),
- and semantic colors (surface, foreground, muted, danger, warning, etc.).

### 3) Shared widgets (minimal set)

Provide a small shared widget set (names illustrative):

- `Button` (with disabled reason support)
- `Tooltip`
- `IconButton`
- `TextInput` / `TextArea` (with submit affordances)
- `Toast`/`Callout` for actionable errors (no silent failures)
- `ProgressPill` / inline “…” indicator (no spinners)

### 4) gpui-component survey

Evaluate `gpui-component` for:

- split panes,
- lists/tables,
- popovers/tooltips,
- text input primitives,
- command palette primitives (if any).

Write down:

- what we will reuse as-is,
- what we will wrap (to fit our conventions),
- and what we must implement ourselves.

### 5) “No silent actions” primitives

Implement a shared pattern for UI-triggered actions:

- every action produces an immediately visible in-flight indicator,
- triggering controls are disabled while in flight (unless concurrency is explicitly safe),
- errors are rendered in place with actionable text, preserving user input when possible.

## Acceptance criteria

- The desktop app has a stable theme/tokens layer.
- At least one view uses the shared widgets (so we know they work end-to-end).
- The gpui-component survey is written down in this ticket and/or a short doc for implementers.

## Dependencies / sequencing

- Depends on GPUI app bootstrap (T-43).

## Reference implementation (today; for behavior orientation only)

- Theme + layout (web today):
  - `dashboard/src/hooks/useTheme.ts` (theme preference).
  - `dashboard/src/index.css` (color tokens).
- Widget conventions (web today):
  - `dashboard/src/components/ui/` (button/tooltip/panels; includes “disabledReason” UX).

