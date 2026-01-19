---
epic: gpui
branch:
  suggested: rn/gpui/T-44-gpui-ui-foundations
rn:
  parent: T-43
---

# T-44 GPUI UI foundations (theme, tokens, shared widgets) (Domain 5)

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
- and an explicit inventory of what GPUI core gives us vs what we must build.

This ticket is meant to unblock parallel work across graph/session/diff UIs by creating a stable base.

## Dependency decision: Zed vs `gpui-component`

We reviewed both Zed’s internal UI kit and `gpui-component` (Longbridge) before deciding what to depend on.

### Zed (`zed` repo) UI kit

- **Quality**: Zed’s in-tree `ui` crate is the best reference implementation we’ve found for a large, high-performance GPUI app (component ergonomics, tokens, animation patterns, scrollbars, etc.).
- **Why it’s not a dependency**:
  - **License**: Zed’s `ui` crate is **GPL-3.0-or-later** (copyleft). We treat it as a reference only.
  - **Coupling**: it depends on many Zed-specific crates (theme/settings/menu/util/etc.), so it is not a drop-in “UI kit crate”.

### `gpui-component` (Longbridge)

- **Pros**: Apache-2.0 license; many ready-made widgets; includes a markdown renderer and a rope-based code editor.
- **Why it’s not our primary choice**:
  - **Too broad / heavy** for the initial port: the `gpui-component` crate pulls in markdown + HTML parsing, Tree-sitter/LSP scaffolding, charts, calendar, etc. even if we don’t need them immediately.
  - **Architectural gravity**: it brings its own theme system, global init, and widget conventions. We’d either contort our design to match it or wrap most of it anyway.
  - **Performance control**: our graph + session + diff surfaces will be highly specialized; we want tight control over allocation patterns, virtualization, and interaction semantics.

### Decision (for the port)

- **Do not depend on `gpui-component`**.
- Build a small, maintainable `redesmyn_ui` crate directly on `gpui`, **inspired by Zed’s patterns** (but implemented from scratch).
- We may selectively adopt **small, narrow** third-party crates (e.g. markdown parsing) later, but avoid importing a full UI kit framework.

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
- `SplitPane` (resizable + collapsible) building block for T-45 and future panes
- `ScrollArea` + scrollbar affordances (enough to support large chat histories)

### 4) Inventory: GPUI core vs Redesmyn UI kit

Write down what we rely on from GPUI core (with enough detail that implementers don’t need internet research), including:

- actions + key dispatch + keymap contexts,
- scroll handles + list virtualization primitives,
- animation primitives (`with_animation`, easing, repeated animations),
- focus and accessibility building blocks.

Then list what we build in `redesmyn_ui`, and what conventions we enforce (naming, disabled reason UX, progress affordances, etc.).

If there are GPUI gaps we must bridge (e.g. missing widget types, missing OS integration), record them here as follow-up tickets.

### 5) “No silent actions” primitives

Implement a shared pattern for UI-triggered actions:

- every action produces an immediately visible in-flight indicator,
- triggering controls are disabled while in flight (unless concurrency is explicitly safe),
- errors are rendered in place with actionable text, preserving user input when possible.

## Acceptance criteria

- The desktop app has a stable theme/tokens layer.
- At least one view uses the shared widgets (so we know they work end-to-end).
- The GPUI-vs-Redesmyn inventory is written down in this ticket and/or a short doc for implementers.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on GPUI app bootstrap (T-43).

## Reference implementation (today; for behavior orientation only)

- Theme + layout (web today):
  - `dashboard/src/hooks/useTheme.ts` (theme preference).
  - `dashboard/src/index.css` (color tokens).
- Widget conventions (web today):
  - `dashboard/src/components/ui/` (button/tooltip/panels; includes “disabledReason” UX).
