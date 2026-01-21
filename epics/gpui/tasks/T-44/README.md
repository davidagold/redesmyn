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

## What makes Zed “better” (and what we copy conceptually)

Zed is our primary GPUI reference implementation because it has already solved the same class of problems we have:

- dense, information-rich UI,
- heavy scrolling + virtualization,
- lots of keyboard-driven workflows,
- and strict performance requirements.

We **do not copy/paste Zed code** (license + coupling). We copy *principles and patterns* and implement them from scratch in `redesmyn_ui`.

### A) Layered UI crate organization (merge-friendly)

Zed’s `ui` crate is intentionally structured to avoid a single “god module”:

- `styles/` for tokens + typography + spacing + elevation,
- `traits/` for behavior composition (`Clickable`, `Disableable`, `AnimationExt`, …),
- `components/` for reusable widgets,
- `utils/` for one-off helpers.

We should mirror this structure in `redesmyn_ui` so multiple agents can work in parallel with fewer conflicts.

Reference orientation: Zed `crates/ui/src/ui.rs` and directory layout.

### B) Typed tokens + density/scaling (calm, consistent UI)

Zed avoids “random px soup” by using:

- a spacing scale derived from a small enum (`DynamicSpacing`) with density settings,
- `px(...)` / `rems(...)` helpers to keep units explicit,
- semantic colors and typography tokens instead of ad-hoc styling per component.

We should implement:

- a small, typed token model (`Spacing`, `Radius`, `Elevation`, `Typography`, `ColorRole`, …),
- UI density + scale knobs (even if we only expose defaults initially),
- and a convention that components consume *tokens*, not raw numbers.

Reference orientation: Zed `crates/ui/src/styles/spacing.rs` (dynamic spacing + density).

### C) “Traits as mixins” for ergonomics (without inheritance)

Zed gets strong composability via trait extensions:

- a base element can become clickable/disableable/animatable/etc. by importing traits,
- component APIs stay small and predictable,
- behavior is reusable without deep component hierarchies.

We should implement a small `redesmyn_ui::traits` layer for cross-cutting behaviors:

- `Disableable` (disabled reason UX),
- `Clickable` (pointer + keyboard activation),
- `AnimationExt` (house animation presets),
- `StyledExt` (common style refinements).

Reference orientation: Zed `crates/ui/src/traits/*`.

### D) Animation used deliberately (no spinner wheels, no noisy motion)

Zed leans on GPUI’s `with_animation` and easing to build subtle, low-noise affordances:

- quick in/out transitions for panels and popovers,
- text-based loading indicators (no “spinner wheel” UI),
- animations that don’t allocate per-frame.

We should adopt the same mindset:

- animations clarify state transitions and in-flight work,
- use a tiny set of approved animation durations/easings,
- avoid per-frame allocations.

Reference orientation:

- Zed `crates/ui/src/styles/animation.rs` (house presets),
- Zed `crates/ui/src/components/label/spinner_label.rs` (text-based progress).

### E) Keyboard-first: actions, contexts, and predictable precedence

Zed treats keyboard UX as first-class:

- actions are typed (`Action`), namespaced, and routed through context predicates,
- keybindings are layered and precedence is well-defined.

We should use GPUI’s action system everywhere (and avoid bespoke key handlers per view):

- define actions in a single module per feature area,
- bind keys with clear contexts,
- prefer explicit, testable command routing.

Reference orientation: GPUI `keymap.rs` and Zed `crates/zed_actions/src/lib.rs`.

### F) Scroll/virtualization is a “system”, not ad-hoc

Zed invests in scroll behavior as a core UX primitive:

- consistent scrollbars (show/hide behavior, reserved thumb space),
- scroll handles that are separated from view state,
- virtualization for large collections.

We should implement `ScrollArea` + `Scrollbar` as foundation primitives and require large surfaces (chat history, graph lists, diffs) to use them.

Reference orientation: Zed `crates/ui/src/components/scrollbar.rs`.

### G) Performance instrumentation built in (but gated)

Zed uses measurements that can be enabled via environment variables to diagnose performance without shipping noisy logs.

We should provide a similar pattern in `redesmyn_ui` and use `redesmyn_logging` spans around:

- layout/measure passes,
- expensive diff render stages,
- session list virtualization updates.

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

## Inventory (as of T-44)

See also: `epics/gpui/GPUI_ENGINEERING_NOTES.md`.

### GPUI core primitives we rely on

- **Element tree + layout**: `div()` + style chaining (flex/gap/padding/borders/rounded/shadows), plus units like `px(...)`, `relative(...)`, `Pixels`, `AbsoluteLength`.
- **Entities/state**: `cx.new(...)`, `Entity<T>`, `Context<T>::notify()`, `Context<T>::subscribe(...)`, and global state via `App::set_global(...)` + `Context<T>::observe_global::<G>(...)`.
- **Actions + key dispatch**:
  - Typed actions via `actions!(...)`.
  - Key binding via `KeyBinding` + `App::bind_keys(...)`.
  - View-local scoping via `.key_context("...")`.
  - Handling via `.on_action(...)`.
- **Focus**: `FocusHandle`, `Focusable`, `.track_focus(...)`, and `Window::focus(...)`.
- **Text input + IME plumbing**:
  - `EntityInputHandler` + `ElementInputHandler` wired via `Window::handle_input(...)`.
  - UTF-16 selection/range contracts via `UTF16Selection` plus `bounds_for_range` / `character_index_for_point` for IME candidate window placement.
- **Scrolling**:
  - `ScrollHandle`, `div().overflow_y_scroll()`, `.track_scroll(...)`, `.scrollbar_width(...)`, `.block_mouse_except_scroll()`.
  - Virtualized list primitives: `UniformList` (uniform row height) and `list` (variable height).
- **Async**: `Context<T>::spawn(...)` + `AsyncApp`, `Task`, and `Timer::after(...)`.
- **Animation**: `with_animation(...)`, `Animation`, and easing functions.
- **Tooltips**: `Div::tooltip(...)` / `hoverable_tooltip(...)`.

### `redesmyn_ui` (what we built)

Location: `rust/crates/redesmyn_ui`.

- **Theme + tokens**
  - `settings::ThemePreference` (`dark` / `light` / `system`) persisted to `ui_settings.json` adjacent to `redesmyn_config::global_config_path()`.
  - `styles::UiTheme` derived from preference + `WindowAppearance`, with typed tokens:
    - `ColorTokens` (Rose Pine / Rose Pine Dawn)
    - `SpacingTokens` (density + scale)
    - `RadiusTokens`, `TypographyTokens`, `AnimationDurations`
  - Global wiring via `UiContext` + `utils::theme_for_window(...)`.
- **No silent actions**
  - `utils::UserActionState`: minimal shared state for in-flight actions (`in_flight`, `error`, `start/succeed/fail`) to drive progress + disable triggers.
- **Shared widgets**
  - `components::TextButton` / `components::IconButton` (+ `ButtonKind`) with disabled reason tooltips.
  - `components::Tooltip` (content view used by `Div::tooltip`).
  - `components::Callout` (`Info|Warning|Danger`) for in-place actionable errors.
  - `components::ProgressPill` with text-based “…” animation (no spinner wheels).
  - `components::ScrollArea` (thin wrapper over GPUI scroll primitives).
  - `components::SplitPane` (resizable + collapsible; `SplitPaneState` is `serde`-serializable for persistence).
  - `components::TextInput` / `components::TextArea` with shared key bindings via `components::bind_text_input_keys(...)`.

### Demo surface (end-to-end)

- `rust/crates/redesmyn_desktop/src/foundations_demo.rs`: `FoundationsDemo` uses theme toggles, buttons/tooltips, callouts, progress, split pane, scroll area, and text input/area.

### GPUI gaps / follow-ups

- **Widget gaps**: GPUI core intentionally does not ship a “full widget kit” (buttons, text fields, toasts, etc.). `redesmyn_ui` is our merge-friendly layer for these primitives.
- **Text editing depth**: current `TextArea` is intentionally minimal (no wrapping, no up/down navigation, limited editor semantics); harden before session/diff viewers.
- **Scrollbars**: current `ScrollArea` is minimal; follow up with consistent scrollbar UX (show/hide, reserved thumb space, theming).
- **Build quirks**
  - `gpui` is configured with the `runtime_shaders` feature to avoid requiring a local Metal toolchain at build time on macOS.
  - `core-text` is pinned in `rust/Cargo.lock` to avoid a `core-graphics` type mismatch in transitive font dependencies.
