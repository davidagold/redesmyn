---
epic: gpui
branch:
  suggested: rn/gpui/T-72-textarea-wrap-autogrow
rn:
  node:
    branch: rn/gpui/T-72-textarea-wrap-autogrow
  parent: T-59
  after:
    - T-44
---

# T-72 TextArea v1: soft-wrap + auto-grow + internal scroll (Domain 7)

## Problem

Our `redesmyn_ui::components::TextArea` is currently intentionally minimal (T-44 “GPUI gaps”):

- long lines do **not** wrap and can visually overflow their container,
- the control does not provide a robust “grow until max height, then scroll” experience,
- and caret/selection visibility is not guaranteed once text exceeds the visible area.

This makes the session composer (T-62) feel broken/rough and blocks efficient dogfooding.

## Goal

Harden `TextArea` into a pleasant, reliable multi-line input suitable for:

- session composer,
- future diff/detail viewers,
- and any other multi-line text surfaces.

“Pleasant” here means: **soft wrap**, **no overflow**, **auto-grow** to a max height, then **scroll**,
with caret always kept visible.

## Requirements

### 1) Soft-wrap (no horizontal overflow)

Implement real wrapping for long lines:

- Use GPUI’s wrapped shaping (`WindowTextSystem::shape_text`) instead of per-line `shape_line(..., None)`.
- Wrap width must respect the available inner content width (bounds minus padding).
- Newlines remain hard breaks; wrapping happens within each logical line.

### 2) Layout + paint must clip to viewport

Even with wrap, text/selection/cursor painting must not draw outside the control:

- Ensure the TextArea visual viewport clips its children (e.g. `.overflow_hidden()` at the correct layer).
- Selection quads and caret quad must be clipped as well.

### 3) Auto-grow then scroll (caret stays visible)

Desired UX:

- Start at a small height (e.g. 3–5 rows / an explicit min height).
- As the user types and the wrapped line count increases, grow the control height up to a max (e.g. 10 rows).
- Once at max height, the TextArea becomes internally scrollable.
- When the caret moves (typing, paste, mouse click, selection collapse, programmatic set_text), scroll so the caret is visible (with a small margin/padding).

Implementation constraints:

- Prefer a **single scroll model** (either GPUI scroll primitives or an internal `scroll_y` offset), not both.
- Must work without introducing per-frame allocations (cache layout; recompute only on content/width/style changes).

### 4) IME correctness (UTF-16 selection + bounds)

We currently implement `EntityInputHandler` semantics, including:

- UTF-16 range conversion helpers
- caret/selection hit-testing
- `bounds_for_range` / `character_index_for_point` (for IME candidate placement)

After adding wrap + scroll:

- IME candidate windows must still appear at the correct caret location.
- `bounds_for_range` and hit-testing must be wrap-aware and scroll-aware.

### 5) Integration contract with T-73 (parallel work)

T-73 will add richer key-driven editing actions. To avoid merge conflicts and “two competing models”:

- T-72 MUST NOT add new keybindings or new semantic text-editing actions.
- T-72 SHOULD make caret-visibility robust by reacting to *any* selection/caret change (not only text insertion),
  so T-73 does not need to sprinkle “scroll to caret” calls everywhere.

### 6) Explicit regression coverage

Add at least one deterministic, non-flaky regression check that guards the reported UX break:

- long single-line input does not overflow horizontally (wrap is observable via layout metrics or by checking
  “max painted x <= bounds.right” in a snapshot/semantic assertion, if available),
- and/or caret remains visible after inserting enough text to exceed max height.

If full GPUI visual testing is not available yet, factor pure helpers (wrap measurement, scroll clamp) into
testable functions and unit-test those.

## Acceptance criteria

- In the session composer, typing/pasting long text:
  - wraps instead of overflowing,
  - grows until max height,
  - then scrolls internally while keeping caret visible.
- Selection and caret rendering remain correct under wrap and scroll.
- IME works (caret bounds are correct for candidate UI).
- Steady-state rendering is allocation-free (no per-frame `Vec` growth / repeated shaping when idle).

## Dependencies / sequencing

- Depends on: T-44 (UI foundations; current TextArea baseline), T-59 (SessionView scaffolding uses TextArea).
- Works in parallel with: T-73 (text editing actions + default bindings).
- Unblocks / improves: T-62 (composer UX), later session/diff viewers.

## Code pointers (today)

- `TextArea` implementation: `rust/crates/redesmyn_ui/src/components/text_input.rs`
- Session composer usage: `rust/crates/redesmyn_ui_session/src/lib.rs`
