---
epic: gpui
branch:
  suggested: rn/gpui/T-75-text-editing-polish-v2
rn:
  node:
    branch: rn/gpui/T-75-text-editing-polish-v2
  parent: T-59
  after:
    - T-72
    - T-73
    - T-74
---

# T-75 Text editing polish v2: undo/redo + vertical nav + mouse selection (Domain 7)

## Problem

Even after:

- T-72 (TextArea wrap + auto-grow + internal scroll),
- T-73 (word/line nav + deletion actions),
- T-74 (user keymap overrides),

our text controls will still feel “non-editor” in a few high-impact ways that matter for an LLM
orchestration app:

- no undo/redo (users can’t safely iterate on prompts),
- multiline vertical navigation is incomplete (hard to edit longer messages),
- mouse selection feels underpowered compared to what users expect (double-click select word, drag selection auto-scroll, etc.),
- and cursor stepping may split grapheme clusters (emoji/combining marks).

We don’t need a full editor widget, but we *do* need a “composer-grade” text experience.

## Goal

Upgrade `TextInput` and `TextArea` to a “composer-grade” baseline:

- undo/redo with reasonable grouping,
- robust Up/Down navigation in `TextArea` (wrap-aware),
- mouse selection affordances (double/triple click, drag autoscroll),
- and grapheme-cluster-safe character stepping.

## Requirements

### 1) Undo/redo stack (with grouping)

Implement undo/redo for both `TextInput` and `TextArea`:

- Maintain an edit history (bounded) that captures:
  - prior content,
  - prior selection (and reversed-ness if needed),
  - and (optionally) marked range for IME.
- Group consecutive inserts into a single undo step when they are “part of the same thought”:
  - consecutive typed characters within a short time window,
  - and/or consecutive inserts adjacent to the caret without intervening cursor moves.
- Treat “structural” edits as hard boundaries:
  - paste,
  - cut,
  - deletion,
  - selection replace,
  - newline insertion.

Default bindings:

- macOS: `cmd-z` undo, `shift-cmd-z` redo (also support `cmd-y` redo if desired).
- Windows/Linux: `ctrl-z` undo, `ctrl-shift-z` or `ctrl-y` redo.

Actions:

- Add `Undo` / `Redo` as GPUI actions under the same action namespace as other text input actions
  (so user keymaps can override them via T-74).

### 2) Grapheme-cluster-safe “character” stepping

For Left/Right, Backspace/Delete in the “single-character” variants:

- Move/delete by extended grapheme cluster, not Unicode scalar value.
- Use `unicode-segmentation` if available (T-73 may already choose it for word boundaries).

If we cannot depend on segmentation for some reason, document the fallback and add tests showing the behavior.

### 3) Vertical navigation in TextArea (wrap-aware)

Implement:

- `Up` / `Down` caret movement (and `Shift-Up` / `Shift-Down` selection extension),
- `PageUp` / `PageDown` (optional but recommended),

with these semantics:

- Moves operate on **visual lines** (wrapped lines), not logical `\n` lines.
- Preserve a “desired x” column so repeated Up/Down stays in the same visual column even when line lengths vary.
- Scrolling:
  - moving the caret must ensure caret visibility (works with T-72’s internal scroll model).

Keybindings:

- Bind these as actions (similar to T-73) and wire through `.on_action(...)` for `TextArea`.

### 4) Mouse selection affordances

Implement the common mouse interactions:

- single click: place caret
- shift-click: extend selection from anchor to clicked position
- double click: select word at point
- triple click: select logical line at point (between `\n` boundaries)
- drag selection: extends selection while dragging, with **auto-scroll** when dragging beyond the visible viewport

Notes:

- Word selection should reuse the same word-boundary logic as keyboard word actions (T-73) for consistency.
- Auto-scroll should use the same scroll model as T-72 (no second competing scroll state).

### 5) Tests (deterministic)

Add unit tests for the pure editing logic:

- undo/redo correctness across:
  - typed insert grouping,
  - paste,
  - selection replace,
  - deletion.
- grapheme-safe stepping on at least:
  - emoji sequences / combining marks (a couple representative cases).
- vertical navigation desired-column semantics (can be tested against synthetic wrapped-line metrics if GPUI layout isn’t available in unit tests).

If GPUI-level integration tests exist for input handling, add one minimal regression there as well.

## Acceptance criteria

- Users can confidently edit prompts with undo/redo in both `TextInput` and `TextArea`.
- `TextArea` supports wrap-aware Up/Down movement without “jumps”, preserving a desired column.
- Mouse selection feels standard (shift-click, double/triple click, drag autoscroll).
- Cursor movement/deletion does not split grapheme clusters.
- Works with T-74 user keymap overrides (actions are stable and string-addressable).

## Dependencies / sequencing

- Stacked after:
  - T-72 (wrap/scroll + IME correctness),
  - T-73 (text actions + word boundaries),
  - T-74 (keymap overrides).

## Code pointers (today)

- Text controls: `rust/crates/redesmyn_ui/src/components/text_input.rs`
