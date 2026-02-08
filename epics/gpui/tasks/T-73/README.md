---
epic: gpui
branch:
  suggested: rn/gpui/T-73-text-editing-actions-v1
rn:
  parent: T-59
  after:
    - T-44
---

# T-73 Text editing actions v1: word/line nav + deletion (TextInput/TextArea) (Domain 7)

## Problem

Our `TextInput` / `TextArea` key support is currently minimal:

- arrow left/right are character-step only,
- there is no word navigation/deletion (Option/Ctrl modifiers),
- and “line start/end” navigation on macOS (Cmd+Left/Cmd+Right) is missing.

This makes the session composer and other text fields feel non-native and slows down dogfooding.

## Goal

Add a coherent, semantic text-editing action surface and default bindings for `TextInput` and `TextArea`:

- word navigation (move + select),
- word deletion (backward/forward),
- line start/end navigation (move + select),
- line deletion to start/end (optional but recommended),

implemented via GPUI actions + keymap bindings (not widget-specific key handling).

## Requirements

### 1) Add semantic actions (no “raw key handling”)

Extend `redesmyn_ui_text_input` actions with a minimal, explicit set. Suggested actions:

- `MoveWordLeft` / `MoveWordRight`
- `SelectWordLeft` / `SelectWordRight`
- `DeleteWordBackward` / `DeleteWordForward`
- `MoveLineStart` / `MoveLineEnd`
- `SelectLineStart` / `SelectLineEnd`
- (Optional) `DeleteToLineStart` / `DeleteToLineEnd`

Notes:

- Keep action names stable; these become the long-term API that T-74 keymap customization will target.
- Actions must apply to both `TextInput` and `TextArea` where they make sense.

### 2) Default keybindings (macOS-first, with cross-platform parity)

Add default bindings in `bind_text_input_keys(...)`:

**macOS**
- `alt-left` / `alt-right` → move by word
- `shift-alt-left` / `shift-alt-right` → select by word
- `alt-backspace` / `alt-delete` → delete word backward/forward
- `cmd-left` / `cmd-right` → move to line start/end
- `shift-cmd-left` / `shift-cmd-right` → select to line start/end
- (Optional) `cmd-backspace` / `cmd-delete` → delete to line start/end

**Windows/Linux**
- `ctrl-left/right`, `ctrl-backspace/delete`, `home/end` equivalents

### 3) Word boundary semantics (deterministic)

Implement word navigation/deletion with deterministic semantics:

- Prefer Unicode word boundaries if a lightweight dependency is acceptable (e.g. `unicode-segmentation`).
- If not, define a clear fallback: treat `[A-Za-z0-9_]` as “word” and everything else as separators.

Be explicit and add unit tests for the chosen behavior.

### 4) Line start/end semantics

For `TextArea`, “line” means the **logical line** delimited by `\n` (not visual wrapped lines).

For `TextInput`, line start/end equals document start/end.

### 5) Integration contract with T-72 (parallel work)

T-72 is simultaneously hardening TextArea wrapping/scroll/IME. To avoid conflicts:

- T-73 MUST NOT change `TextAreaElement` layout/paint/wrapping logic.
- Keep changes in `rust/crates/redesmyn_ui/src/components/text_input.rs` limited to:
  - `actions!(...)` list additions,
  - `bind_text_input_keys(...)` additions,
  - new handler methods on `TextInput` / `TextArea`,
  - and wiring those handlers via `.on_action(...)`.

Caret visibility under overflow is owned by T-72. T-73 should just update selection/caret state and `cx.notify()`.

### 6) Regression coverage

Add unit tests for:

- word boundary movement and deletion (at least ASCII + a couple of Unicode cases if supported),
- line start/end movement on multi-line content,
- selection extension (shift-modified variants).

## Acceptance criteria

- On macOS, `TextInput` and `TextArea` support standard word/line navigation and deletion key combos.
- Behaviors are consistent and deterministic (documented by tests).
- No hardcoded widget-specific key handling; everything is actions + bindings.

## Dependencies / sequencing

- Depends on: T-44 (text input baseline), T-59 (SessionView uses the components).
- Parallel with: T-72 (TextArea wrap/autogrow/scroll).
- Enables: T-74 (user keymap customization) by providing stable action names.

## Code pointers (today)

- `TextInput` / `TextArea`: `rust/crates/redesmyn_ui/src/components/text_input.rs`
- Existing key bindings live in: `bind_text_input_keys(...)` in the same file.
