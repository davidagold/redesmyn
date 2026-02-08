---
epic: gpui
branch:
  suggested: rn/gpui/T-74-user-keymap-customization
rn:
  node:
    branch: rn/gpui/T-74-user-keymap-customization
  parent: T-73
  after:
    - T-44
---

# T-74 User keymap customization v1 (JSON5) (Domain 7)

## Problem

We are adding more semantic actions (T-73) and will continue to grow action surfaces across the app
(command palette, session viewer, graph view, etc.). If we hardcode key chords in many places, we’ll
end up with:

- inconsistent binding patterns,
- no safe way for users to customize keys,
- and ongoing merge pain when defaults evolve.

We need a single, explicit, user-editable “keymap” layer that maps key chords → action names, with
context scoping and unbinding support.

## Goal

Add support for loading a user keymap file (JSON5) at desktop startup that:

- overrides default bindings,
- can add new bindings,
- and can unbind defaults,

without requiring code changes for most keybinding tweaks.

## Requirements

### 1) Keymap file location + schema

Pick a stable path under the existing config directory (recommended):

- `~/.config/redesmyn/keymap.json5` (or adjacent to existing UI settings path if different)

Define a minimal schema (serde-deserializable), e.g.:

- a list of bindings:
  - `keys`: `"cmd-k"` / `"shift-alt-left"` etc
  - `action`: `"redesmyn_ui_text_input::MoveWordLeft"` (string form)
  - `context`: `"TextArea"` / `"TextInput"` / `"Desktop"` (optional; matches `.key_context(...)`)
  - `params`: optional JSON object for parameterized actions (future)
  - `when`: optional predicate (future; not required for v1)

### 2) Dynamic action construction

Do not require Rust types to be referenced from the keymap. Use GPUI’s dynamic builder:

- `App::build_action(action_name, params)` (or equivalent) to construct an action by string name.

Unbinding:

- Support an explicit “unbind” form (either `action: null` or `action: "zed::NoAction"`),
  so users can remove defaults.

### 3) Apply user keymap over defaults

We need predictable precedence:

- defaults are registered first,
- user keymap entries are then applied and win on conflicts.

Do not silently ignore invalid entries:

- log parse/build errors with enough detail to fix the file,
- but do not crash the app (fail open).

### 4) Documentation + template

Add a small doc + starter template so agents/users can actually use this:

- describe where the file lives,
- show a few representative overrides (e.g. rebind `MoveWordLeft`),
- show how to unbind.

This can live in `epics/gpui/README.md` or a nearby GPUI docs file.

## Acceptance criteria

- With a `keymap.json5` present, the desktop app applies overrides at startup.
- A user can rebind at least one action (e.g. a text editing action from T-73) without code changes.
- A user can unbind a default binding.
- Errors are surfaced via logs and do not crash the app.

## Dependencies / sequencing

- Depends on: T-73 (stable action names to target).
- Should remain generic and reusable across all GPUI surfaces (not text-input-specific).
