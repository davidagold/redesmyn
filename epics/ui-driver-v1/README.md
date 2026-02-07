---
rn:
  slug: ui-driver-v1
  name: UI Driver v1
  root_branch: gpui
  linear:
    project_id: null
---

# UI Driver v1 Epic: Control Doc (Canonical)

This file is the canonical control doc for **UI Driver v1**: intent, scope, invariants, architecture, and task sequencing.

## 1) Vision

Make desktop UI automation:

- script-driven (no hardcoded smoke flows in Rust code),
- parity-safe with real user behavior,
- instance-safe for multi-composer/multi-pane UI,
- discoverable and targetable for dynamic agent-driven testing,
- and explicitly scoped so UI-driver responsibilities do not duplicate control-plane domain APIs.

## 2) Why this epic exists

Current pain points:

- `rn ui-driver` smoke scenarios are hardwired in CLI code.
- Some UI-driver operations bypass user-path guards and focus/context rules.
- Multi-instance UI surfaces (left session vs task-card session) have had coupling/targeting bugs.
- There is no generic target discovery model (IDs/roles/actions) for dynamic automation.
- The boundary between UI-driver and control-plane responsibilities is not explicit enough.

## 3) Architectural principles (v1)

### 3.1 Replace fixed smokers with data-driven scripts

`rn ui-driver` should execute instruction streams (LDJSON/NDJSON), not baked-in scenario functions.

### 3.2 Single semantic operation layer for parity

For automatable UI behavior, keyboard/mouse handlers and UI-driver handlers should call the same semantic operations.

### 3.3 Explicit targeting model

Any operation that can apply to more than one UI instance must require an explicit target scope.

### 3.4 UI-driver vs control-plane boundary

- UI-driver: UI state/actions, focus, menus, selection, wait/snapshot/screenshot, element-target actions.
- Control-plane protocol: domain mutations and durable business operations.
- Script runner can orchestrate both channels, but ownership remains explicit.

### 3.5 Deterministic, inspectable automation

Automation should expose:

- stable target IDs,
- target metadata (role/state/actions),
- deterministic waits,
- and machine-readable per-step outcomes.

## 4) Scope (v1)

### In scope

- Replace hardcoded smoke subcommands with a script runner.
- Migrate existing smoke coverage to checked-in scripts.
- Introduce shared UI automation operation layer.
- Add explicit target scope for session/composer-bound actions.
- Add snapshot target registry and generic target actions (including scroll).
- Add parity and conformance tests.
- Enforce UI-driver/control-plane ownership boundaries.

### Out of scope

- Broad redesign of unrelated UI surfaces.
- Coordinate-based low-level click automation.
- Remote/network UI-driver transport.

## 5) Success criteria

- No hardcoded smoke flow code remains in `rn ui-driver`.
- Script-driven runs can fully replace current smoke coverage.
- Known parity regressions (focus/context/instance targeting) are structurally prevented.
- Dynamic automation can enumerate targets and invoke actions by target ID.
- UI-driver method ownership is explicit and enforced.

## 6) Task map

- `epics/ui-driver-v1/tasks/T-1/README.md`: v1 contract + ownership matrix + migration plan.
- `epics/ui-driver-v1/tasks/T-2/README.md`: script runner core and removal of fixed smoke subcommands.
- `epics/ui-driver-v1/tasks/T-3/README.md`: script fixtures + docs + CI migration.
- `epics/ui-driver-v1/tasks/T-4/README.md`: shared semantic operation layer in desktop UI.
- `epics/ui-driver-v1/tasks/T-5/README.md`: user-vs-driver parity conformance tests.
- `epics/ui-driver-v1/tasks/T-6/README.md`: explicit instance-scoped target model for session/composer actions.
- `epics/ui-driver-v1/tasks/T-7/README.md`: `UiSnapshot` target registry (discoverable target catalog).
- `epics/ui-driver-v1/tasks/T-8/README.md`: generic target actions (`activate`, `set_toggle`, `set_text`, `send_keys`, `scroll`).
- `epics/ui-driver-v1/tasks/T-9/README.md`: multi-channel script orchestration (`ui_driver` + `control_plane`).
- `epics/ui-driver-v1/tasks/T-10/README.md`: hardening + ownership enforcement + final cleanup.

## 7) Dependency graph (high level)

- T-1 is foundation.
- T-2 depends on T-1.
- T-3 depends on T-2.
- T-4 depends on T-1.
- T-5 depends on T-4 and T-2.
- T-6 depends on T-4.
- T-7 depends on T-6.
- T-8 depends on T-7.
- T-9 depends on T-2 and T-1.
- T-10 depends on T-3, T-5, T-8, and T-9.
