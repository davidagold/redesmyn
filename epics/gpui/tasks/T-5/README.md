---
epic: gpui
branch:
  suggested: rn/gpui/T-5-typed-config-layer
rn:
  parent: T-1
---

# T-5 Typed config layer (Domain 0)

## Problem

We need a coherent config story across:

- the desktop app (embedded control plane + modular daemon),
- the `rn` CLI,
- eventual remote daemons and remote control plane deployments.

If each crate parses env vars ad-hoc, we will accumulate inconsistent behavior and fragile defaults.

## Goal

Create a typed configuration layer that:

- supports layered sources (defaults → config file(s) → env overrides),
- provides good diagnostics (actionable errors),
- is explicit about “dev vs release” behaviors,
- keeps control-plane/daemon config concerns separate.

## Requirements

### 1) Config model

Define typed config structs (crate: `redesmyn_config` or similar):

- control plane settings (DB path, bind address, auth settings, etc.)
- daemon settings (repo registry, executor options, sandbox options, etc.)
- desktop settings (windowing, local embedding toggles, etc.)

### 2) Source precedence

Implement layered loading:

1. compiled defaults
2. config file(s) (TOML)
3. environment overrides

### 3) Paths and locations

Choose defaults that work on macOS and Linux (XDG on Linux).

During the split-codebase port:

- avoid disrupting the existing Python `.env` conventions,
- but clearly document which settings are “Rust world” vs “legacy world”.

### 4) Validation

- validate required fields at startup,
- do not defer validation until runtime errors occur in deep subsystems.

## Acceptance criteria

- Control plane and daemon can load config deterministically with good error messages.
- The desktop app can supply embedded config to both modules without special casing.
- Config layer is documented and stable for parallel implementers.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on `epics/gpui/tasks/T-1/README.md` (workspace + crate skeletons).
- Should coordinate with `epics/gpui/tasks/T-6/README.md` (DB config), `epics/gpui/tasks/T-7/README.md` (transport endpoints), and the desktop/server/daemon crate scaffolds from T-1.

## Reference implementation (today; configuration layering)

- Settings (Python today):
  - `redesmyn/settings.py` (`pydantic_settings`; env prefix `REDESMYN_`; `.env` support).
- Orchestration defaults (Python today):
  - `redesmyn/orchestration_config.py` (TOML config layering: global XDG config + repo-scoped config under state dir).
- CLI usage (Python today):
  - `redesmyn/cli.py` (calls `load_settings(...)`; also sets env vars in `rn debug dev`).
