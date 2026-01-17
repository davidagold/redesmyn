---
epic: gpui
branch:
  suggested: rn/gpui/T-5-config-layer
rn:
  parent: null
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

