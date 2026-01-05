# T-6 Harness doctor + capabilities surface (UI + CLI)

## Metadata

```yaml
id: T-6
epic: harness-interface-v0
stacked_on: T-2
branch:
  suggested: rn/harness-interface-v0/T-6-harness-doctor
```

## Problem

When advanced harness features fail, users need to understand:

- which harness is in use
- which capabilities are available
- why a feature (like conflict auto-assist) is disabled

Today, “harness” mostly looks like a command string; capability visibility is low.

## Goal

Add a “doctor” surface and a capabilities summary that makes harness integration legible:

- UI: show selected/inferred harness kind and capabilities; explain why features are unavailable.
- CLI: provide a command to report harness kind + capabilities for a task (and for a command string).

## Requirements

### 1) UI: capabilities surface

In Configure (and/or agent details):

- show “Harness: Auto/Codex/Claude/Generic”
- show a concise capability list (or badges) derived from the harness implementation
- use this to gate advanced actions, with clear copy (no stack traces / noisy internals)

### 2) CLI: doctor command

Provide a simple CLI surface (name TBD):

- `rn harness doctor --command "<cmd>"`
- `rn harness doctor --task <id>`

It should print:

- inferred/selected harness kind
- capabilities
- any degraded-mode warnings (e.g. “cannot detect turn complete; conflict assist disabled”)

## Acceptance criteria

- Users can tell, from UI/CLI, whether conflict auto-assist is expected to work and why/why not.
- The output is concise and actionable (avoid walls of text by default; allow `--verbose` later if needed).
