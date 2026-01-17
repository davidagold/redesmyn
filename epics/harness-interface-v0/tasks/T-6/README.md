---
epic: harness-interface-v0
branch:
  suggested: rn/harness-interface-v0/T-6-agent-doctor
rn:
  linear:
    issue_id: ef3c0d08-31ca-49e2-9b1e-4f9fa51f3324
    identifier: RED-23
  parent: T-2
---

# T-6 Agent doctor + capabilities surface (UI + CLI)

## Problem

When advanced agent features fail, users need to understand:

- which agent kind is in use
- which capabilities are available
- why a feature (like conflict auto-assist) is disabled

Today, the “agent” mostly looks like a command string; capability visibility is low.

## Goal

Add a “doctor” surface and a capabilities summary that makes agent integration legible:

- UI: show selected/inferred agent kind and capabilities; explain why features are unavailable.
- CLI: provide a command to report agent kind + capabilities for a task (and for a command string).

## Requirements

### 1) UI: capabilities surface

In Configure (and/or agent details):

- show “Agent: Auto/Codex/Claude/Generic”
- show a concise capability list (or badges) derived from the agent implementation
- use this to gate advanced actions, with clear copy (no stack traces / noisy internals)
- if available, show “Resume handle” (agent program session identifier) separately from “Attach” (tmux)

### 2) CLI: doctor command

Provide a simple CLI surface (name TBD):

- `rn agent doctor --command "<cmd>"`
- `rn agent doctor --task <id>`

It should print:

- inferred/selected agent kind
- capabilities
- any degraded-mode warnings (e.g. “cannot detect turn complete; conflict assist disabled”)
- external resume handle (if present), and whether it supports:
  - resume by id
  - continue in cwd
  - interactive attach

## Acceptance criteria

- Users can tell, from UI/CLI, whether conflict auto-assist is expected to work and why/why not.
- The output is concise and actionable (avoid walls of text by default; allow `--verbose` later if needed).
