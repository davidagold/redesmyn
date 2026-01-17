---
epic: gpui
branch:
  suggested: rn/gpui/T-4-tracing-logging
rn:
  parent: null
---

# T-4 Logging + tracing foundations (Domain 0)

## Problem

We need excellent observability to build and debug a distributed-by-design app:

- daemon ↔ control plane separation,
- high-throughput event streams,
- UI actions that must never be “silent”.

At the same time, we must avoid noisy logs and accidental leakage of large payloads.

## Goal

Establish a single tracing/logging approach across the Rust workspace that supports:

- debugging during porting,
- production-grade structured logs,
- UI progress instrumentation (“no silent actions”),
- future protocol wiretap tools.

## Requirements

### 1) Use `tracing`

- Standardize on `tracing` + `tracing_subscriber`.
- Provide a crate or module (`redesmyn_logging`) that exposes:
  - environment-driven log filtering,
  - dev-friendly formatting,
  - structured JSON output option for production.

### 2) Stable span fields

Define stable span keys and helper macros/functions to ensure consistent tagging:

- `workspace_id`, `repo_id`, `epic_id`, `task_id`
- `host_id`
- `run_id`, `command_id`

### 3) Payload logging policy

- Default: log metadata only.
- Provide opt-in sampling/redaction for payloads (especially protocol messages and diffs).

### 4) UI progress hooks

Define conventions so user actions can be traced and correlated with UI state:

- every user-triggered mutation gets an idempotency key / command id,
- UI uses that to show “in progress” and to prevent duplicate requests.

## Acceptance criteria

- A tiny demo (or test) emits spans with stable fields and produces readable dev logs.
- Switching between dev output and JSON output is one config flag.
- Logging policy is documented and provides a safe default (no giant payload dumps).

