# T-4 Claude Code agent interface implementation (turn detection + capabilities)

## Metadata

```yaml
id: T-4
epic: harness-interface-v0
stacked_on: T-7
branch:
  suggested: rn/harness-interface-v0/T-4-claude-code-agent
linear:
  issue_id: c5f5a23e-c8e7-4e69-94a3-e91b93be08ca
  identifier: RED-21
```

## Problem

Claude Code has different runtime behavior and output patterns than Codex.
We need an agent implementation that can:

- detect turn completion / readiness (when possible)
- provide correct capability gating
- support the same higher-level workflows (e.g. conflict remediation) when supported

## Goal

Implement a Claude Code agent interface implementation with parity to Codex goals, within what Claude Code can reliably signal.

## Notes / implementation guidance

## Research requirement

The implementer should do web research (docs + GitHub if open source) to identify the most reliable signals available for Claude Code completion/idle detection.

Use the same layered approach:

- explicit signals if available
- heuristics as fallback
- timeouts + “unknown” states as a safe failure mode

### Concrete findings (Jan 2026)

- Claude Code “print mode” supports JSON output (`--output-format json` / `stream-json`) that includes a `session_id`.
- Claude Code supports resuming:
  - by explicit session id (`--resume <id>`), and/or
  - “continue” semantics scoped to the working directory (`--continue`).

Implications for this task:

- Persist the Claude `session_id` as the external resume handle for the Redesmyn session.
- Model “resume by id” and “continue in cwd” as separate capabilities (the UI/CLI should only offer what is supported).
- Prefer structured stream parsing when available; keep heuristics as a fallback for purely interactive tmux-first sessions.

## Acceptance criteria

- Claude Code sessions produce `turn_complete` events in the common case.
- Capability gating is correct (features are enabled only when safe).
- Implementation behavior is testable and does not rely on flaky timing.
