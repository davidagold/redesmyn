# T-4 Claude Code harness interface implementation (turn detection + capabilities)

## Metadata

```yaml
id: T-4
epic: harness-interface-v0
stacked_on: T-7
branch:
  suggested: rn/harness-interface-v0/T-4-claude-code-harness
```

## Problem

Claude Code has different runtime behavior and output patterns than Codex.
We need a harness implementation that can:

- detect turn completion / readiness (when possible)
- provide correct capability gating
- support the same higher-level workflows (e.g. conflict remediation) when supported

## Goal

Implement a Claude Code harness interface implementation with parity to Codex goals, within what Claude Code can reliably signal.

## Notes / implementation guidance

## Research requirement

The implementer should do web research (docs + GitHub if open source) to identify the most reliable signals available for Claude Code completion/idle detection.

Use the same layered approach:

- explicit signals if available
- heuristics as fallback
- timeouts + “unknown” states as a safe failure mode

## Acceptance criteria

- Claude Code sessions produce `turn_complete` events in the common case.
- Capability gating is correct (features are enabled only when safe).
- Implementation behavior is testable and does not rely on flaky timing.
