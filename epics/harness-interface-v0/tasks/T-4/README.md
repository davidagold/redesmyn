# T-4 Claude Code harness adapter (turn detection + capabilities)

## Metadata

```yaml
id: T-4
epic: harness-interface-v0
stacked_on: T-2
branch:
  suggested: rn/harness-interface-v0/T-4-claude-code-harness
```

## Problem

Claude Code has different runtime behavior and output patterns than Codex.
We need a harness adapter that can:

- detect turn completion / readiness (when possible)
- provide correct capability gating
- support the same higher-level workflows (e.g. conflict remediation) when supported

## Goal

Implement a Claude Code harness adapter with parity to Codex adapter goals, within what Claude Code can reliably signal.

## Notes / implementation guidance

Use the same layered approach:

- explicit signals if available
- heuristics as fallback
- timeouts + “unknown” states as a safe failure mode

## Acceptance criteria

- Claude Code sessions produce `turn_complete` events in the common case.
- Capability gating is correct (features are enabled only when safe).
- Adapter behavior is testable and does not rely on flaky timing.

