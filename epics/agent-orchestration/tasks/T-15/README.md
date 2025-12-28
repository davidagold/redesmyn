# T-15 Harness profiles + `rn agent doctor` validation

## Metadata

```yaml
id: T-15
stacked_on: T-3
node:
  branch: rn/agent-orchestration/T-15-harness-profiles-doctor
```

## Brief (local)

- Define a data-driven “harness profile” format that describes:
  - launch (argv, cwd rules), env injection
  - attach semantics (tmux attach command or best-effort alternative)
  - capabilities (hooks/no hooks, CLI/GUI, supports PATH shimming, etc.)
  - UX guidance (bootstrap prelude / skill recommendation, e.g. via `agentskills.io`)
- Persist profiles in the `harness_profiles` registry table (built-in + user-defined):
  - Profiles are the archetypal definition; sessions store `harness_profile_id` plus a resolved snapshot used for that run.
- Implement validation tooling:
  - `rn agent doctor <harness>` to run preflight checks (binary present, tmux available if needed, PATH shim viability, etc.)
  - clear degraded-mode reporting (e.g. “git shim unavailable; invariant enforcement is degraded and some workflows may require manual steps while attached”)
- Provide a single place for a harness support matrix (capability table derived from profiles).

## Acceptance Criteria

- The runner uses profiles rather than hardcoded per-harness logic for v0 launch/attach behavior.
- `rn agent doctor` produces actionable, specific remediation for common failure modes.
- Profiles can be implemented in parallel per harness after this task lands.
