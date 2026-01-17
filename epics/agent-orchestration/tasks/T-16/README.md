---
rn:
  node:
    branch: rn/agent-orchestration/T-16-settings-defaults
  parent: T-15
---

# T-16 Settings defaults: harness + prelude configuration

## Plan

Make agent orchestration configurable and ergonomic for local dogfooding:

- Add repo-scoped defaults in `.redesmyn/config.toml` for:
  - harness command (e.g. `codex ...`)
  - run mode (detached tmux session vs foreground)
  - agent prelude template (with placeholder interpolation)
- Expose these defaults in the dashboard in a “Configure” surface that is readable and tasteful.

Defaults should be usable by both:

- CLI flows (`rn agent start`, `rn agent run`, etc.)
- Dashboard task actions (“Start”, “Restart”, etc.)

## Acceptance Criteria

- Users can set and persist defaults via `rn config set ...` and see them reflected in the dashboard.
- Starting/restarting an agent from the dashboard uses these defaults unless explicitly overridden for that run.
- The prelude supports placeholders and is delivered to the harness at startup.
