# T-16 Settings: default fleet + harness configuration

## Metadata

```yaml
id: T-16
stacked_on: T-3
must_land_after:
  - T-3
  - T-8
node:
  branch: rn/agent-orchestration/T-16-settings-defaults
```

## Brief (local)

Introduce a small, coherent settings system so users don’t have to re-specify “how to run agents” every time:

- Default fleet sizing behavior (fixed size vs auto-size to eligible tasks)
- Default harness command template (and default args)
- Default detach behavior (tmux-first)
- Default epic selection (when multiple epics exist)

Settings should be usable by both:

- CLI flows (`rn run`, `rn agent start`, etc.)
- Dashboard epic-level “Run” panel (copy command + status)

## Acceptance Criteria

- `rn run` works without `--fleet-size` and without `--harness` when defaults are configured.
- When multiple epics exist, a configured default epic avoids “pass --epic …” errors.
- The dashboard can display the configured defaults in the epic “Run” panel and generate an accurate `rn run …` command.

## Final designs

### 1) Configuration layers

Support two layers (later layers override earlier):

1. **Global user config** (optional): applies across repos.
2. **Repo config** (recommended): applies to the current repo.

Repo config is the source of truth for dogfooding ergonomics; global config is for convenience.

### 2) Settings schema (v0)

At minimum:

- `default_epic: <slug|id> | null`
- `fleet`:
  - `mode: "fixed" | "auto"`
  - `size: int | null` (used when mode is fixed)
- `harness`:
  - `command: str` (shell-like command string; first token is the executable)
  - `detach: bool`

### 3) UX surfaces

- CLI:
  - `rn config get`
  - `rn config set <key> <value>` (or a small set of typed subcommands)
  - `rn run` reads defaults when flags are omitted.
- Dashboard:
  - Epic “Run” panel shows the effective settings and provides “Copy rn run …”.

### 4) Storage (v0)

Prefer simple, inspectable files:

- Repo-local: `.redesmyn/config.toml` (or `.redesmyn/config.json` if TOML is undesirable)
- Global: `~/.config/redesmyn/config.toml`

Do not hide settings in opaque DB rows for v0; they should be easy to version and share.

### 5) Eligibility reference

Fleet sizing (“auto”) uses the eligibility rules defined in `T-3`:

- task has branch/node backing
- task state is `todo` or `in_progress`
- exclude `blocked` and `done` by default
