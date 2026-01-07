# T-1 V0 CLI contract: remove `rn observer` + remove `rn dev`

## Metadata

```yaml
id: T-1
stacked_on: null
node:
  branch: rn/v0-launch/T-1-cli-contract
```

## Brief (local)

Lock the v0 command surface so there is one coherent model:

- Remove the user-facing `rn observer …` command family (observer becomes a daemon capability).
- Remove `rn dev` from the CLI (dev HMR/reload stays in `just dev`).
- Establish the high-level UX vocabulary:
  - `rn up` / `rn down` as the user-facing “start/stop local system” entrypoints.
  - `rn server …` for control plane management.
  - `rn daemon …` for executor management (`up/down/status`, attach/registry, logs).

## Acceptance Criteria

- `rn --help` no longer lists `observer` or `dev`.
- Any prior entrypoint that referenced `rn dev` or `rn observer` is redirected to `just dev` or `rn daemon …` with actionable errors.
- The CLI help text makes the split explicit:
  - server = persistence/UI/API
  - daemon = repo executor (git/worktrees/agents/telemetry)

