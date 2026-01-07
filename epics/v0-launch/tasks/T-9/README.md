# T-9 Simplify `justfile`: dev-only glue aligned with CLI

## Metadata

```yaml
id: T-9
stacked_on: T-8
node:
  branch: rn/v0-launch/T-9-justfile-coherence
```

## Brief (local)

Make `just` a dev convenience layer, not a parallel product surface:

- Remove mixed-concern commands (e.g. ones that spawn both server + daemon in confusing combinations).
- Keep:
  - `just dev` for local HMR/reload
  - `just check/test/format`
  - (optional) build/package helpers
- Ensure `just` points at the canonical CLI entrypoints for “real” operation (`rn up`, `rn daemon …`, `rn server …`).

## Acceptance Criteria

- There is no `just` command that implies a different architecture than the CLI.
- The `justfile` reads as “developer conveniences”, not “how to run Redesmyn”.

