# T-9 Simplify `justfile`: dev-only glue aligned with CLI

## Metadata

```yaml
id: T-9
stacked_on: T-7
must_land_after:
  - T-8
node:
  branch: rn/v0-launch/T-9-justfile-coherence
```

## Plan

Right now, `just` is acting like an alternate product surface (including spawning multiple processes in ways that don’t
match the desired architecture). For v0, “how you run Redesmyn” must be the CLI; `just` is for developer convenience only.

This task should be done late in the epic after the canonical CLI commands exist (`rn daemon up`, `rn up`).

### Work

1) Remove mixed-concern entrypoints

- Delete `just run --local`-style behavior that spawns server+daemon in confusing combinations.
- Ensure there is no `just` target that suggests “start the system” in a different way than `rn up`.

2) Keep dev-only targets

- Keep:
  - `just dev` for local dev (HMR/reload)
  - `just check/test/format`
  - packaging/build helpers if needed

3) Align naming and messaging

- Any `just` help text should point to `rn up`/`rn daemon up` for “real” usage.

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
