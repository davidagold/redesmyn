---
epic: ui-driver-v1
branch:
  suggested: rn/ui-driver-v1/T-3-script-fixtures-and-ci-migration
rn:
  parent: T-2
---

# T-3 Script fixtures + docs + CI migration

## Problem

After replacing fixed smokers, we need durable script fixtures and invocation docs so coverage is maintained and easy to evolve.

## Goal

Migrate existing smoke coverage to script files and wire CI/docs to use them.

## Scope

1. Create versioned script fixtures for core smoke paths.
2. Add docs for authoring/running scripts locally and in CI.
3. Update existing automation invocations to use script runner.
4. Remove dead references to legacy smoke command names.

## Dependencies / sequencing

- Depends on T-2.

## Acceptance criteria

- Existing smoke coverage is represented as scripts.
- CI/local docs no longer reference removed smoke subcommands.
- Script fixtures are deterministic and readable.

## Validation

- CI invocation dry-run and local execution checks.
