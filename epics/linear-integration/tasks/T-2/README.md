---
rn:
  node:
    branch: rn/linear-integration/T-2-linear-client-write
  linear:
    issue_id: e173b21e-02b2-4f33-9b1b-aace082e8199
    identifier: RED-12
  parent: T-1
---

# T-2 Linear client: write support (labels, state, dependencies, create/update)

## Brief (local)

- Extend the Linear integration layer to support the operations required for `sync --to linear`:
  - read/write issues in a project
  - ensure/apply a label equal to the epic slug
  - create/update issue title/description/state
  - read/write “blocked by” relations
- Establish a “default team” selection for issue creation.

## Acceptance Criteria

- There are Linear client helpers for:
  - listing issues in a project filtered by label (epic slug)
  - resolving/creating the epic slug label and applying it to an issue
  - creating an issue in the configured project/team
  - updating title/description/state
  - reading/writing blocker relations
- “Default team” is selected automatically (v0) and stored so subsequent creates are stable.

## Notes / Design

- Store ids (team id, label id) as needed; prefer stable ids over names for writes.
- Keep the API surface small and testable (GraphQL wrappers with typed return objects).
