---
rn:
  node:
    branch: rn/remote-execution-v0/T-1-git-bundle-artifacts
  parent: null
---

# T-1 Remote change delivery via git bundle artifacts

## Plan

- Specify "commit SHA as change identity" as the default, and define fallback when the control plane cannot
  fetch commit objects directly.
- Define a git bundle artifact workflow:
  - remote executor creates a bundle containing candidate commit(s) plus required history,
  - bundle is uploaded/streamed via the artifact channel,
  - designated executor imports the bundle and materializes local refs.
- Define integration points with orchestrator flows:
  - what refs downstream queue/gate systems consume,
  - how provenance is retained from bundle to local git objects.

## Acceptance Criteria

- A remote executor without push credentials can still deliver reviewable, reproducible changes.
- Imported changes become git-native objects addressable by SHA in downstream flows.
- Artifact handling is immutable/content-addressed (recommended) so repeated apply is safe.
