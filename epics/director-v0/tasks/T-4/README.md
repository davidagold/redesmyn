---
rn:
  node:
    branch: rn/director-v0/T-4-git-bundle-artifacts
  parent: T-3
---

# T-4 Remote change delivery via git bundle artifacts

## Plan

- Specify “commit SHA as change identity” as the default, and define the fallback when the control plane can’t
  fetch the commit objects directly.
- Define a “git bundle” artifact workflow:
  - remote executor creates a bundle containing the candidate commit(s) + required history
  - bundle is uploaded/streamed as an artifact to the control plane
  - designated executor imports the bundle (fetch) and makes the ref available locally
- Define how this integrates with merge queue + gating:
  - what ref the queue points at before/after import
  - how to surface diffs/logs for review in UI

## Acceptance Criteria

- A remote executor without push credentials can still deliver reviewable, reproducible changes.
- The control plane can surface “what changed” using git-native objects once the bundle is imported.
- Artifacts are immutable and content-addressable (recommended) so the same change can be re-applied safely.
