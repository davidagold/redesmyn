---
rn:
  node:
    branch: rn/director-v0/T-6-director-mode-lifecycle-policy
  parent: T-1
---

# T-6 Director mode lifecycle + merge authority policy

## Implementation Boundary

- Implement director mode lifecycle and policy state in Rust control-plane/desktop integration surfaces.
- Do not add or rely on legacy Python/webview implementations for this task.

## Plan

- Define director mode lifecycle state for each epic:
  - inactive
  - active
  - paused
  - error/resume-required
- Define activation semantics:
  - run in current session
  - run in new session
  - one-click activation after intent selection
- Define interruption semantics:
  - director/session interruption transitions mode to resume-required
  - user must explicitly resume (no silent auto-resume in v0)
- Define merge authority policy model:
  - global default policy
  - per-epic override
  - `yolo_merge` flag defaulting to false
- Define policy/read-model surfaces consumed by UI and controller:
  - current effective policy for selected epic
  - rationale/source (`global-default` vs `epic-override`)

## Acceptance Criteria

- Lifecycle states and transitions are explicit, durable, and queryable.
- Activation and resume-required semantics are deterministic and testable.
- `yolo_merge` is off by default globally and overridable per epic.
- Effective policy resolution is consistent between controller and UI surfaces.
- Implementation targets Rust runtime paths (control plane + desktop integration), not legacy paths.
