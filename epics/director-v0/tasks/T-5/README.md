---
rn:
  node:
    branch: rn/director-v0/T-5-director-session-ui-v0
  parent: T-4
---

# T-5 Director mode session UX v0

## Implementation Boundary

- Implement UI behavior in the Rust GPUI desktop app (`redesmyn_desktop` + Rust UI crates) only.
- Do not add or rely on legacy Python/webview UI implementations for this task.

## Plan

- Use the epic-pinned director session as the primary UI.
- Add a floating, non-scrolling `Director mode` affordance in the top-right of the session timeline viewport.
- Add activation flow:
  - user chooses `Run in current session` or `Run in new session`,
  - after intent selection, activation is one click.
- Add composer interaction contract for director mode:
  - normal send transforms inline into two-step `Pause & Send` (no dialog/modal),
  - `Steer` toggle allows one-shot manual instruction without pausing orchestration,
  - add keyboard shortcut `Cmd+.` to toggle `Steer`.
- Add integrated active/idle director visual treatment:
  - visible but calm structural styling,
  - active and idle states are visually distinct,
  - avoid badge accumulation and duplicate labels.
- Surface interruption/backlog state in a fixed non-modal surface near the session UI (not timeline event spam).

## Acceptance Criteria

- A user can clearly tell when the director session is actively auto-directing the epic.
- Users can activate director mode from the session viewport and choose current/new session path.
- Manual send in director mode requires explicit `Pause & Send`, while `Steer` can send without pausing.
- `Steer` toggle is available in composer with working `Cmd+.` shortcut.
- Implementation targets Rust GPUI UI surfaces, not legacy UI paths.
