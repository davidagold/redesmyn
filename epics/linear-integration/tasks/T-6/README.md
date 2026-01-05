# T-6 Linear automation: push local status → Linear + sync indicator

## Metadata

```yaml
id: T-6
stacked_on: T-5
node:
  branch: rn/linear-integration/T-6-t-6-linear-automation-push-local-status-linear-sync-indicato
linear:
  issue_id: b4e4b9bc-9357-4f67-86d5-df5abacdb4be
  identifier: RED-16
```

## Brief (local)

- Add automation that updates Linear issue status based on local task lifecycle:
  - When an agent starts running a task, set the linked Linear issue to `in progress`.
  - When a task is marked `done` locally, set the linked Linear issue to `done`.
- Add a lightweight UI signal on the task’s Linear pill/icon when local state is known to be out of sync with Linear state.

## Acceptance Criteria

- When a task has `linear.issue_id` and the local task enters `in_progress`, Linear is updated (best-effort) to `started`.
- When a task has `linear.issue_id` and the local task enters `done`, Linear is updated (best-effort) to `completed`.
- The automation is opt-in or guarded to avoid surprising changes (e.g. only when Linear is connected and the epic label matches).
- Dashboard task nodes surface a subtle “sync stale” indicator when local/Linear state diverges (no extra noisy text).

## Notes / Design

- Use the existing coarse mapping in the epic control doc (§4.4).
- Treat all writes as best-effort:
  - If Linear is disconnected/unavailable, do not block local actions; record/log the failure.
- Prefer a single place to trigger updates (API/daemon), so both CLI actions and dashboard actions share behavior.
