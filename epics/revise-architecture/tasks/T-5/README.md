---
rn:
  node:
    branch: rn/revise-architecture/T-5-dashboard-daemon-status
  linear:
    issue_id: cb10b4c4-ed31-413c-bfd9-a62a981cd713
    identifier: RED-34
  parent: T-2
---

# T-5 Dashboard: daemon status + offline guidance

## Brief (local)

- Surface daemon connection status in the dashboard:
  - **daemon (host) connection**: online/offline + last_seen
  - **repo telemetry**: attached/unattached + freshness (stale threshold); “attached” means the daemon is actively managing this repo
  - host identity (when meaningful)
  - primary repo executor (when meaningful):
    - which host is currently the “writer” for git-mutating actions (merge/restack/etc.)
    - if no primary executor is available, degrade git-mutating UI to disabled + guidance
  - “telemetry stale” affordances
- Prefer a compact, always-visible surface (e.g. RHS of the subheader bar) with hover/click details.
- When offline, provide actionable guidance:
  - copy `rn daemon up` (or the configured equivalent)
  - troubleshooting link/section (basic)
- Ensure these UI surfaces remain graph-first (avoid table-heavy “agents list” as the primary view).

## Acceptance Criteria

- A user can tell at a glance whether the daemon is connected and whether updates should be expected.
- The UI provides a single-click path to fix the most common issue (“daemon not running”).

## Updates

### 2025-12-31

- Recent work introduced “out of sync with upstream” surfacing on task cards (`stackInSync`).
  This task should ensure the UI behavior is correct when:
  - the daemon is offline (projection should be unknown/stale, not silently wrong), and
  - multiple daemons/hosts exist (only the primary executor should be allowed to run merge/restack actions).
