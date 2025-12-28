# T-5 Dashboard: daemon status + offline guidance

## Metadata

```yaml
id: T-5
stacked_on: T-2
node:
  branch: rn/revise-architecture/T-5-dashboard-daemon-status
```

## Brief (local)

- Surface daemon connection status in the dashboard:
  - online/offline + last_seen
  - host identity (when meaningful)
  - “telemetry stale” affordances
- When offline, provide actionable guidance:
  - copy `rn up` (or the configured equivalent)
  - troubleshooting link/section (basic)
- Ensure these UI surfaces remain graph-first (avoid table-heavy “agents list” as the primary view).

## Acceptance Criteria

- A user can tell at a glance whether the daemon is connected and whether updates should be expected.
- The UI provides a single-click path to fix the most common issue (“daemon not running”).
