---
epic: gpui
branch:
  suggested: rn/gpui/T-81-session-settings-menu
rn:
  parent: T-80
---

# T-81 UI: Session settings menu (approvals + sandbox) (Domain 7)

## Problem

The GPUI session viewer needs first-class affordances for agent “permissions” configuration:

- A user-visible **settings selector** in the composer (approvals + sandbox).
- Clear **permission request / decision** rendering when the agent asks for access.
- Session resumes / restarts must not silently regress to defaults; the UI should reflect the
  session’s effective configuration as soon as the session is opened.

## Goals

1. **Composer settings button**
   - Add a single `Settings` button in the composer action bar.
   - Use a two-level cascading menu:
     - Primary: `Permissions`, `Sandbox`
     - Secondary: provider-native options for each.

2. **Accurate on-open display**
   - Don’t show `Default` as if it were authoritative when the session has not yet been resolved.
   - Perform a lightweight “latest policy event” fetch on session open so the UI can display the
     most recent approvals/sandbox overrides immediately.

3. **Durable, re-applied policy**
   - Persist session-scoped policy choices in the durable session event log.
   - When starting/resuming a runner, hydrate it from the latest durable policy snapshot before
     issuing the next turn so restarts don’t regress to global defaults.

## Acceptance criteria

- Session viewer shows a `Settings` menu with cascaded submenus for approvals + sandbox.
- On session open, dropdown/menu labels update immediately based on the latest durable policy
  events (no “scroll to load older history” required).
- After restarting the desktop app / daemon, sending a message uses the session’s persisted
  approvals/sandbox policy (no unexpected auto-decline).
