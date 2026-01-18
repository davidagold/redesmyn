---
epic: gpui
branch:
  suggested: rn/gpui/T-61-session-view-virtual-list
rn:
  parent: T-59
  after:
    - T-44
---

# T-61 SessionView virtualized feed + scroll behaviors (Domain 7)

## Problem

Session histories can be long (many messages, tool events, and status updates). A naïve “render everything” approach will:

- stutter on scroll,
- consume too much memory,
- and make the desktop app feel slow.

We also have strict UX requirements:

- no spinner wheels,
- no silent actions,
- and stable scrolling (no jumping) when new events arrive or older history loads.

## Goal

Implement the core GPUI `SessionView` feed UI:

- virtualized rendering (only render visible items),
- stable scroll anchoring for prepend/append,
- “new messages” indicator when not at bottom,
- and visible in-flight affordances for loading and sending.

This ticket focuses on the list/scroll mechanics; markdown rendering and composer behavior are separate tickets.

## Requirements

### 1) Virtualization strategy

Implement a virtual list for session timeline items:

- only render items in/near the viewport (configurable overscan),
- maintain accurate scroll height via measured row heights,
- cache row measurements and invalidate on width/theme changes.

### 2) Scroll anchoring rules (must be explicit)

- Append (live events):
  - if user is “at bottom”: auto-scroll to bottom
  - else: preserve scroll position and show a “new messages” indicator
- Prepend (load older):
  - preserve the user’s current top-visible anchor row
  - avoid any perceptible jump

### 3) Progress affordances (no spinner wheels)

Use calm, visible indicators:

- “Loading older…” inline row with subtle pulse/ellipsis
- “Syncing…” status in header (optional)
- disable “Load older” button while in flight

### 4) Empty/error states

- Empty session: show helpful prompt (“No messages yet”) and composer if applicable.
- Error while loading: show actionable error + retry; do not drop drafts.

### 5) Keyboard + selection basics

At minimum:

- page up/down scroll works,
- focus can move to composer,
- links/buttons in visible rows are keyboard reachable.

## Acceptance criteria

- Scrolling remains smooth with thousands of timeline items (mocked data).
- Prepending older history does not jump the viewport.
- When user is not at bottom, new events do not steal scroll; “new messages” indicator appears and scroll-to-bottom works.

## Dependencies / sequencing

- Depends on UI foundations (T-44) and the session view-model/cursor semantics (T-59).
- Markdown rendering (T-60) and composer behavior (T-62) can be integrated incrementally.
