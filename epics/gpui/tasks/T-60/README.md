---
epic: gpui
branch:
  suggested: rn/gpui/T-60-session-markdown-renderer
rn:
  parent: T-59
  after:
    - T-44
---

# T-60 Full Markdown rendering for session messages (GPUI) (Domain 7)

## Problem

The session viewer’s primary content is agent/user messages, and we explicitly want **full markdown** rendering.

If we treat messages as plain text, we lose:

- structure (headings, lists, code blocks),
- link affordances,
- and readability for long agent outputs.

If we implement markdown in an ad-hoc way, we risk:

- poor performance (re-layout on every event),
- inconsistent styling across the app,
- and fragile rendering when we later introduce diff/code visualization.

## Goal

Implement a reusable, performant **MarkdownView** that can render full markdown for session messages inside GPUI, with consistent styling and a clear path to embedding richer “artifact views” later (diff viewer, file snapshots, etc.).

## Requirements

### 1) Markdown feature set (v1)

Support (at minimum):

- paragraphs + soft/hard breaks
- headings (H1–H6)
- emphasis + strong
- inline code + fenced code blocks (triple backticks)
- block quotes
- ordered/unordered lists (nested)
- links (clickable; opens via OS/browser)

Non-goals for the port (can be future work):

- tables, footnotes, task lists
- HTML passthrough
- syntax highlighting (optional; can be added later)

### 2) Rendering architecture (performance + maintainability)

Design for:

- **incremental** updates (new messages append without re-rendering the whole feed),
- caching parsed markdown per message event id,
- minimal layout thrash (avoid rebuilding complex view trees on every frame),
- predictable memory bounds (cap extremely large markdown payloads; redirect to artifacts per T-14).

Suggested approach:

- Parse markdown into an intermediate typed structure (`MarkdownDoc` → blocks → inline spans).
- Render blocks to GPUI elements with stable keys for diffing.
- For large code blocks, render as a specialized `CodeBlockView` (monospace, background, copy action).

### 3) Styling/tokens

Follow UI foundations (T-44):

- typography scale (heading sizes, body, monospace),
- color tokens for subtle separators (avoid busy borders),
- link colors that are visible in light/dark,
- selection/copy affordances.

### 4) Accessibility

- Links must be keyboard reachable.
- Provide stable labels for important controls (e.g. “Copy code block”).

### 5) Integration points

The markdown renderer must be usable for:

- assistant messages,
- user messages (treat as markdown too; users often paste code/logs),
- future “artifact previews” embedded inline (Domain 8).

## Acceptance criteria

- `MarkdownView` renders the v1 feature set with acceptable performance for long messages.
- `SessionView` can render a realistic Codex response (multiple headings + lists + fenced code blocks) without noticeable jank.
- A small unit test suite validates markdown parsing → block model (no GPUI required).

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on UI foundations (T-44).
- Integrates with session feed scaffolding (T-59) and virtualized UI (T-61).

## Reference implementation (today; for behavior orientation only)

- We do not have a full chat history view in the web UI today; this is new.
- Markdown-y outputs today are typically visible only as:
  - short previews on task cards (`dashboard/src/components/graph/TaskCard.tsx`),
  - or in external tools (tmux/terminal).
