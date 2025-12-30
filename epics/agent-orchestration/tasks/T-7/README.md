# T-7 Dashboard: agent presence + activity integrated into the graph

## Metadata

```yaml
id: T-7
stacked_on: T-6
node:
  branch: rn/agent-orchestration/T-7-dashboard-graph-presence
```

## Brief (local)

- Render agent presence directly on graph nodes:
  - not-started / running / stopped / failed / blocked
  - recent activity indicators (commit/worktree pulse)
- Consume WebSocket updates and reflect them in the graph UI in a tasteful way.

## Acceptance Criteria

- Nodes visually communicate “is someone working on this, and are they healthy?” at-a-glance.
- Activity indicators do not add noisy borders or busy UI; use subtle emphasis consistent with existing styling.

## Updates

- Graph nodes render agent presence (status dot + state) and subtle activity pulses from streamed events.
- Node cards should be task-first (no “Node <id>” in the UI): remove node ids from the card UI as nodes are not user-facing.
- Adjust node card layout: top-align text content, avoid concatenated ID/title repeats, and prefer spacing over extra borders.
- Use a single status affordance: a colored status circle in the upper-right with a tooltip (status + short health summary).
- Surface the task’s agent as a “resource tag” aligned to the bottom of the card (small border radius; distinct from regular tags).

### Final designs

#### A) Task-first card content

- Primary text: task identifier + title (no node ids).
- Secondary text (optional, subtle): short “stack position” hint only if it materially helps (e.g. parent title on hover), otherwise keep cards sparse.

#### B) Single status circle (top-right)

One circle communicates overall “should I worry?”:

- **Green (filled)**: agent running
- **Blue (outline)**: agent stopped
- **Blue (dashed outline)**: no agent (not started yet)
- **Amber**: blocked (task state blocked) or stopping/starting transient
- **Red**: failed (agent error/exit) or worktree missing/mismatch

Tooltip (compact, actionable; avoid inert enumerations):

- `Agent`: `a-<task_id>`
- `Status`: running/stopped/blocked/error (+ elapsed since start if available)
- `Worktree`: clean/dirty/missing/mismatch (only when not healthy)
- `Last activity`: last commit subject/sha prefix if available

#### C) Agent “resource tag” (bottom)

- When the task agent exists, show a pill like `a-123` at the bottom edge.
- The pill is a visual affordance for “this task has an agent identity”, not a property list.

#### D) Activity pulses

- A subtle ring/pulse on the card (or status circle) when a new `git.commit` event lands for the task/node.
- Pulse timing is short-lived and rate-limited (avoid noisy flapping during rapid commits).
