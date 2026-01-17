---
id: T-3
epic: ui-v0
branch:
  suggested: rn/ui-v0/T-3-frontend-refactor
---

# T-3 Frontend refactor (v0 legibility + reuse)

## Problem

The dashboard UI has grown quickly with many “v0 dogfooding” iterations.
We now have large components and duplicated patterns:

- oversized views (especially `EpicView`/graph wiring) that are hard to reason about
- repeated “async action” patterns (pending state, error handling, retry, refresh)
- repeated layout/styling patterns (small cards, callouts, popovers, tooltips)

This slows iteration and makes it harder to:

- introduce TanStack Query cleanly
- keep UI behavior consistent
- add tests and maintain high confidence changes

## Goal

Refactor the frontend codebase to be:

- easier to navigate (smaller components, clear responsibilities)
- more consistent (shared primitives/patterns)
- less fragile (fewer “spooky” inter-component couplings)

This is not a “rewrite”; it is a careful decomposition.

## Scope

### 1) Decompose large components into cohesive modules

Target the biggest/most entangled components first (expected candidates):

- `dashboard/src/routes/EpicView.tsx`
- `dashboard/src/components/graph/GraphView.tsx`
- `dashboard/src/components/graph/NodeCard.tsx` (rename to `TaskCard.tsx` as part of this task)
- `dashboard/src/components/layout/DetailsPanel.tsx`

Approach:

- Extract domain-specific hooks (e.g. “selection model”, “graph data shaping”, “bulk actions”).
- Extract presentational components that are purely UI (no data fetching).
- Keep the public surface of the route small.

### 2) Standardize common “async action” patterns

Create reusable helpers/components for:

- pending/disabled reason conventions
- error capture + display (non-noisy, actionable)
- confirmation modals (shared “Proceed anyway?” patterns)

Important: do not over-abstract; the reuse should be obvious and readable.

### 3) Consolidate common styling/layout patterns

Where we repeat the same structures:

- “card with header + small badges + right actions”
- “callout panel with compact buttons” (use shadcn `Alert` as the base component)
- “popover menus with consistent spacing”

Factor out small primitives or shared `cn(...)` helpers so we aren’t manually re-tuning the same classes everywhere.

In particular, there are several high-ROI patterns worth factoring into primitives (or `cva` variants) to keep JSX focused on behavior rather than Tailwind plumbing:

- “glass surface” containers: `bg-*/80 + ring + shadow + backdrop-blur` combos (selection bar, popovers, floating actions)
- bordered button groups: wrapper + `rounded-none` buttons + `border-l` separators (action clusters in graph + details)
- chip/pill buttons: rounded-full bordered “pills” with consistent hover/focus and mono label styles (e.g. Linear pill)
- callout cards: tone mapping (icon + border/bg/ring + compact action row) consolidated via shadcn `Alert`
- icon-only action buttons: standardize `ghost` icon button chrome (kebab menus, expand/collapse, etc.)
- tooltip boilerplate: wrap the common `TooltipTrigger(render)` + class merge pattern so callsites stay readable

### 4) Prepare for state/query modernization

This task should make it easy to implement T-2 (TanStack Query) by:

- isolating data fetching into hook modules
- reducing “manual refresh” wiring complexity
- ensuring components can render well from cached/partial data

## Additional requirements (v0 polish + simplification)

These are concrete follow-ups discovered while dogfooding. They are explicitly in-scope for this refactor because they remove complexity and set stronger invariants for future work.

### A) React best practices (effect hygiene)

- Avoid `useEffect` for derived state, orchestration, or “keep X in sync with Y” patterns.
  - Prefer deriving state from props/query state via pure functions + `useMemo`.
  - Effects are acceptable for true subscriptions / imperative bridges (WebSocket wiring, timers, DOM integration), but they should be isolated in small hooks and kept rare in large view components.

### B) Remove incidental motion (reduce visual + code footprint)

- Remove the pulsing overlay around the agent status icon in the task card.
- Remove graph edge animations that fire when git operations succeed (the “edge pulse” system).
  - This feature currently spans `GraphView` + edge components + CSS keyframes; deleting it meaningfully reduces surface area.
- In general: avoid adding motion unless it encodes new information that cannot be conveyed with simpler status cues.

### C) Naming consistency

- Rename `NodeCard` → `TaskCard`.
  - “node” should mean “graph node” (layout/XYFlow concept); “task” should mean the domain object.
  - Keep naming consistent across file names, component names, and props.

## Concrete codebase-specific opportunities (reduce footprint)

These are specific, high-leverage cleanups that measurably shrink the amount of code touched per change.

### 1) Collapse duplicated “iterate over tasks” logic in `EpicView`

In `dashboard/src/routes/EpicView.tsx`, we currently compute:

- `runSummary`
- `actionTargets`
- `runBuckets`

Each of these loops over the tasks list with near-identical predicates and slightly different outputs. Create a single derived “run toolbar model” (counts, buckets, action targets, and disabled-reason helpers) and have the UI read from it.

### 2) Delete the edge pulse subsystem (if we keep the UI change above)

If we are removing success edge animations, delete the related code rather than leaving it dormant:

- `dashboard/src/components/graph/edgePulse.ts`
- `dashboard/src/components/graph/CommitStringEdge.tsx`
- `dashboard/src/components/graph/RoundedSmoothStepEdge.tsx`
- supporting logic in `dashboard/src/components/graph/GraphView.tsx`
- CSS keyframes/classes in `dashboard/src/index.css`

### 3) Standardize callouts into a single reusable primitive

We have multiple “callout-like” cards (graph card overlays and Details panel) that repeat:

- tone → border/bg/ring/icon mappings
- compact action rows
- “show details” affordances

Use shadcn’s `Alert` component as the base building block (and, if needed, wrap it in a thin `Callout` facade that only maps `tone` → `Alert` variants). Avoid a bespoke callout component that re-implements shadcn patterns.

## Shadcn-first guidance

When introducing or standardizing UI primitives, prefer shadcn components as the base wherever possible (e.g. `Alert`, `Button`, `Badge`, `Popover`, `DropdownMenu`, `Tooltip`). Only add bespoke components when the shadcn base is clearly insufficient.

### 4) Isolate “imperative / effectful” glue into hooks

Examples that should live outside large view components:

- sessionStorage state (cursors, bulk action state)
- timers/TTL bookkeeping for transient UI state
- WebSocket event plumbing

The route should read as mostly declarative UI + “compose hooks”.

### 5) Split `TaskCard` along stable boundaries

Even before T-2, split the card into smaller components:

- header (title + badges)
- right-side quick actions
- callouts/remediation content
- status iconography

This makes it easier to remove features (like motion overlays) and reduces the risk of incidental regressions.

### 6) Reduce bundle/startup footprint where possible

ELK is imported synchronously via `elkjs/lib/elk.bundled.js` (see `dashboard/src/components/graph/elkLayout.ts`). If startup cost becomes noticeable, consider lazy-loading ELK and/or moving layout to a worker to keep the initial chunk smaller.

### 7) Reduce API boilerplate before adding TanStack Query

`dashboard/src/api.ts` contains a lot of repeated fetch/error handling patterns. A small `requestJson` helper (method + path + body → typed result) would reduce code and make query/mutation functions smaller and more consistent once T-2 lands.

## Test/verification guidance

- Prefer small, pure-logic tests in `dashboard/tests/*` for:
  - selection/bucket computation
  - status derivations (daemon/merge run)
  - “run toolbar model” logic (once extracted)
- Keep UI tests minimal; aim for correctness by shrinking and purifying the logic layer.

## Non-goals

- Visual redesign.
- Introducing a heavy component library beyond what we already use.
- Changing graph layout algorithms.

## Acceptance Criteria

- The largest route/component files are meaningfully smaller, with responsibilities clearly separated.
- Shared UX patterns (errors, confirmations, action pending states) are consistent across surfaces.
- Adding a new UI feature requires touching fewer files and less duplicated code.
- No loss of functionality; existing `just check` remains green.
