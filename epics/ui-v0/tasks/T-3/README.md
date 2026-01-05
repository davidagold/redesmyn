# T-3 Frontend refactor (v0 legibility + reuse)

## Metadata

```yaml
id: T-3
epic: ui-v0
branch:
  suggested: rn/ui-v0/T-3-frontend-refactor
```

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
- “callout panel with compact buttons”
- “popover menus with consistent spacing”

Factor out small primitives or shared `cn(...)` helpers so we aren’t manually re-tuning the same classes everywhere.

### 4) Prepare for state/query modernization

This task should make it easy to implement T-2 (TanStack Query) by:

- isolating data fetching into hook modules
- reducing “manual refresh” wiring complexity
- ensuring components can render well from cached/partial data

## Non-goals

- Visual redesign.
- Introducing a heavy component library beyond what we already use.
- Changing graph layout algorithms.

## Acceptance Criteria

- The largest route/component files are meaningfully smaller, with responsibilities clearly separated.
- Shared UX patterns (errors, confirmations, action pending states) are consistent across surfaces.
- Adding a new UI feature requires touching fewer files and less duplicated code.
- No loss of functionality; existing `just check` remains green.
