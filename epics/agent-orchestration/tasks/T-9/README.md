# T-9 Worktrees UX: `rn checkout` + worktree status surfaces

## Metadata

```yaml
id: T-9
stacked_on: T-4
must_land_after:
  - T-2
node:
  branch: rn/agent-orchestration/T-9-worktree-ux
```

## Brief (local)

- Provide ergonomic worktree-aware navigation:
  - `rn checkout <node|task|branch>` prints/opens the correct worktree and ensures it exists
  - optional “subshell” helper for fast switching (within constraints of terminal UX)
- Expose worktree health to the UI (host-local path, clean/dirty, branch):
  - show from the graph selection state (Details panel) without devolving into inert property enumerations

## Acceptance Criteria

- Users can move between worktrees as easily as “checking out a branch”, without violating the one-worktree-per-node invariant.
- The system remains conceptually correct for cloud deployment (worktrees are host-local; UI does not assume server filesystem access).

## Updates

- Worktree UI surfaces should stay “health-first”: missing/dirty/mismatch signals + a small set of actions, with full paths behind copy/expand (avoid inert enumerations).
- Align the Details panel Worktree section with `rn checkout` ergonomics (copy checkout command, open path, etc.) rather than expecting users to manually navigate long filesystem paths.
