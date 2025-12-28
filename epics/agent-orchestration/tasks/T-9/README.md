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

### Final designs

#### A) Details panel: Worktree section (health-first)

Show only what matters for action:

- **Health line** (single sentence): `Missing`, `Clean`, `Dirty`, or `Branch mismatch`
- Optional secondary: current branch only when mismatch is present

Avoid showing raw filesystem paths by default.

#### B) Worktree actions (small, ergonomic set)

Primary action:

- `Copy rn checkout --task <task_id>` (this is the canonical “take me there / create it” workflow)

Secondary actions (copy-first):

- `Copy path` (only if known)
- `Copy cd command` (only if known)
- Optional: “Open in Finder” / “Reveal” (if we decide to support it; keep out of the critical path)

#### C) Cloud correctness constraint

- Treat worktree paths as host-local strings.
- The UI must never imply the server can access the path; all actions should be framed as local `rn` commands.
