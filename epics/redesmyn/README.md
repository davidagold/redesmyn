# Redesmyn Epic: Control Doc (Canonical)

This file is the canonical “control doc” for the **Redesmyn** epic: intent, v0 spec, invariants, and key architectural decisions. Keep it current.

## 1) Vision

Build a local-first “cockpit” for orchestrating multi-agent work on a git repository via a **branch graph** where:

- Each branch node is mapped to an agent.
- Agents work independently on their assigned branches (typically in separate worktrees).
- Cross-branch operations (rebases, moving commits) are executed by a local daemon that enforces invariants.
- A dynamic web UI visualizes the evolving commit/branch topology, agent activity, and review feedback in near real-time.

Unlike Graphite-style “one commit per branch” stacks, Redesmyn’s goal is to support **stacked PRs with linear, multi-commit work ranges** (per node) while retaining safe rebase/move workflows.

## 2) Principles

- **Local-first**: Optimize for a single developer’s machine and repo before multi-user collaboration.
- **Git is the substrate for history**: We observe and manipulate real git history; we do not invent a parallel VCS.
- **GitHub + Linear are first-class for planning**: Issues/PRs/comments are the substrate for planning, discussion, and progress tracking.
- **Auditable state**: Persist an append-only event log so rebases and “what happened?” questions are answerable.
- **Agent-agnostic**: Agents are external processes; integration is primarily via a CLI contract, not deep harness coupling.
- **Graph-first UX**: The UI is a modern, beautiful, highly functional graph experience (not a wall of tables).

## 3) Dogfooding

We will use Redesmyn to build Redesmyn.

## 4) Glossary

- **Repository**: A git repository on disk.
- **Epic**: A collection of tasks for a specific objective within a repository.
- **Task**: A unit of planned work (typically backed by a Linear issue; optionally a GitHub issue) that can be linked to a node.
- **Node / Branch Node**: A branch in the epic’s branch graph with a single parent (except the root).
- **Branch Graph**: The topology of nodes for an epic. It is a tree (one parent per node); visually a DAG when including commit ancestry.
- **Stack**: A path through the branch graph from an upstream node to a connected leaf node. Stacks can overlap (shared prefix) and are primarily a focus/view concept.
- **Agent**: An external worker (Codex, Claude Code, etc.) assigned to a node.
- **Daemon**: The local long-running orchestrator managing state, locks, commands, and integrations.
- **`rn` CLI**: The user/agent-facing CLI. Agents are instructed to funnel git actions through `rn`, which proxies `git` while enforcing invariants.
- **Work Range**: The commit range representing a node’s “work”: `parent..branch` (multi-commit allowed).
- **Event Log**: Append-only record of domain events (git, agent, integration, orchestration).
- **Projection**: A materialized view computed from the event log for fast UI queries.

## 5) v0 Scope (MVP)

### 5.0 Tasks (tickets) and planning

An epic is a **collection of tasks** (tickets) plus the branch graph that implements them. Redesmyn treats GitHub + Linear as first-class planning/discussion substrates and provides two common starting points:

- **Markdown-first**: create local Markdown “task specs” (e.g. under `epics/<slug>/tasks/`) and have `rn` generate/sync the corresponding Linear tickets (and optionally GitHub issues).
- **Linear-first**: start from an existing set of Linear tickets (query/label/milestone) and have `rn` import them into the local database, **auto-creating nodes/branches and linking tasks by default** (and optionally generating local task specs as an offline cache).

Each node typically links to a **primary task** (usually a Linear issue) so the UI can join “code progress” (git/PR) with “planning progress” (issue state, comments, discussion). v0 should support linking at minimum; richer bi-directional updates are roadmap.

#### Linear mapping (v0)

- **Epic == Linear Project** (the epic stores a `linearProjectId`).
- **Tasks == Linear Issues** in that project.
- Linear **parent/sub-issue** structure is primarily used for **grouping and focus** in the UI, not for determining branch topology.

#### Topology inference from Linear (v0)

Branch topology must be a **tree** (one parent per node), but Linear dependencies are a general graph. For v0 we choose a simple, explicit mapping:

- Use Linear **blocked-by** as “must land before” (a merge-order constraint).
  - Blocked tasks can still be started in parallel (stacked on the blocker branch), but their PRs cannot be merged until blockers merge, and they may require rebases as blockers evolve.
- Inference rules:
  - `blocked_by == 0` → node parent is the epic root branch (typically the repo default branch).
  - `blocked_by == 1` → node parent is the blocker’s node (PR base = blocker branch).
  - `blocked_by > 1` → **error in v0** (user must resolve by choosing a single primary blocker; interactive selection is deferred).
  - Cycles/self-dependencies → error.

#### Branch naming from Linear (v0)

- Default branch naming includes the Linear issue identifier:
  - `rn/<epicSlug>/<LINEAR-123>-<short-slug>`
- Branch names should be treated as stable identifiers once created (do not auto-rename on title edits in v0).

#### v0 task spec format (Markdown-first)

- One file per task under `epics/<slug>/tasks/`.
- File contains a small YAML frontmatter block (written/updated by `rn`) plus a Markdown body:
  - `rn_task_id`: stable local identifier (UUID)
  - `title` (informational when an external source is authoritative)
  - `authority`: `linear | local | github` (v0 supports `linear` and `local`; `github` is a seam)
  - `linear_issue_id` (optional; filled after first sync/push)
  - `github_issue` (optional; seam for later)
  - `node_branch` (optional; filled when linked)
  - Body conventions (v0):
    - `rn` may manage a “synced” section populated from Linear (or other providers).
    - Users may keep a “local notes / agent brief” section that `rn` never overwrites (so “Linear-authoritative” doesn’t destroy local instructions).

#### Authority model (v0)

Default stance: **Linear is authoritative** for task metadata and discussion, because it is where planning happens and where teams collaborate.

- When `authority: linear`, `rn task pull` updates local task specs from Linear; local edits do not overwrite Linear unless explicitly pushed/forced.
- When `authority: local`, `rn task push` updates/creates Linear issues from local specs (useful for “local-first spec writing” or offline workflows).
- Allow per-epic defaults with per-task overrides (design seam; exact config format is an implementation detail).

### 5.1 Must-have user flows

1. Initialize orchestration for a repo + epic; start daemon.
2. Bootstrap a backlog of epic tasks (Markdown-first or Linear-first) and link them to nodes.
3. Create an epic branch graph and assign agents to nodes.
4. Provide per-node worktrees so agents don’t fight over working directories.
5. Proxy git operations through `rn` and enforce key invariants.
6. Observe git activity and update agent/node “progress” automatically.
7. Execute daemon-managed operations:
   - Rebase a node onto its parent.
   - Cascade rebase a subtree when a parent advances.
   - Move/split a contiguous commit range from one node to another (with synchronization).
8. Web UI that renders (graph-first):
   - Branch graph (nodes) with agent status
   - Commit graph per node (work range) and lineage across rebases
   - Timeline of events (git + orchestration)
   - Messages/commands panel
9. GitHub + Linear integrations (v0): polling ingest of comments + basic linking to nodes.
10. “Select review comments → send to agent” workflow (initially manual selection in UI).

### 5.2 Explicit non-goals for v0

- Multi-user shared orchestration state.
- Enforcing invariants against arbitrary raw `git` usage (we detect/repair, but cannot prevent).
- Fully automated doc synchronization across all providers.
- Webhook-driven integrations (polling is fine for v0).

## 6) System Architecture (Conceptual)

### 6.1 Components

- **Daemon**
  - Owns authoritative state (epics, nodes, agents, locks, commands).
  - Runs background jobs (repo observer, integration polling, projection updates).
  - Exposes a local API for CLI + Web UI.

- **Git Service**
  - A library/service used by daemon + CLI that performs safe operations:
    - create/move branches
    - rebase node onto parent
    - cascade rebase subtree
    - move/split commits between nodes
  - Uses worktrees where possible.

- **Repo Observer**
  - Watches for ref movements, new commits, and worktree status changes.
  - Converts raw git observations into semantic domain events.
  - Must be robust if changes occur outside `rn`.

- **`rn` CLI**
  - Human interface and agent interface.
  - Used for git proxying, synchronization, messaging, and integration control.

- **Web UI**
  - Uses projections for queries; subscribes to an event stream for live updates.

- **Integrations**
  - GitHub + Linear adapters that poll and emit normalized events.

### 6.2 Local API surfaces (daemon)

The daemon exposes:

- Query endpoints for projections (graph, nodes, agents, timelines, comments).
- Command endpoints (issue command, acknowledge barrier, lock operations).
- Event stream endpoint (SSE/WebSocket) for UI.

(Exact protocol is a technology decision; the contract is what matters.)

## 7) Invariants (Hard Rules)

These are enforced by the daemon and by `rn` when possible:

1. **One parent per node**: Branch topology is a tree.
2. **One agent per node (v0)**: At most one active agent assignment per node.
3. **Agents commit only on their assigned node branch** (best-effort enforcement via `rn` + worktrees).
4. **Rebases/moves are daemon-only**: cross-branch history rewriting is executed by the daemon, not by agents.
5. **Work Range is contiguous**: A node’s changes are represented by the commit range `parent..branch`.
6. **Prefer linear history**: No merge commits in work ranges (daemon rejects operations that would introduce merges; `rn` warns on detect).

## 8) State Model

### 8.1 Core entities

- **Repository**
  - `repoPath`
  - `defaultBranch` (upstream/root, e.g. `main`)
  - `createdAt`

- **Epic**
  - `epicId`
  - `name`, `slug`
  - `rootBranch` (usually `defaultBranch`)
  - `linearProjectId | null`
  - `createdAt`
  - `docRefs`: canonical control doc + external references (optional)

- **Task**
  - `taskId`
  - `epicId`
  - `title`, `body`
  - `source`: `local | linear | github`
  - `authority`: `linear | local | github` (v0 supports `linear` and `local`)
  - `refs`: `{ linearIssueId?, githubIssueId?, localPath? }`
  - `state` (todo/in_progress/done/blocked; provider-specific mapping)
  - `nodeId | null` (the primary node for this task, if assigned)

- **Node**
  - `nodeId`
  - `epicId`
  - `branchName`
  - `parentNodeId | null`
  - `childNodeIds[]`
  - `agentId | null`
  - `worktreePath | null`
  - `primaryTaskId | null`
  - `links`: `{ githubPrId?, linearIssueId? }`
  - `policy`: (e.g., merge policy, allowed commands)

- **Agent**
  - `agentId`
  - `displayName`
  - `capabilities` (optional; e.g., “can run tests”, “can open PRs”)
  - `lastSeenAt`
  - `status` (idle/running/blocked/error)

- **Command**
  - `commandId`
  - `target`: `agentId` and/or `nodeId`
  - `type` (e.g., address_comments, implement_task, rebase_required)
  - `payload` (structured)
  - `state` (queued/running/succeeded/failed/canceled)
  - `createdAt/updatedAt`

- **Barrier / Sync Point**
  - `barrierId`
  - `scope` (node/subtree)
  - `requiredAcks` (agents/nodes)
  - `mode`: `loose | tight`
  - `state` (open/fulfilled/expired/canceled)

### 8.2 Persistence approach (v0)

- **Local embedded DB** for event log + projections + integration cursors + secrets references.
- Do not commit orchestration DB into git.
- Provide `rn export` / `rn import` for snapshot portability (JSON).

## 9) Event Log (Append-only)

### 9.1 Canonical event envelope

Every event has:

- `eventId`, `ts`
- `repoId`, optional `epicId`
- `actor` (user/agent/daemon/integration)
- `type`
- `payload` (type-specific)
- `correlationId` (ties together multi-step operations like cascade rebases)

### 9.2 Key event types (v0)

**Topology / orchestration**

- `epic.created`, `epic.renamed`
- `task.created`, `task.updated`, `task.linked`, `task.synced`
- `node.created`, `node.parent_set`, `node.deleted`
- `agent.created`, `agent.assigned`, `agent.unassigned`, `agent.heartbeat`
- `command.issued`, `command.state_changed`
- `barrier.created`, `barrier.acked`, `barrier.fulfilled`, `barrier.expired`
- `pause.set`, `pause.cleared`
- `message.sent` (agent↔agent, user↔agent; all via daemon)

**Git semantics**

- `git.ref_moved` (raw observation)
- `git.commit_created` (detected new commit on a branch)
- `git.rebase_started`, `git.rebase_completed`, `git.rebase_conflict`
- `git.range_moved` (commit range migrated node→node)
- `git.worktree_created`, `git.worktree_removed`

**Integrations**

- `github.pr_linked`, `github.comment_received`, `github.review_received`
- `linear.issue_linked`, `linear.comment_received`

### 9.3 Rebase lineage (critical)

For each history rewrite, persist a mapping allowing “total history across rebases”:

- old base/head, new base/head
- per-commit mapping where possible (via patch-id or equivalent content fingerprint)

This enables UI to render “ancestral chains” even when SHAs change.

## 10) Git + Worktrees

### 10.1 Worktree strategy (recommended)

- One worktree per node branch:
  - Prevents agents from stomping on each other’s working directories.
  - Makes per-node status and uncommitted changes observable and attributable.

### 10.2 `rn git` proxy rules (v0)

`rn git …` runs `git …` faithfully except when it would violate invariants. Example policy:

- Allow: `status`, `diff`, `add`, `commit`, `log`, `fetch`, `pull` (with constraints)
- Disallow or redirect to daemon:
  - `rebase`, `merge`, `reset --hard`, `push --force` (unless daemon-approved)
  - `checkout` to a branch that is not the agent’s assigned branch in that worktree

When disallowing, `rn` provides a remediation:

- “Use `rn rebase <node>`”
- “Use `rn move-range …`”
- “Request a barrier with `rn sync tight …`”

#### `rn pause` gating (v0)

`rn pause` is a user/agent-facing control that blocks **mutating** operations on a node or subtree (read-only operations always work). It is meant to let humans and agents coordinate “don’t change this while we decide” moments without relying on long-running/hanging commands.

- Modes:
  - `lax`: allow local progress (e.g. `add`/`commit`), but block push + rewrites/cross-branch operations.
  - `strict`: block all mutating operations in-scope.
- Additional rule: `fetch` is allowed even during `strict` (it mutates `.git` but not the working tree/branch state).
- When an operation is blocked by a pause, `rn` returns immediately with a non-zero exit code and a clear message (no hanging/waiting in v0).

First-pass mutating command classification (v0):

- Always allowed (read-only): `status`, `diff`, `log`, `show`, `blame`, `grep`, `ls-files`, `rev-parse`, `cat-file`.
- Allowed even in `strict`: `fetch`.
- `lax` allows:
  - `add`, `restore` (incl `--staged`), `rm`, `mv`
  - `commit` (but not `--amend`), `revert`, `apply`
- `lax` blocks (and `strict` blocks everything mutating):
  - Any push: `push`
  - Rewrites/surgery: `rebase`, `reset` (all modes), `commit --amend`, `cherry-pick`, `merge`
  - Ref edits: branch/tag create/delete/force
  - Worktree management: `worktree add/remove` (must go through `rn`)

### 10.3 Daemon-managed operations (v0)

- `rebase-node(nodeId)` onto parent
- `cascade-rebase(subtreeRootId)`
- `move-range(fromNodeId, toNodeId, commitRangeSpec)`

All of the above require appropriate locks, may create barriers, and must respect `rn pause` state (daemon refuses to rewrite/move history in paused scopes).

## 11) Synchronization Model (Loose vs Tight)

- **Loose sync**
  - Daemon notifies relevant agents and proceeds after either acknowledgements or a timeout.
  - Used when risk is low (e.g., non-overlapping work, idle agents).

- **Tight sync**
  - Daemon requires explicit acknowledgements before proceeding.
  - Used for commit-range moves, subtree rebases, or anything that risks conflicting local work.

Agents acknowledge via `rn barrier ack <id>`.

## 12) CLI (v0 contract)

This is the stable surface for both humans and agents.

### 12.1 Repo + epic lifecycle

- `rn init` (init orchestration for the repo)
- `rn epic create --name <name>` (v0 may default to a single epic)
- `rn epic list`
- `rn daemon start|stop|status`
- `rn status` (summary)

### 12.2 Tasks (tickets)

- `rn task list [--epic <epic>]`
- `rn task add --title <title> [--epic <epic>]` (creates a local task spec)
- `rn task link <task> --node <node>`
- `rn task show <task>`
- `rn task pull [--epic <epic>]` (refresh local task cache from Linear)
- `rn task push [--epic <epic>]` (create/update Linear issues from local tasks when `authority: local`)

### 12.3 Topology

- `rn node create --branch <name> --parent <nodeOrBranch> [--epic <epic>]`
- `rn node list [--epic <epic>]`
- `rn node show <node>`
- `rn node set-parent <node> --parent <node>`
- `rn stack show <leafNode>` (render the path upstream)

### 12.4 Agents

- `rn agent register --name <displayName>`
- `rn agent assign <agent> --node <node>`
- `rn agent heartbeat` (optional; best-effort liveness signal)

### 12.5 Git operations (safe)

- `rn git <args…>` (proxy)
- `rn rebase <node>` (daemon-managed)
- `rn cascade <node>` (daemon-managed)
- `rn move-range --from <node> --to <node> --range <spec>` (daemon-managed)

### 12.6 Messages + synchronization + commands

- `rn msg send --to <agent|node> --text <…>`
- `rn inbox` / `rn outbox` (names TBD)
- `rn sync loose|tight --scope <node|subtree> [--reason <…>]`
- `rn barrier ack <id>`
- `rn pause lax|strict --scope <node|subtree> [--reason <…>]`
- `rn pause clear --scope <node|subtree>`
- `rn command issue …` (structured commands; UI can generate these too)

### 12.7 Integrations

- `rn gh status|sync`
- `rn linear status|sync`
- `rn linear import --project <id|name> [--epic <epic>] [--create-nodes | --no-create-nodes]` (default: create nodes/branches and link tasks; errors on tasks with multiple blockers)

## 13) Web UI (v0)

### 13.1 Required screens

- **Epic overview (graph-first)**
  - Branch graph with agent status and PR/issue links.
  - “Needs cascade rebase” indicators for subtrees.
  - Stack focus mode (select a leaf → highlight the upstream path).

- **Node detail**
  - Work range commit list/graph.
  - Rebase lineage view (“history across rebases”).
  - Worktree status (dirty/clean, conflicts).
  - Commands/messages for the node’s agent.

- **Timeline**
  - Stream of events (git + orchestration + integrations).

- **Review triage**
  - List PR comments/reviews.
  - Select subset → generate command/message to assigned agent.

### 13.2 Realtime updates

- UI subscribes to daemon event stream; projections update incrementally.

## 14) Canonical Docs (v0 + roadmap)

We want a notion of a single canonical “control doc” per epic, with the ability to point to external providers (Notion, Google Docs, GitHub issue, Linear issue/doc).

### 14.1 v0 stance

- Canonical control doc lives as local Markdown inside the repo (`epics/<slug>/README.md`).
- Prefer git history + PRs + Linear/GitHub metadata as the changelog in v0 (no separate changelog file).
- The system may store an external reference, but does not guarantee multi-provider synchronization in v0.

### 14.2 Roadmap: connector interface

Define a provider-agnostic interface:

- `pull(ref) -> normalizedDocument`
- `push(ref, normalizedDocument) -> providerRevision`
- `diff(providerRevisionA, providerRevisionB)`

Start with one-way sync (canonical → mirrors) and require explicit user approval for overwriting remote edits.

## 15) Security / Secrets

- Store integration tokens locally (never in git).
- Scope tokens minimally; rotate easily.
- UI should never display tokens; only “connected/disconnected”.

## 16) Roadmap Beyond v0 (high-level)

- Multi-epic concurrency UX (multiple branch graphs in one repo)
- Webhooks for GH/Linear
- Richer agent protocols (harness hooks, structured task schemas)
- Conflict resolution UX for cascade rebases
- “Suggested refactors” like auto-splitting commits per node

## 17) Open Questions (Decisions Needed)

- How strict is “no merge commits” (hard reject vs warn)?
- What is the canonical unit of “progress” (commits, diff size, CI state, PR review state)?
- How do we represent agent “work in progress” when changes are uncommitted?
- How should `move-range` be specified (commit SHAs, counts, ranges, patch-id selectors)?
- `rn pause` command coverage: allow or block `stash`, `pull`, and `switch/checkout` in `lax`?
- Branch naming details: exact format and whether/when renames are supported.
