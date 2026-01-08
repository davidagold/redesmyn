# T-9 Repo instances + canonical executor routing (migration)

## Metadata

```yaml
id: T-9
stacked_on: T-2
node:
  branch: rn/revise-architecture/T-9-repo-instances-canonical-executor
linear:
  issue_id: 9d96c5f2-f3d4-434e-8a7d-8568b5c64e26
  identifier: RED-38
```

## Context / Motivation

The revised architecture separates:

- the **control plane** (server): state + APIs + UI projection, and
- a **repo executor** (v1: a host-local daemon): the component with filesystem access that can execute git/worktree actions.

Earlier work (T-1/T-2/T-5/T-6/T-7/T-8) focused on removing the assumption that the control plane shares a filesystem with the daemon.
However, it still implicitly assumed a single “active repo executor” per repo, and did not fully specify how the system behaves when there
are multiple potential executors for the same logical repo (multiple client daemons, server-managed repo instances, failover/scale-up, etc.).

This task formalizes and implements the missing nuance:

- the system must support **multiple repo instances** of the same logical repo, and
- repo-scoped “canonical” mutations must be single-writer to avoid correctness hazards.

### Target use cases

- **Local-first co-located mode**: user runs the control plane locally; server host and daemon host are identical; there is only one repo instance.
- **Cloud UI with server-managed repo instance**: a hosted control plane may also manage its own repo instance (canonical executor) and run merge/restack there.
- **Remote executor**: a user uses a local or cloud UI to orchestrate agent runners on a remote compute host that has its own repo instance.
- **Multi-host attachment**: two daemons attach to the same logical repo simultaneously (e.g. two laptops), and the system must avoid issuing canonical mutations on both.

## Model refinement (what changes)

Keep the **logical repo** as the shared identity for desired state, event log attribution, and UI navigation:

- logical repo identity: `(workspace_id, repo_id)`

Introduce a **repo instance** identity as the unit of git/worktree execution:

- repo instance identity: `(workspace_id, repo_id, host_key)`

The host identity (`host_key`) comes from the daemon handshake and already exists as the stable identity for a connected executor.

### Command targeting modes

Executor commands should fall into two categories:

1) **Instance-scoped commands**
   - Target a specific repo instance by explicitly specifying `host_key`.
   - Examples: “ensure worktree exists on host X”, “start a local agent session on host X”.

2) **Repo-scoped canonical commands**
   - Target the logical repo and are intended to mutate the “canonical” repo state (merge/restack onto base, ref movements treated as authoritative).
   - Must route to a **single writer** (primary executor) at any given time to avoid concurrent canonical mutations.

Single-writer can be implemented as a lease/primary executor mechanism (exact naming is flexible), but the invariant is the important part:

- for a given `(workspace_id, repo_id)`, at most one `host_key` should receive repo-scoped canonical mutation intents at a time.

## Work (detailed)

### 1) Attribution: make repo instance identity first-class

Ensure git-derived telemetry/projections and merge/restack progress can be attributed to a specific repo instance:

- Update event/projection payloads emitted by executors to include `host_key` in addition to `(workspace_id, repo_id)`.
- Update persistence/projections on the control plane side so “which host produced this?” is recoverable.
- Decide which projections are “authoritative” by default (typically the primary executor’s).

### 2) Canonical executor routing for git mutations (merge/restack)

Revise existing merge/restack orchestration so the control plane does not execute git steps, and canonical intents target a single executor:

- API surface:
  - For git mutations like merge/restack, accept either:
    - a repo-scoped request (no `host_key`, route via primary executor), or
    - an explicitly targeted request (explicit `host_key`) for advanced/dev workflows.
- Routing:
  - If routing via primary executor: resolve which `host_key` is primary for `(workspace_id, repo_id)`; if none, return actionable guidance.
  - If explicitly targeted: allow the call, but treat results/projections as instance-scoped and do not silently reinterpret them as canonical unless the target is primary.
- Enforcement:
  - The control plane must not deliver repo-scoped canonical mutation commands to non-primary executors.
  - The executor should reject canonical mutation commands when it is not the primary (defense in depth).

### 3) Dashboard semantics (primary executor + guidance)

Update the dashboard behavior to match the refined model:

- Surface which `host_key` is currently the primary executor for the repo (when meaningful).
- Disable canonical git actions when no primary executor is available and provide guidance (“start daemon”, “attach repo”, “acquire primary executor”).
- Default UI projections (trunk timeline, out-of-sync indicators, etc.) to those produced by the primary executor; optionally expose non-primary views as debug-only.

### 4) Local-first ergonomics (optional follow-on)

In co-located dev/local-first mode, avoid a “WS hop to self” by keeping a stable interface and swapping implementations:

- Define a `RepoExecutor` interface used by control-plane routes.
- Provide:
  - `DaemonRepoExecutor`: routes commands to a remote daemon over WS.
  - `LocalRepoExecutor`: executes the same intents in-process against the local checkout.

This preserves a single architectural surface while keeping local dev fast.

## Coordination notes

- T-3 (daemon execution) should implement the repo instance attribution and canonical mutation eligibility checks described here.
  If T-3 is not merged yet, incorporate these semantics directly into its implementation.
- Existing merged work in T-2/T-5/T-7 may require revisions to:
  - store `host_key` attribution on persisted projections/merge runs, and
  - route merge/restack intents through the executor interface instead of calling git directly from the control plane.

## Acceptance Criteria

- Repo instance identity `(workspace_id, repo_id, host_key)` is represented end-to-end for executor-sourced telemetry/projections and merge/restack runs.
- The control plane can safely operate with multiple connected executors for the same logical repo without issuing concurrent canonical mutations.
- Merge/restack (and resume) are routed to a single canonical executor by default, with clear UI guidance when no primary executor is available.
