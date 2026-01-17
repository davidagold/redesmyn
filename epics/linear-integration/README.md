---
rn:
  slug: linear-integration
  name: Linear Integration
  root_branch: main
  linear:
    project_id: 0c41f23e-219c-4feb-908b-3dff3cb7e906
---

# Linear Integration Epic: Control Doc (Canonical)

This file is the canonical “control doc” for the **Linear Integration** epic: intent, v0 spec, invariants, and key decisions. Keep it current.

## 1) Vision

Make Linear a first-class planning substrate by supporting **explicit sync** between:

- **Local-first task docs + DB tasks** (canonical for topology + agent briefs), and
- **Linear issues** (canonical for collaboration + workflow state).

The user should:

- Authenticate **once per machine** and use Linear in any repo that uses Redesmyn.
- Pull issues into a specific epic in a controlled, deterministic way.
- Push local changes back to Linear with a simple overwrite model (v0), while preserving a seam for smarter conflict handling later.

## 2) Scope (v0)

### 2.1 Must-have

- `rn linear auth/status/logout` with **machine-scoped** credentials.
- Token refresh + expiration handling (read/write scopes).
- `rn sync --from linear`:
  - pull Linear issues → local task docs + DB tasks
  - filter by a label matching the epic slug (see §4.2)
  - allocate stable local task ids (`T-###`) for imported issues
  - infer `parent`/`after` from Linear blockers with an interactive “pick parent” flow for multi-blockers
- `rn sync --to linear`:
  - create issues for local-only tasks and apply the epic label
  - overwrite title/description/state/dependencies for linked issues (naive overwrite)
- Dashboard: a **single “Linear” sync menu button** (with embedded status) surfaced at:
  - epic-level (graph view subheader)
  - task detail panel (for the selected task)

### 2.2 Explicit non-goals (v0)

- Continuous polling or webhook-driven updates (later).
- Syncing comments/reactions or modeling Linear-derived items as events.
- Smart conflict resolution; we record known conflict situations and overwrite in v0.
- Multi-tenant cloud auth/onboarding (but we preserve architectural seams; see §4.1).

## 3) Principles

- **User-scoped credentials**: tokens belong to a user identity, not a repo.
- **Determinism over magic**: sync is explicit; the user decides when to pull/push.
- **Docs stay human-editable**: task docs remain Markdown-first; local “Brief” is never overwritten.
- **Graph-first UX**: sync affordances should be actionable and non-noisy (no table-heavy “integration settings” screens).

## 4) Key decisions

### 4.1 Auth architecture: simple local now, cloud later

v0 targets a single developer on a single machine:

- Store Linear credentials in a **machine-level OS keychain** (not the repo DB).
- Any client process (CLI, daemon, UI helper) may use the credential store; the daemon is not required to be running to authenticate.
- Provide a migration path from the current repo-scoped `LinearAuth` record to the machine credential store.

Future cloud mode informs the design:

- The control plane must not assume a single credential set exists.
- Credentials remain **user-scoped**; server-side sync (if any) requires explicit user identity and secret storage.

Implementation seam:

- Introduce a small credential-store interface so we can swap:
  - local machine store (v0), and
  - per-user server store (v1+).

### 4.2 Epic scoping in Linear: label == epic slug

Multiple epics may map to the same Linear project (many epics → one project).

Scoping rule (v0):

- An epic’s Linear label is **exactly the epic slug**.
- `sync --from linear` pulls only issues in the configured project with that label.
- `sync --to linear` ensures the label is applied to created/updated issues.

### 4.3 Dependencies mapping (best-effort, lossy)

- Local `parent` (single parent) maps to one Linear “blocked by” edge.
- Local `after` maps to additional “blocked by” edges.
- On pull:
  - 0 blockers → `parent = null`
  - 1 blocker → set `parent` to that issue
  - >1 blockers → prompt user to pick `parent` (including “No parent”), and store the remaining blockers in `after`

### 4.4 State mapping (coarse)

Local `TaskState` maps to Linear state types:

- `todo` → `unstarted`
- `in_progress` → `started`
- `blocked` → `blocked`
- `done` → `completed`

We accept that other integrations may also update Linear state; v0 overwrites on push and does not attempt reconciliation.

### 4.5 Overwrite model (v0)

On push to an already-linked issue, overwrite Linear fields we manage (title/description/state/dependencies/labels).

Known conflict situations we explicitly punt on in v0:

- edits in both Linear and local between syncs
- issues moving across epics/projects
- dependency shapes that don’t fit the tree constraint

## 5) UI/UX: “Linear” sync menu button

One shared control pattern at epic + task levels:

- Button label includes connection status (“Connected”/“Not connected”) in a subtle, non-redundant way.
- Menu items are actionable:
  - Connect / Disconnect
  - Sync from Linear / Sync to Linear
  - Open in Linear (project or issue)

## 6) Task map

- `epics/linear-integration/tasks/T-1/README.md`: Machine-scoped OAuth (PKCE) + credential store + refresh.
- `epics/linear-integration/tasks/T-2/README.md`: Linear write-capable client (labels/projects/teams/issues/state/dependencies).
- `epics/linear-integration/tasks/T-3/README.md`: `rn sync --from linear` (label-filtered import + ID allocation + interactive parent selection).
- `epics/linear-integration/tasks/T-4/README.md`: `rn sync --to linear` (create/update + overwrite semantics + dependency push).
- `epics/linear-integration/tasks/T-5/README.md`: Dashboard sync button + minimal endpoints/wiring.
- `epics/linear-integration/tasks/T-6/README.md`: Automation: push local status → Linear + per-task sync indicator.

## 7) Notes on coordination with other epics

- `epics/revise-architecture/README.md`: this epic should avoid assumptions that the control plane has repo filesystem access or a single global credential set.
- `epics/redesmyn/tasks/T-4/README.md` (“Remove task authority”): Linear sync should not deepen reliance on `TaskAuthority`; treat it as a transitional field and prefer explicit refs + `source`.
