---
slug: github-integration
name: GitHub Integration
root_branch: main
github:
  host: github.com
  auth: oauth_machine_scoped
---

# GitHub Integration Epic: Control Doc (Canonical)

This file is the canonical “control doc” for the **GitHub Integration** epic: intent, v0 spec, invariants, and key decisions. Keep it current.

## 1) Vision

Make GitHub a first-class **PR surface** for Redesmyn stacks:

- PRs are the primary remote artifact for work-in-progress on a task branch.
- Stacked branches imply stacked PRs (a PR’s base is the task’s effective upstream branch).
- The dashboard should make it easy to create/open PRs and understand stack health without leaving the graph.

This integration should share the same “integration affordance” shape as Linear:

- a small, icon-only connection indicator in the top-right,
- an epic-level config badge in the subheader (next to the Linear project badge), and
- a task-level badge above each task card (next to the Linear badge).

## 2) Scope (v0)

### 2.1 Must-have

- **Machine-scoped GitHub OAuth**:
  - `rn github auth/status/logout` (no daemon requirement).
  - Persist tokens in OS keychain.
  - “Connected” means: token exists + a lightweight API call succeeds.
  - If scopes are insufficient, remain “connected” but show a subtle warning signal + tooltip.
- **Repo association**:
  - Default GitHub repo is derived from the local git repo’s remote (likely `origin`).
  - Epic-level GitHub repo mapping, with optional per-task override (v0: keep to same-repo PRs).
  - UI/CLI provides an unobtrusive way to override when auto-detection is wrong.
- **PR operations**:
  - Create PR for a task branch (push first if needed).
  - Open PR in browser.
  - Auto-detect an existing PR for a task branch (best-effort).
  - PR title defaults to task title.
  - PR body should include a Linear issue link when available.
- **Stack-aware PR bases**:
  - Base branch is the task’s effective upstream branch:
    - use the parent branch when unmerged,
    - skip merged ancestors so downstream PRs retarget appropriately.
- **Dashboard**:
  - Add a GitHub integration menu (icon-only button; no provider text, no chevron).
  - Remove “Linear” text from the existing Linear indicator and use icon-only as well.
  - Show a GitHub PR badge above each task card (icon + PR number), with border color reflecting PR state.
  - Add a global toggle (in the GitHub menu) for “auto force-push” behavior when stacks are rewritten.

### 2.2 Explicit non-goals (v0)

- Issue integration (syncing, creation, comment mirroring, etc.).
- Fork-based PRs (head repo != base repo), org-wide repo discovery, and multi-remote complexity.
- Webhooks / background polling; v0 is on-demand fetch + explicit actions.
- PR review / checks UI (nice-to-have in later tasks).
- Merging PRs from the dashboard (v0).

## 3) Key decisions

### 3.1 Connection semantics

- “Connected” means Redesmyn can successfully authenticate to GitHub.
- Missing scopes should show as a **warning state** (not disconnected), since users may still want read-only affordances.

### 3.2 Repository mapping model

Even though a Redesmyn “repository” already exists in the DB, GitHub repo mapping is an integration concern.

v0 plan:

- Derive `owner/repo` from the local git repo (remote URL parsing).
- Persist the chosen repo mapping at epic-level with an optional per-task override.
- Keep the data model extensible to forks (base repo vs head repo) without requiring v0 to implement forks.

### 3.3 PR identity in the DB

Store a stable PR identity that can round-trip across API calls and is human-friendly in the UI.

Recommendation for v0:

- Store `owner/repo` (or equivalent) + PR number.
- Avoid relying on GraphQL node ids in the UI; they’re stable but not ergonomic.
- Keep room to add `node_id` later if we adopt GraphQL for richer PR queries.

### 3.4 Force-push behavior

When a stack is rewritten (restack), branches may need a force-push to keep remote PRs aligned.

v0 introduces a global toggle:

- default: off (do not force push implicitly),
- when enabled: use `--force-with-lease` (never raw `--force`).

## 4) Task map

- `epics/github-integration/tasks/T-1/README.md`: Machine-scoped GitHub OAuth + keychain credential store + status/scopes surface.
- `epics/github-integration/tasks/T-2/README.md`: GitHub repo association (auto-detect + epic/task overrides) + badge UI in epic subheader.
- `epics/github-integration/tasks/T-3/README.md`: PR create/open + PR auto-detection + push/force-push toggle + persist PR identity.
- `epics/github-integration/tasks/T-4/README.md`: Dashboard UX: icon-only integration indicators (Linear + GitHub) + task-card PR badge.
- `epics/github-integration/tasks/T-5/README.md`: Nice-to-have: PR checks/review/close affordances (post-v0).
