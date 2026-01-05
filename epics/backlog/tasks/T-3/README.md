# T-3 Linear: recover from 401 “not authenticated” in server API

## Metadata

```yaml
id: T-3
epic: backlog
stacked_on:
branch:
  suggested: rn/backlog/T-3-linear-auth-recovery
```

## Problem

When interacting with Linear via the **server** API, we sometimes return a 502 due to Linear returning a 401, even though `rn linear status` reports “connected”.

Observed server logs (Jan 5, 2026):

- Linear GraphQL op `IssueUrl` returns 401:
  - `Authentication required, not authenticated`
- Request route: `GET /v1/linear/issues/<issue_id>/open`
- Server responds: `502 Bad Gateway`

This is confusing and breaks the “Linear is connected” expectation.

## Why this matters

- The UI uses server routes like `/v1/linear/issues/<id>/open` as part of the Linear integration UX.
- A transient auth failure should not produce a scary 502 with no clear recovery path.
- We need reliable auth refresh + clear user guidance when re-auth is required.

## Goals

1. **Correctness**: the server should use valid Linear credentials (refreshing if needed) for all Linear routes.
2. **Good UX**: if Linear returns 401:
   - do not surface it as a 502
   - respond with a clear “needs re-auth” error that the UI can handle
3. **Explainability**: make it obvious whether the CLI and server are using the same credential source and why they might diverge.

## Investigation checklist

### A) Reproduction

- Confirm a minimal repro:
  - `rn linear status` reports connected
  - UI triggers `/v1/linear/issues/<id>/open`
  - server receives a 401 from Linear
- Capture:
  - whether the server process was started before/after auth was established
  - whether the server uses the same DB file as the CLI (and same workspace/repo context)

### B) Credential lifecycle

- Inspect how credentials are stored and loaded:
  - where access token / refresh token live (DB vs files)
  - whether server caches credentials in memory (and can become stale)
  - whether refresh is performed lazily per request (preferred) vs at startup
- Validate `connected_at` semantics:
  - does “connected” mean “has tokens” or “tokens verified recently”?

### C) HTTP / error mapping

- Ensure we don’t wrap upstream 401 as 502.
- Define a stable error response shape for the frontend:
  - suggested: 401 or 403 with `detail` explaining re-auth is required
  - include a `code` string (e.g. `linear_auth_required`) if we have a pattern for that

### D) Auto-recovery

- If we have refresh tokens:
  - attempt refresh on 401 once, then retry the operation
  - if refresh fails, fall back to “reauth required”
- If we do not have refresh tokens or they’re invalid:
  - return “reauth required” immediately

## Acceptance Criteria

- `/v1/linear/issues/<id>/open`:
  - succeeds when credentials are valid
  - if credentials are stale/invalid:
    - performs refresh + retry if possible
    - otherwise returns a non-502 response that clearly indicates re-auth is needed
- `rn linear status` and server behavior are aligned (or divergences are explained / fixed).
- Add a focused regression test (unit or integration) covering:
  - upstream 401 → refresh+retry path (if supported)
  - upstream 401 → “reauth required” response path

## Notes

- Implementer should consult Linear’s auth/token docs (and any upstream SDK behavior) to ensure the refresh flow is correct and safe.
- Prefer minimal surface area: one shared “Linear request” helper that handles retry-on-401 for all server Linear routes, rather than per-route bespoke logic.

