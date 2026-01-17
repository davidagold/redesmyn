---
rn:
  node:
    branch: rn/github-integration/T-1-github-auth
  parent: null
  after: []
---

# T-1 Machine-scoped GitHub auth (OAuth) + credential store

## Brief (local)

- Add machine-scoped GitHub OAuth authentication for GitHub.com.
- Persist credentials in OS keychain (not the repo DB).
- Surface “connected” vs “warning” (missing scopes) in both CLI and dashboard.

## Acceptance Criteria

- `rn github auth` completes successfully without requiring the daemon to be running.
- Credentials are stored in the OS keychain and can be cleared via `rn github logout`.
- `rn github status` reports:
  - connected/disconnected (based on an authenticated API call),
  - granted scopes (or best-effort equivalent),
  - warning when scopes are insufficient for PR creation in the current repo.
- Dashboard can query a status endpoint to render connection state + warning.

## Notes / Design

- Prefer OAuth with PKCE or an equivalent approach that avoids requiring a client secret for basic local use.
- “Connected” should be tolerant: if a token exists and authenticates, keep “connected” but show a warning when required scopes are missing.
