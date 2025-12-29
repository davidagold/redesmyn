# T-1 Machine-scoped Linear auth (PKCE) + refresh

## Metadata

```yaml
id: T-1
stacked_on:
node:
  branch: rn/linear-integration/T-1-linear-auth
```

## Brief (local)

- Make Linear credentials user-scoped and usable across repos on the same machine.
- Keep UX smooth: `rn linear auth` should not require the daemon to be running.
- Support read/write scopes and robust token refresh.

## Acceptance Criteria

- `rn linear auth` completes successfully without requiring `rn dev` / daemon/API server to be running.
- Tokens are stored in the machine OS keychain (not `.redesmyn/redesmyn.sqlite3`).
- Existing repo-scoped credentials (if present) can be migrated to the machine credential store.
- Token refresh is implemented and used automatically when the access token is expired/expiring.
- `rn linear status` reports connected/disconnected (and optionally the connected account/workspace).
- `rn linear logout` clears local credentials.

## Notes / Design

- Prefer OAuth PKCE so Redesmyn can ship a client id without shipping a client secret.
- Keep a thin credential-store interface so cloud mode can store creds per user.
