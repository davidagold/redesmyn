---
rn:
  parent: T-1
---

# T-8 Package dashboard assets for installs

## Plan

The v0 install/startup flow should not require Node. Today, the control plane only serves the dashboard if the target repo
contains a built `dashboard/dist` directory. For “run against an arbitrary repo”, that won’t be true.

This task makes the UI assets a property of the Redesmyn install (the package), not a property of the user’s repo.

### Work

1) Build + include assets in the Python package

- Decide the packaging strategy (wheel/sdist):
  - include `dashboard/dist` (or a dedicated `redesmyn/dashboard_dist` folder) as package data.
  - ensure build tooling produces deterministic assets for release builds.

2) Serve packaged assets by default

- Update the server’s dashboard mounting logic to prefer packaged assets.
- Keep the repo-local `dashboard/dist` path as a dev-only override if useful (but do not require it).

3) Smoke-test the installed flow

- With only Python deps installed (no Node), `rn up` should still render the dashboard.
- Ensure the root route serves `index.html` correctly and SPA routing works.

Make installed Redesmyn self-contained:

- Ship prebuilt dashboard assets with the Python package (wheel/sdist).
- Serve the packaged dashboard by default.
- Keep local dev HMR flow via `just dev` (Vite) without requiring packaged assets.

## Acceptance Criteria

- A user can install `redesmyn` without Node and still get a working dashboard UI.
- The control plane serves UI assets even when the target repo does not contain a `dashboard/` directory.
