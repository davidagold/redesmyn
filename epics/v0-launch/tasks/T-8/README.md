# T-8 Package dashboard assets for installs

## Metadata

```yaml
id: T-8
stacked_on: T-7
node:
  branch: rn/v0-launch/T-8-packaged-dashboard
```

## Brief (local)

Make installed Redesmyn self-contained:

- Ship prebuilt dashboard assets with the Python package (wheel/sdist).
- Serve the packaged dashboard by default.
- Keep local dev HMR flow via `just dev` (Vite) without requiring packaged assets.

## Acceptance Criteria

- A user can install `redesmyn` without Node and still get a working dashboard UI.
- The control plane serves UI assets even when the target repo does not contain a `dashboard/` directory.

