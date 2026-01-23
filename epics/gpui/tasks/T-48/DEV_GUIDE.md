# GPUI Desktop UI Driver (T-48) — Developer Guide

This guide describes how to run the GPUI desktop app with the local-only UI driver enabled, how
to capture artifacts, and how to reproduce a failing run from those artifacts.

## Environment variables

- `REDESMYN_UI_DRIVER_SOCKET_PATH`  
  Enables the UI driver server and binds a Unix domain socket at this path.

- `REDESMYN_TEST_ARTIFACTS_DIR`  
  If set, `CaptureScreenshot` writes checkpoint artifacts under:
  - `gpui_desktop_ui_driver/run_<pid>_<unix_ms>/ui_snapshot_<label>.json`
  - `gpui_desktop_ui_driver/run_<pid>_<unix_ms>/screenshot_<label>.png`

- `REDESMYN_UI_TEST_MODE=1`  
  Enables deterministic fixture behavior for UI testing:
  - forces a pinned theme (default: dark),
  - freezes time-based progress animations,
  - sets `UiSnapshot.captured_at` to Unix epoch for stable diffs.

- `REDESMYN_UI_TEST_THEME=dark|light` (optional)  
  Overrides the pinned theme when `REDESMYN_UI_TEST_MODE=1`.

## Run the desktop app with the driver enabled

From the repo root:

```bash
mkdir -p /tmp/redesmyn-artifacts
export REDESMYN_UI_TEST_MODE=1
export REDESMYN_TEST_ARTIFACTS_DIR=/tmp/redesmyn-artifacts
export REDESMYN_UI_DRIVER_SOCKET_PATH=/tmp/redesmyn-ui-driver.sock

cd rust
cargo run -p redesmyn_desktop
```

## Smoke path (one command)

If you want a single end-to-end "smoke" flow (launch app → select epic → create chat → wait-for-idle → capture artifacts),
run:

```bash
cd rust
cargo run -p rn -- ui-driver smoke --launch --epic gpui --label smoke --artifacts-dir /tmp/redesmyn-artifacts
```

Use `--keep-open` to leave the desktop app running after the smoke flow.

## Driver transport

The UI driver is a **local-only** Unix domain socket server with a 4-byte big-endian length prefix
followed by a protobuf-encoded `UiDriverFrame` (see `rust/proto/ui_driver.proto`).

## Artifacts (snapshots + screenshots)

To capture checkpoint artifacts, call the UI driver method:

- `CaptureScreenshot { name_hint: "<label>", ... }`

When `REDESMYN_TEST_ARTIFACTS_DIR` is set, the driver writes both:

- `ui_snapshot_<label>.json` (semantic snapshot; machine-readable)
- `screenshot_<label>.png` (pixel screenshot)

The driver response includes `png_path`, which can be used to locate the run directory and the
matching `ui_snapshot_<label>.json`.

### Screenshot capture notes

- On macOS, screenshots are captured using the system `screencapture` tool.
- `include_decorations=false` maps to passing `screencapture -o` (omit window shadow).
- `window=all` currently captures the full screen on macOS.

## Reproducing a failing run from artifacts

Given a failing run directory under `REDESMYN_TEST_ARTIFACTS_DIR`:

1. Open `screenshot_<label>.png` to see what the UI looked like.
2. Inspect `ui_snapshot_<label>.json` to see the stable semantic state:
   - selected epic slug (or empty),
   - left pane collapsed + width,
   - primary view,
   - visible in-flight actions,
   - visible error callouts.
