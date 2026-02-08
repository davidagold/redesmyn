# GPUI engineering notes (for agents)

This doc captures practical gotchas and patterns learned while implementing GPUI work in this repo.
It is intentionally biased toward “what will bite you” rather than an end-user tutorial.

## Dependency + toolchain

- GPUI dependency is crates.io `gpui = 0.2.2`.
- GPUI uses the Rust 2024 edition. The workspace pins `rust-version = 1.88` (see `rust/Cargo.toml` and `rust/rust-toolchain.toml`).
- We intentionally control GPUI features at the workspace level (see `rust/Cargo.toml`):
  - `default-features = false` to avoid pulling in platform backends we don’t use on a given OS.
  - macOS: enable `macos-blade` + `font-kit`.
  - Enable `runtime_shaders` so macOS builds don’t require the Metal shader toolchain at build time.
- Linux crates that need a window backend add `gpui` with `features = ["wayland", "x11"]` under a target-specific dependency stanza (see `rust/crates/*/Cargo.toml`).

## Known build quirks

- `core-text` is pinned in `rust/Cargo.lock` (currently `21.0.0`) to avoid a `core-graphics` type mismatch in transitive font dependencies. If you refresh the lockfile and see type mismatch errors in CoreGraphics/CoreText, this is the first thing to re-check.
- We vendor a tiny `zune-jpeg` fix via `[patch.crates-io]` in `rust/Cargo.toml` so GPUI builds on Apple Silicon.

## Where to look for upstream reference code

- GPUI’s best “documentation” is its examples. After `cargo fetch`, they live in your Cargo registry checkout, e.g. `~/.cargo/registry/src/.../gpui-0.2.2/examples/`.
  - `examples/input.rs` is a particularly useful reference for text input plumbing.

## Patterns we standardize on

### Actions + keybindings

- Prefer typed `actions!(...)` + `KeyBinding` + `.on_action(...)` over bespoke per-view key handlers.
- Many `cx.*` methods come from the `gpui::AppContext` trait; if you get “no method found” errors, you may need `use gpui::AppContext as _;`.
- For custom inputs, use explicit key contexts and bind keys centrally:
  - `TextInput` uses `.key_context("TextInput")`
  - `TextArea` uses `.key_context("TextArea")`
  - Call `redesmyn_ui::components::bind_text_input_keys(cx)` once during app initialization.

### Text input + IME gotchas

- GPUI’s selection contracts are UTF-16 based (`UTF16Selection`). If you store text in `String`/`SharedString`, you must translate between UTF-8 byte offsets and UTF-16 code unit offsets correctly.
- IME candidate window placement relies on `bounds_for_range` and `character_index_for_point`. Even a minimal “v1” input needs these to avoid broken composition UX.

### Focus + window lifecycle

- If you want “quit when last window closes”, wire `cx.on_window_closed` and check `cx.windows().is_empty()`.
- Focusing an element on startup often requires a post-open `window.update(...)` call that focuses a `FocusHandle`, plus `cx.activate(true)` to bring the app to the foreground.

### Async tasks without leaks

- Use `cx.spawn` with a `WeakEntity` when work may outlive the view/window. Always bail if `weak.upgrade()` fails.

### “No silent actions” UX

- Any user-triggered request must flip visible in-flight state immediately and disable triggers to prevent duplicates.
- Prefer calm progress affordances (e.g. animated ellipses) over spinner wheels.

### Theme + tokens

- Treat “System” as first-class: derive effective theme from preference + `WindowAppearance`.
- Use typed tokens (`styles::UiTheme`) instead of ad-hoc colors and “px soup”.

## Running the UI survey surface

- Normal desktop shell: `cargo run -p redesmyn_desktop`
- Foundations demo window: `REDESMYN_UI_FOUNDATIONS_DEMO=1 cargo run -p redesmyn_desktop`

