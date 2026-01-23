use std::path::Path;
use std::process::Command;

use gpui::{AnyWindowHandle, App};
use redesmyn_protocol::ui_driver::UiScreenshotWindow;
use redesmyn_protocol::{ErrorCategory, ErrorEnvelope};

#[derive(Debug, Clone, Copy)]
pub enum ScreenshotTarget {
    #[cfg(target_os = "macos")]
    MacosWindow {
        window_number: i64,
    },
    #[cfg(target_os = "macos")]
    MacosScreen,
    Unsupported,
}

pub fn select_target(
    window: Option<UiScreenshotWindow>,
    cx: &mut App,
) -> Result<ScreenshotTarget, ErrorEnvelope> {
    let window = window.unwrap_or(UiScreenshotWindow::Primary);

    #[cfg(target_os = "macos")]
    {
        if window == UiScreenshotWindow::All {
            return Ok(ScreenshotTarget::MacosScreen);
        }

        let handle = select_primary_window(cx)?;
        let window_number = handle
            .update(cx, |_, window, _cx| macos_window_number(window))
            .map_err(|err| ErrorEnvelope::new(ErrorCategory::Unavailable, err.to_string()))??;

        return Ok(ScreenshotTarget::MacosWindow { window_number });
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = window;
        let _ = cx;
        Err(ErrorEnvelope::new(
            ErrorCategory::Unavailable,
            "Screenshot capture is not implemented on this platform.",
        ))
    }
}

pub fn capture_png(
    target: ScreenshotTarget,
    include_decorations: bool,
    output_path: &Path,
) -> Result<(), ErrorEnvelope> {
    match target {
        #[cfg(target_os = "macos")]
        ScreenshotTarget::MacosWindow { window_number } => {
            let mut cmd = Command::new("screencapture");
            cmd.arg("-x");
            if !include_decorations {
                cmd.arg("-o");
            }
            cmd.arg("-l");
            cmd.arg(window_number.to_string());
            cmd.arg(output_path);

            let status = cmd.status().map_err(|err| {
                ErrorEnvelope::new(
                    ErrorCategory::Unavailable,
                    format!("Failed to run screencapture: {err}"),
                )
            })?;

            if !status.success() {
                return Err(ErrorEnvelope::new(
                    ErrorCategory::Unavailable,
                    format!("screencapture exited with status {status}"),
                ));
            }

            Ok(())
        }
        #[cfg(target_os = "macos")]
        ScreenshotTarget::MacosScreen => {
            let status = Command::new("screencapture")
                .arg("-x")
                .arg(output_path)
                .status()
                .map_err(|err| {
                    ErrorEnvelope::new(
                        ErrorCategory::Unavailable,
                        format!("Failed to run screencapture: {err}"),
                    )
                })?;

            if !status.success() {
                return Err(ErrorEnvelope::new(
                    ErrorCategory::Unavailable,
                    format!("screencapture exited with status {status}"),
                ));
            }

            Ok(())
        }
        ScreenshotTarget::Unsupported => Err(ErrorEnvelope::new(
            ErrorCategory::Unavailable,
            "Screenshot capture is unsupported on this platform.",
        )),
    }
}

fn select_primary_window(cx: &mut App) -> Result<AnyWindowHandle, ErrorEnvelope> {
    if let Some(handle) = cx.active_window() {
        return Ok(handle);
    }

    if let Some(stack) = cx.window_stack() {
        if let Some(handle) = stack.first().copied() {
            return Ok(handle);
        }
    }

    cx.windows()
        .into_iter()
        .next()
        .ok_or_else(|| ErrorEnvelope::new(ErrorCategory::Unavailable, "No windows available."))
}

#[cfg(target_os = "macos")]
fn macos_window_number(window: &gpui::Window) -> Result<i64, ErrorEnvelope> {
    use objc2::MainThreadMarker;
    use objc2::rc::Retained;
    use objc2_app_kit::NSView;
    use raw_window_handle::RawWindowHandle;

    if MainThreadMarker::new().is_none() {
        return Err(ErrorEnvelope::new(
            ErrorCategory::Unavailable,
            "Screenshot capture must run on the main thread.",
        ));
    }

    let handle = raw_window_handle::HasWindowHandle::window_handle(window).map_err(|err| {
        ErrorEnvelope::new(
            ErrorCategory::Unavailable,
            format!("Window handle is unavailable: {err}"),
        )
    })?;

    let RawWindowHandle::AppKit(handle) = handle.as_raw() else {
        return Err(ErrorEnvelope::new(
            ErrorCategory::Unavailable,
            "Unsupported window handle type for screenshot capture.",
        ));
    };

    let ns_view_ptr = handle.ns_view.as_ptr();
    let Some(ns_view) = (unsafe { Retained::<NSView>::retain(ns_view_ptr.cast()) }) else {
        return Err(ErrorEnvelope::new(
            ErrorCategory::Unavailable,
            "Failed to retain NSView for screenshot capture.",
        ));
    };

    let Some(ns_window) = ns_view.window() else {
        return Err(ErrorEnvelope::new(
            ErrorCategory::Unavailable,
            "NSView is not attached to an NSWindow.",
        ));
    };

    Ok(ns_window.windowNumber() as i64)
}
