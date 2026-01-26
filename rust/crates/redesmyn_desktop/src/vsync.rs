#[cfg(target_os = "macos")]
use std::sync::OnceLock;

use gpui::Window;

#[cfg(target_os = "macos")]
use objc2::ClassType as _;
#[cfg(target_os = "macos")]
use objc2::MainThreadMarker;
#[cfg(target_os = "macos")]
use objc2::rc::Retained;
#[cfg(target_os = "macos")]
use objc2_app_kit::NSView;
#[cfg(target_os = "macos")]
use objc2_foundation::NSObjectProtocol as _;
#[cfg(target_os = "macos")]
use objc2_quartz_core::CAMetalLayer;
#[cfg(target_os = "macos")]
use raw_window_handle::RawWindowHandle;

#[cfg(target_os = "macos")]
pub fn ensure_vsync(window: &Window) {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    let enabled = *ENABLED.get_or_init(|| std::env::var_os("REDESMYN_DISABLE_VSYNC").is_none());
    if !enabled {
        return;
    }

    match try_enable_macos_vsync(window) {
        Ok(true) => {
            redesmyn_logging::tracing::info!("enabled display sync (vsync) for window rendering");
        }
        Ok(false) => {}
        Err(err) => {
            redesmyn_logging::tracing::debug!(error = %err, "failed to ensure window vsync");
        }
    }
}

#[cfg(target_os = "macos")]
#[derive(Debug, thiserror::Error)]
enum EnableVsyncError {
    #[error("vsync must be enabled from the main thread")]
    NotMainThread,
    #[error("failed to read window handle: {0}")]
    WindowHandle(raw_window_handle::HandleError),
    #[error("unsupported window handle type for vsync configuration")]
    UnsupportedWindowHandle,
    #[error("failed to retain NSView for vsync configuration")]
    RetainView,
    #[error("NSView is missing a backing layer (expected CAMetalLayer)")]
    MissingLayer,
    #[error("backing layer is not a CAMetalLayer")]
    NotMetalLayer,
}

/// Returns `Ok(true)` if we toggled vsync on, `Ok(false)` if already enabled.
#[cfg(target_os = "macos")]
fn try_enable_macos_vsync(window: &Window) -> Result<bool, EnableVsyncError> {
    if MainThreadMarker::new().is_none() {
        return Err(EnableVsyncError::NotMainThread);
    }

    let handle = raw_window_handle::HasWindowHandle::window_handle(window)
        .map_err(EnableVsyncError::WindowHandle)?;
    let RawWindowHandle::AppKit(handle) = handle.as_raw() else {
        return Err(EnableVsyncError::UnsupportedWindowHandle);
    };

    let ns_view_ptr = handle.ns_view.as_ptr();
    let Some(ns_view) = (unsafe { Retained::<NSView>::retain(ns_view_ptr.cast()) }) else {
        return Err(EnableVsyncError::RetainView);
    };

    let Some(layer) = ns_view.layer() else {
        return Err(EnableVsyncError::MissingLayer);
    };

    if !layer.isKindOfClass(&CAMetalLayer::class()) {
        return Err(EnableVsyncError::NotMetalLayer);
    }

    let layer: Retained<CAMetalLayer> = unsafe { Retained::cast_unchecked(layer) };
    if layer.displaySyncEnabled() {
        return Ok(false);
    }

    layer.setDisplaySyncEnabled(true);
    Ok(true)
}

#[cfg(not(target_os = "macos"))]
pub fn ensure_vsync(_window: &Window) {}
