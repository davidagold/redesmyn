use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use redesmyn_logging::tracing;

const ARTIFACTS_DIR_ENV: &str = "REDESMYN_TEST_ARTIFACTS_DIR";

#[derive(Debug)]
pub struct UiTestArtifacts {
    run_dir: PathBuf,
}

static UI_TEST_ARTIFACTS: OnceLock<Option<Arc<UiTestArtifacts>>> = OnceLock::new();

pub fn ui_test_artifacts() -> Option<Arc<UiTestArtifacts>> {
    UI_TEST_ARTIFACTS
        .get_or_init(|| {
            let base_dir = std::env::var_os(ARTIFACTS_DIR_ENV).map(PathBuf::from)?;
            match UiTestArtifacts::new(base_dir) {
                Ok(artifacts) => Some(Arc::new(artifacts)),
                Err(err) => {
                    tracing::warn!(error = %err, "failed to initialize ui test artifacts");
                    None
                }
            }
        })
        .clone()
}

impl UiTestArtifacts {
    fn new(base_dir: PathBuf) -> Result<Self, std::io::Error> {
        let run_dir = base_dir
            .join("gpui_desktop_ui_driver")
            .join(unique_run_id());
        std::fs::create_dir_all(&run_dir)?;
        tracing::info!(run_dir = %run_dir.display(), "ui test artifacts initialized");
        Ok(Self { run_dir })
    }

    pub fn run_dir(&self) -> &Path {
        &self.run_dir
    }

    pub fn ui_snapshot_path(&self, label: &str) -> PathBuf {
        self.run_dir
            .join(format!("ui_snapshot_{}.json", sanitize_label(label)))
    }

    pub fn screenshot_path(&self, label: &str) -> PathBuf {
        self.run_dir
            .join(format!("screenshot_{}.png", sanitize_label(label)))
    }
}

pub fn sanitize_label(label: &str) -> String {
    let label = label.trim();
    if label.is_empty() {
        return "checkpoint".to_string();
    }

    label
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => ch,
            _ => '_',
        })
        .collect()
}

pub fn write_json_pretty<T: serde::Serialize>(path: &Path, value: &T) -> Result<(), std::io::Error> {
    ensure_parent_dir(path)?;
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;
    std::fs::write(path, bytes)?;
    Ok(())
}

pub fn ensure_parent_dir(path: &Path) -> Result<(), std::io::Error> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    std::fs::create_dir_all(parent)?;
    Ok(())
}

pub fn temp_png_path(label: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "redesmyn_screenshot_{}_{}.png",
        sanitize_label(label),
        unique_run_id()
    ));
    path
}

fn unique_run_id() -> String {
    let pid = std::process::id();
    let unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0);
    format!("run_{pid}_{unix_ms}")
}
