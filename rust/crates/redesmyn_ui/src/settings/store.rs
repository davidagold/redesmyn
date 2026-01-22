use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::components::SplitPaneState;
use crate::settings::ThemePreference;

const UI_SETTINGS_FILE_NAME: &str = "ui_settings.json";

#[derive(Debug, Error)]
pub enum UiSettingsError {
    #[error("ui settings path is unavailable")]
    PathUnavailable,

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct UiSettings {
    pub version: u32,
    pub theme: ThemePreference,
    pub main_split_pane: SplitPaneState,
}

impl Default for UiSettings {
    fn default() -> Self {
        Self {
            version: 1,
            theme: ThemePreference::default(),
            main_split_pane: SplitPaneState::default(),
        }
    }
}

#[derive(Debug)]
pub struct UiSettingsStore {
    path: Option<PathBuf>,
    settings: UiSettings,
}

impl UiSettingsStore {
    pub fn load() -> Self {
        let span = redesmyn_logging::redesmyn_info_span!("ui_settings_load");
        let _guard = span.enter();

        let path = ui_settings_path();
        let settings = match path.as_deref() {
            Some(path) => load_file(path),
            None => {
                redesmyn_logging::tracing::warn!("ui settings path unavailable; using defaults");
                UiSettings::default()
            }
        };

        Self { path, settings }
    }

    pub fn settings(&self) -> &UiSettings {
        &self.settings
    }

    pub fn settings_mut(&mut self) -> &mut UiSettings {
        &mut self.settings
    }

    pub fn save(&self) -> Result<(), UiSettingsError> {
        let path = self
            .path
            .as_deref()
            .ok_or(UiSettingsError::PathUnavailable)?;
        save_file(path, &self.settings)
    }
}

fn ui_settings_path() -> Option<PathBuf> {
    let config_path = redesmyn_config::global_config_path()?;
    let dir = config_path.parent()?;
    Some(dir.join(UI_SETTINGS_FILE_NAME))
}

fn load_file(path: &Path) -> UiSettings {
    match fs::read_to_string(path) {
        Ok(contents) => match serde_json::from_str::<UiSettings>(&contents) {
            Ok(settings) => settings,
            Err(error) => {
                redesmyn_logging::tracing::error!(
                    error = %error,
                    path = %path.display(),
                    "failed to parse ui settings; using defaults"
                );
                UiSettings::default()
            }
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => UiSettings::default(),
        Err(error) => {
            redesmyn_logging::tracing::error!(
                error = %error,
                path = %path.display(),
                "failed to read ui settings; using defaults"
            );
            UiSettings::default()
        }
    }
}

fn save_file(path: &Path, settings: &UiSettings) -> Result<(), UiSettingsError> {
    let span = redesmyn_logging::redesmyn_info_span!("ui_settings_save", path = %path.display());
    let _guard = span.enter();

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let contents = serde_json::to_string_pretty(settings)?;
    let tmp_path = tmp_path_for(path);

    fs::write(&tmp_path, contents)?;
    fs::rename(&tmp_path, path)?;

    Ok(())
}

fn tmp_path_for(path: &Path) -> PathBuf {
    let mut tmp_path = path.as_os_str().to_os_string();
    tmp_path.push(".tmp");
    PathBuf::from(tmp_path)
}
