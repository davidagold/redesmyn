use std::ffi::OsString;
use std::sync::Mutex;

use redesmyn_config::{
    ConfigFiles, ConfigProfile, DotenvMode, LoadConfigError, LoadConfigOptions, ValidationError,
    load_rust_config,
};

static ENV_MUTEX: Mutex<()> = Mutex::new(());

#[allow(unused_unsafe)]
fn set_env_var(key: &str, value: &str) {
    unsafe { std::env::set_var(key, value) };
}

#[allow(unused_unsafe)]
fn set_env_var_os(key: &str, value: &OsString) {
    unsafe { std::env::set_var(key, value) };
}

#[allow(unused_unsafe)]
fn remove_env_var(key: &str) {
    unsafe { std::env::remove_var(key) };
}

struct EnvVarGuard {
    key: &'static str,
    original: Option<OsString>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let original = std::env::var_os(key);
        set_env_var(key, value);
        Self { key, original }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match &self.original {
            Some(value) => set_env_var_os(self.key, value),
            None => remove_env_var(self.key),
        }
    }
}

#[test]
fn layered_precedence_defaults_files_env() {
    let _guard = ENV_MUTEX
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let tmp = tempfile::tempdir().expect("temp dir");
    let repo_root = tmp.path().join("repo");

    let global_path = tmp.path().join("global.toml");
    std::fs::write(
        &global_path,
        r#"
[rust.control_plane.api]
bind = "127.0.0.1:1111"

[rust.daemon.executor]
max_concurrency = 9
"#,
    )
    .expect("write global config");

    let repo_path = tmp.path().join("repo.toml");
    std::fs::write(
        &repo_path,
        r#"
[rust.control_plane.api]
bind = "127.0.0.1:2222"
"#,
    )
    .expect("write repo config");

    let _env = EnvVarGuard::set("REDESMYN_RUST__CONTROL_PLANE__API__BIND", "127.0.0.1:3333");

    let config = load_rust_config(LoadConfigOptions {
        repo_root: Some(repo_root),
        config_files: ConfigFiles::Explicit(vec![global_path, repo_path]),
        dotenv: DotenvMode::Never,
        include_env: true,
        default_profile: ConfigProfile::Dev,
    })
    .expect("load config");

    assert_eq!(config.control_plane.api.bind.port(), 3333);
    assert_eq!(config.daemon.executor.max_concurrency, 9);
}

#[test]
fn release_profile_rejects_default_daemon_token() {
    let _guard = ENV_MUTEX
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let tmp = tempfile::tempdir().expect("temp dir");
    let config_path = tmp.path().join("config.toml");
    std::fs::write(
        &config_path,
        r#"
[rust]
profile = "release"
"#,
    )
    .expect("write config");

    let err = load_rust_config(LoadConfigOptions {
        repo_root: Some(tmp.path().to_path_buf()),
        config_files: ConfigFiles::Explicit(vec![config_path]),
        dotenv: DotenvMode::Never,
        include_env: false,
        default_profile: ConfigProfile::Dev,
    })
    .expect_err("expected validation error");

    match err {
        LoadConfigError::Validation(ValidationError::ReleaseDaemonTokenIsDev) => {}
        other => panic!("unexpected error: {other:?}"),
    }
}
