use std::fs;
use std::path::{Path, PathBuf};

use toml_edit::{DocumentMut, Item, Table, value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentKindSelection {
    Auto,
    Generic,
    Codex,
    ClaudeCode,
}

impl AgentKindSelection {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Generic => "generic",
            Self::Codex => "codex",
            Self::ClaudeCode => "claude_code",
        }
    }
}

impl Default for AgentKindSelection {
    fn default() -> Self {
        Self::Auto
    }
}

impl std::str::FromStr for AgentKindSelection {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "auto" => Ok(Self::Auto),
            "generic" => Ok(Self::Generic),
            "codex" => Ok(Self::Codex),
            "claude_code" => Ok(Self::ClaudeCode),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandboxType {
    None,
    Worktree,
}

impl SandboxType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Worktree => "worktree",
        }
    }
}

impl Default for SandboxType {
    fn default() -> Self {
        Self::None
    }
}

impl std::str::FromStr for SandboxType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "none" => Ok(Self::None),
            "worktree" => Ok(Self::Worktree),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandboxNetworkMode {
    Allow,
    Deny,
}

impl SandboxNetworkMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Allow => "allow",
            Self::Deny => "deny",
        }
    }
}

impl Default for SandboxNetworkMode {
    fn default() -> Self {
        Self::Allow
    }
}

impl std::str::FromStr for SandboxNetworkMode {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "allow" => Ok(Self::Allow),
            "deny" => Ok(Self::Deny),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarnessDefaults {
    pub command: Option<String>,
    pub agent_kind: AgentKindSelection,
    pub detach: bool,
    pub prelude: Option<String>,
    pub send_prelude: bool,
    pub submit_prelude: bool,
}

impl Default for HarnessDefaults {
    fn default() -> Self {
        Self {
            command: None,
            agent_kind: AgentKindSelection::Auto,
            detach: true,
            prelude: None,
            send_prelude: true,
            submit_prelude: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrchestrationDefaults {
    pub harness: HarnessDefaults,
    pub sandbox_type: SandboxType,
    pub sandbox_network: SandboxNetworkMode,
}

impl Default for OrchestrationDefaults {
    fn default() -> Self {
        Self {
            harness: HarnessDefaults::default(),
            sandbox_type: SandboxType::None,
            sandbox_network: SandboxNetworkMode::Allow,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct PartialHarnessDefaults {
    command: Option<Option<String>>,
    agent_kind: Option<AgentKindSelection>,
    detach: Option<bool>,
    prelude: Option<Option<String>>,
    send_prelude: Option<bool>,
    submit_prelude: Option<bool>,
}

#[derive(Debug, Clone, Default)]
struct PartialOrchestrationDefaults {
    harness: PartialHarnessDefaults,
    sandbox_type: Option<SandboxType>,
    sandbox_network: Option<SandboxNetworkMode>,
}

#[derive(Debug, thiserror::Error)]
pub enum OrchestrationConfigError {
    #[error("failed to resolve the repo root: {source}")]
    RepoRoot { source: std::io::Error },
    #[error("failed to read config: {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to parse config: {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml_edit::TomlError,
    },
    #[error("failed to write config: {path}: {source}")]
    Write {
        path: PathBuf,
        source: std::io::Error,
    },
}

pub fn repo_root_from_cwd() -> Result<PathBuf, OrchestrationConfigError> {
    let cwd =
        std::env::current_dir().map_err(|source| OrchestrationConfigError::RepoRoot { source })?;
    let root = redesmyn_config::discover_repo_root_from(&cwd).unwrap_or(cwd);
    Ok(root)
}

pub fn repo_config_path(repo_root: &Path) -> PathBuf {
    redesmyn_config::repo_config_path(repo_root)
}

pub fn load_effective_defaults(
    repo_root: &Path,
) -> Result<OrchestrationDefaults, OrchestrationConfigError> {
    let mut defaults = OrchestrationDefaults::default();
    if let Some(global_path) = redesmyn_config::global_config_path() {
        if global_path.exists() {
            let partial = load_partial_defaults(&global_path)?;
            apply_partial(&mut defaults, partial);
        }
    }

    let repo_path = repo_config_path(repo_root);
    if repo_path.exists() {
        let partial = load_partial_defaults(&repo_path)?;
        apply_partial(&mut defaults, partial);
    }

    normalize(&mut defaults);
    Ok(defaults)
}

pub fn write_repo_defaults(
    repo_root: &Path,
    defaults: &OrchestrationDefaults,
) -> Result<(), OrchestrationConfigError> {
    let path = repo_config_path(repo_root);
    write_defaults_to_path(&path, defaults)
}

fn normalize(defaults: &mut OrchestrationDefaults) {
    if !defaults.harness.send_prelude {
        defaults.harness.submit_prelude = false;
    }
}

fn load_partial_defaults(
    path: &Path,
) -> Result<PartialOrchestrationDefaults, OrchestrationConfigError> {
    let content = fs::read_to_string(path).map_err(|source| OrchestrationConfigError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let doc = content
        .parse::<DocumentMut>()
        .map_err(|source| OrchestrationConfigError::Parse {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(parse_partial_from_doc(&doc))
}

fn parse_partial_from_doc(doc: &DocumentMut) -> PartialOrchestrationDefaults {
    let mut partial = PartialOrchestrationDefaults::default();

    if let Some(harness) = doc.get("harness").and_then(|item| item.as_table()) {
        if harness.contains_key("command") {
            partial.harness.command = Some(
                harness
                    .get("command")
                    .and_then(|item| item.as_str())
                    .map(|s| s.to_string())
                    .filter(|s| !s.trim().is_empty()),
            );
        }

        partial.harness.agent_kind = harness
            .get("agent_kind")
            .and_then(|item| item.as_str())
            .and_then(|value| value.parse::<AgentKindSelection>().ok());

        partial.harness.detach = harness.get("detach").and_then(|item| item.as_bool());

        if harness.contains_key("prelude") {
            partial.harness.prelude = Some(
                harness
                    .get("prelude")
                    .and_then(|item| item.as_str())
                    .map(|s| s.to_string())
                    .filter(|s| !s.trim().is_empty()),
            );
        }

        partial.harness.send_prelude = harness.get("send_prelude").and_then(|item| item.as_bool());

        partial.harness.submit_prelude = harness
            .get("submit_prelude")
            .and_then(|item| item.as_bool());
    }

    if let Some(sandbox) = doc.get("sandbox").and_then(|item| item.as_table()) {
        partial.sandbox_type = sandbox
            .get("type")
            .and_then(|item| item.as_str())
            .and_then(|value| value.parse::<SandboxType>().ok());
        partial.sandbox_network = sandbox
            .get("network")
            .and_then(|item| item.as_str())
            .and_then(|value| value.parse::<SandboxNetworkMode>().ok());
    }

    partial
}

fn apply_partial(defaults: &mut OrchestrationDefaults, partial: PartialOrchestrationDefaults) {
    if let Some(command) = partial.harness.command {
        defaults.harness.command = command;
    }
    if let Some(agent_kind) = partial.harness.agent_kind {
        defaults.harness.agent_kind = agent_kind;
    }
    if let Some(detach) = partial.harness.detach {
        defaults.harness.detach = detach;
    }
    if let Some(prelude) = partial.harness.prelude {
        defaults.harness.prelude = prelude;
    }
    if let Some(send_prelude) = partial.harness.send_prelude {
        defaults.harness.send_prelude = send_prelude;
    }
    if let Some(submit_prelude) = partial.harness.submit_prelude {
        defaults.harness.submit_prelude = submit_prelude;
    }

    if let Some(sandbox_type) = partial.sandbox_type {
        defaults.sandbox_type = sandbox_type;
    }
    if let Some(sandbox_network) = partial.sandbox_network {
        defaults.sandbox_network = sandbox_network;
    }
}

fn write_defaults_to_path(
    path: &Path,
    defaults: &OrchestrationDefaults,
) -> Result<(), OrchestrationConfigError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| OrchestrationConfigError::Write {
            path: path.to_path_buf(),
            source,
        })?;
    }

    let mut doc = if path.exists() {
        let content =
            fs::read_to_string(path).map_err(|source| OrchestrationConfigError::Read {
                path: path.to_path_buf(),
                source,
            })?;
        content
            .parse::<DocumentMut>()
            .map_err(|source| OrchestrationConfigError::Parse {
                path: path.to_path_buf(),
                source,
            })?
    } else {
        DocumentMut::new()
    };

    let harness = ensure_table(&mut doc, "harness");
    match defaults
        .harness
        .command
        .as_ref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
    {
        Some(command) => {
            harness["command"] = value(command);
        }
        None => {
            harness.remove("command");
        }
    }

    harness["agent_kind"] = value(defaults.harness.agent_kind.as_str());
    harness["detach"] = value(defaults.harness.detach);

    match defaults
        .harness
        .prelude
        .as_ref()
        .filter(|s| !s.trim().is_empty())
    {
        Some(prelude) => {
            harness["prelude"] = value(prelude.as_str());
        }
        None => {
            harness.remove("prelude");
        }
    }

    harness["send_prelude"] = value(defaults.harness.send_prelude);
    harness["submit_prelude"] = value(if defaults.harness.send_prelude {
        defaults.harness.submit_prelude
    } else {
        false
    });

    let sandbox = ensure_table(&mut doc, "sandbox");
    sandbox["type"] = value(defaults.sandbox_type.as_str());
    sandbox["network"] = value(defaults.sandbox_network.as_str());

    fs::write(path, doc.to_string()).map_err(|source| OrchestrationConfigError::Write {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(())
}

fn ensure_table<'a>(doc: &'a mut DocumentMut, key: &str) -> &'a mut Table {
    let entry = doc.entry(key).or_insert(Item::Table(Table::new()));
    if !entry.is_table() {
        *entry = Item::Table(Table::new());
    }
    entry.as_table_mut().expect("table inserted above")
}
