//! Agent taxonomy, argv inference, and turn builders.
//!
//! This module is the canonical place to reason about:
//! - agent *provider* (Codex vs Claude Code vs Shell), and
//! - agent *runtime kind* (tmux shell, structured exec, app-server).
//!
//! Centralizing this logic prevents subtle drift between CLI, daemon, and
//! control plane behavior.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentProvider {
    Codex,
    ClaudeCode,
    /// Generic executable; no structured semantics are assumed.
    ///
    /// Legacy note: Python used the term "generic". We accept `"generic"` as a
    /// deserialization alias for backwards compatibility during the split-codebase period.
    #[serde(alias = "generic")]
    Shell,
}

impl AgentProvider {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Codex => "codex",
            Self::ClaudeCode => "claude_code",
            Self::Shell => "shell",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentRuntimeKind {
    /// Interactive; tmux-first; unstructured.
    ShellTmux,
    /// Per-turn process spawn; structured stdout.
    StructuredExec,
    /// Long-lived server; request/response + event stream.
    AppServer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResolvedAgentInterface {
    pub provider: AgentProvider,
    pub runtime: AgentRuntimeKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentProviderSelection {
    Auto,
    Codex,
    ClaudeCode,
    #[serde(alias = "generic")]
    Shell,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentInvocation {
    pub raw_argv: Vec<String>,
    pub wrapper: Vec<String>,
    pub agent_argv: Vec<String>,
}

impl AgentInvocation {
    #[must_use]
    pub fn from_argv(argv: &[String]) -> Self {
        Self::parse(argv.to_vec())
    }

    #[must_use]
    pub fn parse(raw_argv: Vec<String>) -> Self {
        let wrapper_prefix_len = wrapper_prefix_len(&raw_argv);
        let agent_start = if wrapper_prefix_len == 0 {
            0
        } else {
            find_agent_start_index(&raw_argv, wrapper_prefix_len)
                .unwrap_or(wrapper_prefix_len.min(raw_argv.len()))
        };

        let wrapper = raw_argv.get(..agent_start).unwrap_or_default().to_vec();
        let agent_argv = raw_argv.get(agent_start..).unwrap_or_default().to_vec();

        Self {
            raw_argv,
            wrapper,
            agent_argv,
        }
    }

    #[must_use]
    pub fn agent_executable_name(&self) -> Option<String> {
        self.agent_argv.first().map(|token| argv_token_name(token))
    }

    #[must_use]
    pub fn contains_subcommand(&self, name: &str) -> bool {
        self.agent_argv.iter().any(|token| token == name)
    }

    #[must_use]
    pub fn contains_flag(&self, flag: &str) -> bool {
        self.agent_argv
            .iter()
            .any(|token| token == flag || token.starts_with(&format!("{flag}=")))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExternalSessionRef {
    None,
    CodexThread {
        thread_id: String,
        turn_id: Option<String>,
    },
    CodexSession {
        session_id: String,
        turn_id: Option<String>,
    },
    ClaudeSession {
        session_id: String,
    },
    /// Forward-compat placeholder for future providers / ref types.
    Unknown {
        type_: String,
        raw: serde_json::Value,
    },
}

impl ExternalSessionRef {
    #[must_use]
    pub fn provider_hint(&self) -> Option<AgentProvider> {
        match self {
            Self::CodexThread { .. } | Self::CodexSession { .. } => Some(AgentProvider::Codex),
            Self::ClaudeSession { .. } => Some(AgentProvider::ClaudeCode),
            Self::None | Self::Unknown { .. } => None,
        }
    }

    #[must_use]
    pub fn type_string(&self) -> String {
        match self {
            Self::None => "none".to_owned(),
            Self::CodexThread { .. } => "codex_thread".to_owned(),
            Self::CodexSession { .. } => "codex_session".to_owned(),
            Self::ClaudeSession { .. } => "claude_session".to_owned(),
            Self::Unknown { type_, .. } => type_.clone(),
        }
    }
}

impl Serialize for ExternalSessionRef {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        match self {
            Self::None => {
                let mut st = serializer.serialize_struct("ExternalSessionRef", 1)?;
                st.serialize_field("type", "none")?;
                st.end()
            }
            Self::CodexThread { thread_id, turn_id } => {
                let mut st = serializer.serialize_struct("ExternalSessionRef", 3)?;
                st.serialize_field("type", "codex_thread")?;
                st.serialize_field("thread_id", thread_id)?;
                if let Some(turn_id) = turn_id {
                    st.serialize_field("turn_id", turn_id)?;
                }
                st.end()
            }
            Self::CodexSession {
                session_id,
                turn_id,
            } => {
                let mut st = serializer.serialize_struct("ExternalSessionRef", 3)?;
                st.serialize_field("type", "codex_session")?;
                st.serialize_field("session_id", session_id)?;
                if let Some(turn_id) = turn_id {
                    st.serialize_field("turn_id", turn_id)?;
                }
                st.end()
            }
            Self::ClaudeSession { session_id } => {
                let mut st = serializer.serialize_struct("ExternalSessionRef", 2)?;
                st.serialize_field("type", "claude_session")?;
                st.serialize_field("session_id", session_id)?;
                st.end()
            }
            Self::Unknown { raw, .. } => raw.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for ExternalSessionRef {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        let serde_json::Value::Object(obj) = &value else {
            return Ok(Self::Unknown {
                type_: "<non_object>".to_owned(),
                raw: value,
            });
        };

        let Some(serde_json::Value::String(type_str)) = obj.get("type") else {
            return Ok(Self::Unknown {
                type_: "<missing_type>".to_owned(),
                raw: value,
            });
        };

        match type_str.as_str() {
            "none" => Ok(Self::None),
            "codex_thread" => {
                let thread_id = match obj.get("thread_id") {
                    Some(serde_json::Value::String(s)) => s.clone(),
                    _ => {
                        return Ok(Self::Unknown {
                            type_: type_str.clone(),
                            raw: value,
                        });
                    }
                };
                let turn_id = match obj.get("turn_id") {
                    None | Some(serde_json::Value::Null) => None,
                    Some(serde_json::Value::String(s)) => Some(s.clone()),
                    _ => None,
                };
                Ok(Self::CodexThread { thread_id, turn_id })
            }
            "codex_session" | "codex_conversation" => {
                let session_id = match obj.get("session_id").or_else(|| obj.get("conversation_id"))
                {
                    Some(serde_json::Value::String(s)) => s.clone(),
                    _ => {
                        return Ok(Self::Unknown {
                            type_: type_str.clone(),
                            raw: value,
                        });
                    }
                };
                let turn_id = match obj.get("turn_id") {
                    None | Some(serde_json::Value::Null) => None,
                    Some(serde_json::Value::String(s)) => Some(s.clone()),
                    _ => None,
                };
                Ok(Self::CodexSession {
                    session_id,
                    turn_id,
                })
            }
            "claude_session" => {
                let session_id = match obj.get("session_id") {
                    Some(serde_json::Value::String(s)) => s.clone(),
                    _ => {
                        return Ok(Self::Unknown {
                            type_: type_str.clone(),
                            raw: value,
                        });
                    }
                };
                Ok(Self::ClaudeSession { session_id })
            }
            _ => Ok(Self::Unknown {
                type_: type_str.clone(),
                raw: value,
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResumeByIdTurn {
    pub argv: Vec<String>,
    pub stdin_prompt: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AppServerTurnIntent {
    StartNew {
        prompt: String,
    },
    Resume {
        external: ExternalSessionRef,
        prompt: String,
    },
}

#[must_use]
pub fn build_app_server_turn_intent(
    prompt: &str,
    external_session_ref: Option<&ExternalSessionRef>,
) -> AppServerTurnIntent {
    let prompt = prompt.to_owned();
    match external_session_ref {
        None | Some(ExternalSessionRef::None) => AppServerTurnIntent::StartNew { prompt },
        Some(external) => AppServerTurnIntent::Resume {
            external: external.clone(),
            prompt,
        },
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ResumeByIdExecTurnError {
    #[error("Missing external resume handle; cannot build resume-by-id turn.")]
    MissingExternalResumeHandle,
    #[error("External resume handle is not a Codex thread id.")]
    ExternalHandleNotCodexThread,
    #[error("External resume handle is not a Claude session id.")]
    ExternalHandleNotClaudeSession,
    #[error("Unable to find Codex executable in argv.")]
    MissingCodexExecutable,
    #[error("Codex argv is missing `exec`.")]
    MissingCodexExec,
    #[error("Codex argv must include `--json` for structured resume-by-id turns.")]
    CodexMissingJsonFlag,
    #[error("Unable to find Claude executable in argv.")]
    MissingClaudeExecutable,
    #[error("Claude argv must include `--print` for structured resume-by-id turns.")]
    ClaudeMissingPrintFlag,
    #[error("Claude argv must include `--output-format stream-json` for structured turns.")]
    ClaudeMissingStreamJsonOutput,
    #[error("Agent provider {provider:?} does not support resume-by-id exec turns.")]
    UnsupportedProvider { provider: AgentProvider },
}

#[must_use]
pub fn infer_agent_provider_from_argv(argv: &[String]) -> AgentProvider {
    let invocation = AgentInvocation::from_argv(argv);
    match invocation.agent_argv.first() {
        Some(token) if is_codex_token(token) => AgentProvider::Codex,
        Some(token) if is_claude_code_token(token) => AgentProvider::ClaudeCode,
        _ => AgentProvider::Shell,
    }
}

#[must_use]
pub fn resolve_agent_provider(
    selection: AgentProviderSelection,
    argv: &[String],
    external_session_ref_hint: &ExternalSessionRef,
) -> AgentProvider {
    match selection {
        AgentProviderSelection::Codex => AgentProvider::Codex,
        AgentProviderSelection::ClaudeCode => AgentProvider::ClaudeCode,
        AgentProviderSelection::Shell => AgentProvider::Shell,
        AgentProviderSelection::Auto => external_session_ref_hint
            .provider_hint()
            .unwrap_or_else(|| infer_agent_provider_from_argv(argv)),
    }
}

#[must_use]
pub fn infer_agent_runtime_kind(
    invocation: &AgentInvocation,
    provider: AgentProvider,
) -> AgentRuntimeKind {
    match provider {
        AgentProvider::Shell => AgentRuntimeKind::ShellTmux,
        AgentProvider::Codex => {
            if invocation.contains_subcommand("app-server") {
                AgentRuntimeKind::AppServer
            } else if is_codex_structured_exec_invocation(invocation) {
                AgentRuntimeKind::StructuredExec
            } else {
                AgentRuntimeKind::ShellTmux
            }
        }
        AgentProvider::ClaudeCode => {
            if is_claude_structured_exec_invocation(invocation) {
                AgentRuntimeKind::StructuredExec
            } else {
                AgentRuntimeKind::ShellTmux
            }
        }
    }
}

#[must_use]
pub fn infer_resolved_agent_interface(
    invocation: &AgentInvocation,
    provider: AgentProvider,
) -> ResolvedAgentInterface {
    ResolvedAgentInterface {
        provider,
        runtime: infer_agent_runtime_kind(invocation, provider),
    }
}

pub fn build_resume_by_id_exec_turn(
    base_argv: &[String],
    provider: AgentProvider,
    external_session_ref: &ExternalSessionRef,
    prompt: &str,
) -> Result<ResumeByIdTurn, ResumeByIdExecTurnError> {
    let stdin_prompt = if prompt.ends_with('\n') {
        prompt.to_owned()
    } else {
        format!("{prompt}\n")
    };

    match (provider, external_session_ref) {
        (_, ExternalSessionRef::None) => {
            redesmyn_logging::tracing::warn!(
                provider = provider.as_str(),
                "resume-by-id exec turn requested without an external session ref"
            );
            Err(ResumeByIdExecTurnError::MissingExternalResumeHandle)
        }
        (AgentProvider::Codex, ExternalSessionRef::CodexThread { thread_id, .. }) => {
            Ok(ResumeByIdTurn {
                argv: build_codex_exec_resume_argv(base_argv, thread_id)?,
                stdin_prompt,
            })
        }
        (AgentProvider::ClaudeCode, ExternalSessionRef::ClaudeSession { session_id }) => {
            Ok(ResumeByIdTurn {
                argv: build_claude_exec_resume_argv(base_argv, session_id)?,
                stdin_prompt,
            })
        }
        (AgentProvider::Codex, _) => Err(ResumeByIdExecTurnError::ExternalHandleNotCodexThread),
        (AgentProvider::ClaudeCode, _) => {
            Err(ResumeByIdExecTurnError::ExternalHandleNotClaudeSession)
        }
        (AgentProvider::Shell, _) => Err(ResumeByIdExecTurnError::UnsupportedProvider { provider }),
    }
}

fn argv_token_name(token: &str) -> String {
    let normalized = token.replace('\\', "/");
    normalized
        .rsplit('/')
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase()
}

fn is_codex_token(token: &str) -> bool {
    if token.is_empty() || token.starts_with('-') {
        return false;
    }
    argv_token_name(token) == "codex"
}

fn is_claude_code_token(token: &str) -> bool {
    if token.is_empty() || token.starts_with('-') {
        return false;
    }

    let name = argv_token_name(token);
    if name == "claude" || name == "claude-code" {
        return true;
    }

    let normalized = token.replace('\\', "/").to_ascii_lowercase();
    normalized.ends_with("@anthropic-ai/claude-code")
        || normalized.starts_with("@anthropic-ai/claude-code@")
}

fn wrapper_prefix_len(argv: &[String]) -> usize {
    if argv.is_empty() {
        return 0;
    }

    match argv_token_name(&argv[0]).as_str() {
        "uv" => argv
            .iter()
            .skip(1)
            .position(|t| t == "run")
            .map(|pos| pos + 1 + 1)
            .unwrap_or(1),
        "npx" | "bunx" => 1,
        "npm" => {
            if argv.len() > 1 && (argv[1] == "exec" || argv[1] == "x") {
                2
            } else {
                0
            }
        }
        "pnpm" => {
            if argv.len() > 1 && (argv[1] == "dlx" || argv[1] == "exec") {
                2
            } else {
                0
            }
        }
        "yarn" => {
            if argv.len() > 1 && (argv[1] == "dlx" || argv[1] == "exec" || argv[1] == "run") {
                2
            } else {
                0
            }
        }
        _ => 0,
    }
}

fn find_agent_start_index(argv: &[String], start: usize) -> Option<usize> {
    for (idx, token) in argv.iter().enumerate().skip(start) {
        if token.is_empty() || token.starts_with('-') {
            continue;
        }
        if is_codex_token(token) || is_claude_code_token(token) {
            return Some(idx);
        }
    }
    None
}

fn argv_has_flag_value(argv: &[String], flag: &str, value: &str) -> bool {
    for (idx, token) in argv.iter().enumerate() {
        if token == flag && idx + 1 < argv.len() && argv[idx + 1] == value {
            return true;
        }
        if let Some(rest) = token.strip_prefix(flag) {
            if let Some(eq_value) = rest.strip_prefix('=') {
                if eq_value == value {
                    return true;
                }
            }
        }
    }
    false
}

fn is_codex_structured_exec_invocation(invocation: &AgentInvocation) -> bool {
    let Some(exec_idx) = invocation.agent_argv.iter().position(|t| t == "exec") else {
        return false;
    };
    invocation
        .agent_argv
        .iter()
        .skip(exec_idx + 1)
        .any(|t| t == "--json")
}

fn is_claude_structured_exec_invocation(invocation: &AgentInvocation) -> bool {
    let has_print = invocation.contains_flag("--print") || invocation.contains_flag("-p");
    has_print && argv_has_flag_value(&invocation.agent_argv, "--output-format", "stream-json")
}

fn find_executable_index(argv: &[String], names: &[&str]) -> Option<usize> {
    argv.iter().position(|token| {
        if token.is_empty() || token.starts_with('-') {
            return false;
        }
        let name = argv_token_name(token);
        names.iter().any(|needle| name == *needle)
    })
}

fn codex_exec_option_tokens(tokens: &[String]) -> Vec<String> {
    let mut end = tokens.len();
    if let Some(pos) = tokens.iter().position(|t| t == "resume") {
        end = end.min(pos);
    }
    if let Some(pos) = tokens.iter().position(|t| t == "--") {
        end = end.min(pos + 1);
    }
    if let Some(pos) = tokens.iter().position(|t| t == "-") {
        end = end.min(pos);
    }
    tokens[..end].to_vec()
}

fn build_codex_exec_resume_argv(
    base_argv: &[String],
    thread_id: &str,
) -> Result<Vec<String>, ResumeByIdExecTurnError> {
    let Some(codex_idx) = find_executable_index(base_argv, &["codex"]) else {
        redesmyn_logging::tracing::warn!("unable to find codex executable in argv");
        return Err(ResumeByIdExecTurnError::MissingCodexExecutable);
    };

    let Some(exec_pos) = base_argv
        .iter()
        .skip(codex_idx + 1)
        .position(|t| t == "exec")
    else {
        redesmyn_logging::tracing::warn!("codex argv missing `exec`");
        return Err(ResumeByIdExecTurnError::MissingCodexExec);
    };
    let exec_idx = codex_idx + 1 + exec_pos;

    let rest = &base_argv[exec_idx + 1..];
    let options = codex_exec_option_tokens(rest);
    if !options.iter().any(|t| t == "--json") {
        redesmyn_logging::tracing::warn!("codex argv missing `--json` for structured resume-by-id");
        return Err(ResumeByIdExecTurnError::CodexMissingJsonFlag);
    }

    let mut resumed: Vec<String> = Vec::new();
    resumed.extend_from_slice(&base_argv[..exec_idx + 1]);
    resumed.extend(options);
    resumed.push("resume".to_owned());
    resumed.push(thread_id.to_owned());
    resumed.push("-".to_owned());
    Ok(resumed)
}

fn split_flags_with_values<'a>(
    tokens: &'a [String],
    flags_with_values: &[&str],
) -> (Vec<String>, &'a [String]) {
    let mut options: Vec<String> = Vec::new();

    let mut idx = 0;
    while idx < tokens.len() {
        let token = tokens[idx].as_str();
        if token == "--" {
            options.push(tokens[idx].clone());
            return (options, &tokens[idx + 1..]);
        }
        if token == "-" {
            return (options, &tokens[idx..]);
        }
        if !token.starts_with('-') {
            return (options, &tokens[idx..]);
        }

        options.push(tokens[idx].clone());
        if flags_with_values.contains(&token) && idx + 1 < tokens.len() {
            options.push(tokens[idx + 1].clone());
            idx += 2;
            continue;
        }
        idx += 1;
    }
    (options, &[])
}

fn build_claude_exec_resume_argv(
    base_argv: &[String],
    session_id: &str,
) -> Result<Vec<String>, ResumeByIdExecTurnError> {
    let Some(idx) = find_executable_index(base_argv, &["claude", "claude-code"]) else {
        redesmyn_logging::tracing::warn!("unable to find claude executable in argv");
        return Err(ResumeByIdExecTurnError::MissingClaudeExecutable);
    };

    const CLAUDE_FLAGS_WITH_VALUES: &[&str] = &[
        "--output-format",
        "--input-format",
        "--json-schema",
        "--max-budget-usd",
        "--allowedTools",
        "--allowed-tools",
        "--tools",
        "--disallowedTools",
        "--disallowed-tools",
        "--mcp-config",
        "--system-prompt",
        "--append-system-prompt",
        "--permission-mode",
        "--model",
        "--agent",
        "--betas",
        "--fallback-model",
        "--settings",
        "--add-dir",
        "--session-id",
        "--agents",
        "--setting-sources",
        "--plugin-dir",
    ];

    let tokens_after = &base_argv[idx + 1..];
    let (options, _) = split_flags_with_values(tokens_after, CLAUDE_FLAGS_WITH_VALUES);

    // Drop continuation flags; we are explicitly resuming by id.
    let mut cleaned: Vec<String> = Vec::with_capacity(options.len());
    let mut skip_next = false;
    for token in options {
        if skip_next {
            skip_next = false;
            continue;
        }

        if token == "-c" || token == "--continue" {
            continue;
        }
        if token == "-r" || token == "--resume" {
            skip_next = true;
            continue;
        }
        if token.starts_with("--resume=") {
            continue;
        }

        cleaned.push(token);
    }

    if !cleaned.iter().any(|t| t == "--print" || t == "-p") {
        redesmyn_logging::tracing::warn!(
            "claude argv missing `--print` for structured resume-by-id"
        );
        return Err(ResumeByIdExecTurnError::ClaudeMissingPrintFlag);
    }
    if !argv_has_flag_value(&cleaned, "--output-format", "stream-json") {
        redesmyn_logging::tracing::warn!(
            "claude argv missing `--output-format stream-json` for structured resume-by-id"
        );
        return Err(ResumeByIdExecTurnError::ClaudeMissingStreamJsonOutput);
    }

    let insert_at = cleaned
        .iter()
        .position(|t| t == "--print" || t == "-p")
        .map(|idx| idx + 1)
        .unwrap_or(cleaned.len());

    let mut resumed = Vec::new();
    resumed.extend_from_slice(&base_argv[..idx + 1]);
    resumed.extend_from_slice(&cleaned[..insert_at]);
    resumed.push("--resume".to_owned());
    resumed.push(session_id.to_owned());
    resumed.extend_from_slice(&cleaned[insert_at..]);
    Ok(resumed)
}

#[cfg(test)]
mod tests {
    use super::{
        AgentInvocation, AgentProvider, AgentProviderSelection, AgentRuntimeKind,
        AppServerTurnIntent, ExternalSessionRef, ResumeByIdExecTurnError,
        build_app_server_turn_intent, build_resume_by_id_exec_turn, infer_agent_provider_from_argv,
        infer_agent_runtime_kind, resolve_agent_provider,
    };

    fn argv(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|part| (*part).to_owned()).collect()
    }

    #[test]
    fn invocation_splits_wrapper_and_agent_argv() {
        let invocation = AgentInvocation::from_argv(&argv(&[
            "uv", "run", "--python", "3.12", "codex", "exec", "--json",
        ]));
        assert_eq!(invocation.wrapper, argv(&["uv", "run", "--python", "3.12"]));
        assert_eq!(invocation.agent_argv, argv(&["codex", "exec", "--json"]));
    }

    #[test]
    fn infer_agent_provider_avoids_false_positives() {
        assert_eq!(
            infer_agent_provider_from_argv(&argv(&["echo", "codex"])),
            AgentProvider::Shell
        );
    }

    #[test]
    fn infer_agent_provider_direct() {
        assert_eq!(
            infer_agent_provider_from_argv(&argv(&["codex", "exec", "--json"])),
            AgentProvider::Codex
        );
        assert_eq!(
            infer_agent_provider_from_argv(&argv(&["claude", "--print"])),
            AgentProvider::ClaudeCode
        );
        assert_eq!(
            infer_agent_provider_from_argv(&argv(&["claude-code", "--print"])),
            AgentProvider::ClaudeCode
        );
    }

    #[test]
    fn infer_agent_provider_scans_wrappers() {
        assert_eq!(
            infer_agent_provider_from_argv(&argv(&["uv", "run", "codex", "exec", "--json"])),
            AgentProvider::Codex
        );
        assert_eq!(
            infer_agent_provider_from_argv(&argv(&["npx", "@anthropic-ai/claude-code", "--print"])),
            AgentProvider::ClaudeCode
        );
        assert_eq!(
            infer_agent_provider_from_argv(&argv(&["npm", "exec", "codex", "exec"])),
            AgentProvider::Codex
        );
    }

    #[test]
    fn resolve_agent_provider_auto_prefers_external_session_ref() {
        assert_eq!(
            resolve_agent_provider(
                AgentProviderSelection::Auto,
                &argv(&["echo", "codex"]),
                &ExternalSessionRef::CodexThread {
                    thread_id: "th_1".to_owned(),
                    turn_id: None,
                }
            ),
            AgentProvider::Codex
        );
    }

    #[test]
    fn infer_runtime_kind_codex_app_server() {
        let invocation = AgentInvocation::from_argv(&argv(&["codex", "app-server"]));
        assert_eq!(
            infer_agent_runtime_kind(&invocation, AgentProvider::Codex),
            AgentRuntimeKind::AppServer
        );
    }

    #[test]
    fn infer_runtime_kind_structured_exec_markers() {
        let invocation = AgentInvocation::from_argv(&argv(&["codex", "exec", "--json"]));
        assert_eq!(
            infer_agent_runtime_kind(&invocation, AgentProvider::Codex),
            AgentRuntimeKind::StructuredExec
        );

        let invocation = AgentInvocation::from_argv(&argv(&[
            "claude",
            "--print",
            "--output-format",
            "stream-json",
        ]));
        assert_eq!(
            infer_agent_runtime_kind(&invocation, AgentProvider::ClaudeCode),
            AgentRuntimeKind::StructuredExec
        );
    }

    #[test]
    fn infer_runtime_kind_wrapper_cases() {
        let invocation =
            AgentInvocation::from_argv(&argv(&["uv", "run", "codex", "app-server", "--port", "0"]));
        assert_eq!(
            infer_agent_runtime_kind(&invocation, AgentProvider::Codex),
            AgentRuntimeKind::AppServer
        );
    }

    #[test]
    fn build_app_server_turn_intent_start_new_vs_resume() {
        assert_eq!(
            build_app_server_turn_intent("hi", None),
            AppServerTurnIntent::StartNew {
                prompt: "hi".to_owned()
            }
        );
        assert_eq!(
            build_app_server_turn_intent("hi", Some(&ExternalSessionRef::None)),
            AppServerTurnIntent::StartNew {
                prompt: "hi".to_owned()
            }
        );

        let external = ExternalSessionRef::CodexThread {
            thread_id: "th_123".to_owned(),
            turn_id: None,
        };
        assert_eq!(
            build_app_server_turn_intent("go", Some(&external)),
            AppServerTurnIntent::Resume {
                external,
                prompt: "go".to_owned()
            }
        );
    }

    #[test]
    fn build_resume_by_id_exec_turn_codex_exec_resume_stdin() {
        let turn = build_resume_by_id_exec_turn(
            &argv(&["codex", "exec", "--json"]),
            AgentProvider::Codex,
            &ExternalSessionRef::CodexThread {
                thread_id: "th_123".to_owned(),
                turn_id: None,
            },
            "hello",
        )
        .unwrap();

        assert_eq!(
            turn.argv,
            argv(&["codex", "exec", "--json", "resume", "th_123", "-"])
        );
        assert_eq!(turn.stdin_prompt, "hello\n");
    }

    #[test]
    fn build_resume_by_id_exec_turn_codex_drops_stdin_marker_from_base_argv() {
        let turn = build_resume_by_id_exec_turn(
            &argv(&["codex", "exec", "--json", "-"]),
            AgentProvider::Codex,
            &ExternalSessionRef::CodexThread {
                thread_id: "th_123".to_owned(),
                turn_id: None,
            },
            "hello",
        )
        .unwrap();

        assert_eq!(
            turn.argv,
            argv(&["codex", "exec", "--json", "resume", "th_123", "-"])
        );
        assert!(!turn.argv.windows(2).any(|w| w == ["-", "resume"]));
    }

    #[test]
    fn build_resume_by_id_exec_turn_codex_preserves_unknown_flag_values() {
        let turn = build_resume_by_id_exec_turn(
            &argv(&["codex", "exec", "--json", "--new-flag", "value", "-"]),
            AgentProvider::Codex,
            &ExternalSessionRef::CodexThread {
                thread_id: "th_abc".to_owned(),
                turn_id: None,
            },
            "go",
        )
        .unwrap();

        assert_eq!(
            turn.argv,
            argv(&[
                "codex",
                "exec",
                "--json",
                "--new-flag",
                "value",
                "resume",
                "th_abc",
                "-",
            ])
        );
    }

    #[test]
    fn build_resume_by_id_exec_turn_codex_preserves_exec_options_with_wrapper() {
        let turn = build_resume_by_id_exec_turn(
            &argv(&[
                "uv",
                "run",
                "codex",
                "exec",
                "--json",
                "-C",
                "/tmp/repo",
                "--color",
                "never",
            ]),
            AgentProvider::Codex,
            &ExternalSessionRef::CodexThread {
                thread_id: "th_abc".to_owned(),
                turn_id: Some("turn_1".to_owned()),
            },
            "go",
        )
        .unwrap();

        assert_eq!(
            turn.argv,
            argv(&[
                "uv",
                "run",
                "codex",
                "exec",
                "--json",
                "-C",
                "/tmp/repo",
                "--color",
                "never",
                "resume",
                "th_abc",
                "-",
            ])
        );
    }

    #[test]
    fn build_resume_by_id_exec_turn_claude_resume_inserts_after_print() {
        let turn = build_resume_by_id_exec_turn(
            &argv(&["claude", "--print", "--output-format", "stream-json"]),
            AgentProvider::ClaudeCode,
            &ExternalSessionRef::ClaudeSession {
                session_id: "sess_123".to_owned(),
            },
            "fix conflicts",
        )
        .unwrap();

        assert_eq!(
            turn.argv,
            argv(&[
                "claude",
                "--print",
                "--resume",
                "sess_123",
                "--output-format",
                "stream-json"
            ])
        );
    }

    #[test]
    fn build_resume_by_id_exec_turn_claude_drops_continue_and_replaces_resume() {
        let turn = build_resume_by_id_exec_turn(
            &argv(&[
                "claude",
                "--print",
                "--output-format",
                "stream-json",
                "--continue",
                "--resume",
                "old",
            ]),
            AgentProvider::ClaudeCode,
            &ExternalSessionRef::ClaudeSession {
                session_id: "sess_new".to_owned(),
            },
            "go",
        )
        .unwrap();

        assert!(!turn.argv.iter().any(|t| t == "--continue"));
        assert_eq!(turn.argv.iter().filter(|t| *t == "--resume").count(), 1);
        let resume_idx = turn.argv.iter().position(|t| t == "--resume").unwrap();
        assert_eq!(turn.argv[resume_idx + 1], "sess_new");
    }

    #[test]
    fn build_resume_by_id_exec_turn_requires_external_session_ref() {
        let err = build_resume_by_id_exec_turn(
            &argv(&["codex", "exec", "--json"]),
            AgentProvider::Codex,
            &ExternalSessionRef::None,
            "hello",
        )
        .unwrap_err();

        assert!(matches!(
            err,
            ResumeByIdExecTurnError::MissingExternalResumeHandle
        ));
        assert!(err.to_string().contains("Missing external resume handle"));
    }

    #[test]
    fn build_resume_by_id_exec_turn_claude_requires_print_mode() {
        let err = build_resume_by_id_exec_turn(
            &argv(&["claude", "--output-format", "stream-json"]),
            AgentProvider::ClaudeCode,
            &ExternalSessionRef::ClaudeSession {
                session_id: "sess_123".to_owned(),
            },
            "hi",
        )
        .unwrap_err();

        assert!(matches!(
            err,
            ResumeByIdExecTurnError::ClaudeMissingPrintFlag
        ));
        assert!(err.to_string().contains("must include `--print`"));
    }

    #[test]
    fn build_resume_by_id_exec_turn_codex_requires_json_flag() {
        let err = build_resume_by_id_exec_turn(
            &argv(&["codex", "exec"]),
            AgentProvider::Codex,
            &ExternalSessionRef::CodexThread {
                thread_id: "th_123".to_owned(),
                turn_id: None,
            },
            "hello",
        )
        .unwrap_err();

        assert!(matches!(err, ResumeByIdExecTurnError::CodexMissingJsonFlag));
        assert!(err.to_string().contains("must include `--json`"));
    }
}
