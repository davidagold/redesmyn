/// Typed daemon capability set (T-23).
///
/// Wire format: stable strings in `DaemonHello.capabilities` (T-11).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DaemonCapabilities {
    pub supports_repo_execution: bool,
    pub supports_worktrees: bool,
    pub supports_git_observation: bool,
    pub supports_session_exec: bool,
    pub supports_session_attach_tmux: bool,
    pub supports_artifacts: bool,
}

impl DaemonCapabilities {
    pub const REPO_EXECUTION: &'static str = "repo_execution";
    pub const WORKTREES: &'static str = "worktrees";
    pub const GIT_OBSERVATION: &'static str = "git_observation";
    pub const SESSION_EXEC: &'static str = "session_exec";
    pub const SESSION_ATTACH_TMUX: &'static str = "session_attach_tmux";
    pub const ARTIFACTS: &'static str = "artifacts";

    #[must_use]
    pub fn to_wire_strings(self) -> Vec<String> {
        let mut caps = Vec::new();
        if self.supports_repo_execution {
            caps.push(Self::REPO_EXECUTION.to_string());
        }
        if self.supports_worktrees {
            caps.push(Self::WORKTREES.to_string());
        }
        if self.supports_git_observation {
            caps.push(Self::GIT_OBSERVATION.to_string());
        }
        if self.supports_session_exec {
            caps.push(Self::SESSION_EXEC.to_string());
        }
        if self.supports_session_attach_tmux {
            caps.push(Self::SESSION_ATTACH_TMUX.to_string());
        }
        if self.supports_artifacts {
            caps.push(Self::ARTIFACTS.to_string());
        }
        caps
    }
}

