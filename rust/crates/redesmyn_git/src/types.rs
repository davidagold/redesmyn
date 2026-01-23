use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Debug, Clone)]
pub struct ParseGitOidError {
    input: String,
}

impl ParseGitOidError {
    fn new(input: impl Into<String>) -> Self {
        Self {
            input: input.into(),
        }
    }
}

impl fmt::Display for ParseGitOidError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid git object id: {}", self.input)
    }
}

impl std::error::Error for ParseGitOidError {}

/// A full git object id (SHA-1 or SHA-256 hex).
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct GitOid(String);

impl GitOid {
    pub fn parse(value: &str) -> Result<Self, ParseGitOidError> {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err(ParseGitOidError::new(value));
        }

        let len = trimmed.len();
        let valid_len = len == 40 || len == 64;
        let valid_hex = trimmed.bytes().all(|b| b.is_ascii_hexdigit());
        if !valid_len || !valid_hex {
            return Err(ParseGitOidError::new(value));
        }

        Ok(Self(trimmed.to_owned()))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for GitOid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GitOid({})", self.0)
    }
}

impl fmt::Display for GitOid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl FromStr for GitOid {
    type Err = ParseGitOidError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

/// A git revision expression (branch name, tag, SHA, etc).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GitRevision(String);

impl GitRevision {
    pub fn new(value: impl Into<String>) -> Result<Self, InvalidGitRevisionError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(InvalidGitRevisionError { input: value });
        }
        if value.as_bytes().iter().any(|b| *b == 0) {
            return Err(InvalidGitRevisionError { input: value });
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for GitRevision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone)]
pub struct InvalidGitRevisionError {
    input: String,
}

impl fmt::Display for InvalidGitRevisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid git revision: {}", self.input)
    }
}

impl std::error::Error for InvalidGitRevisionError {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GitRefName(String);

impl GitRefName {
    pub fn new(value: impl Into<String>) -> Result<Self, InvalidGitRefNameError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(InvalidGitRefNameError { input: value });
        }
        if value.as_bytes().iter().any(|b| *b == 0) {
            return Err(InvalidGitRefNameError { input: value });
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for GitRefName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone)]
pub struct InvalidGitRefNameError {
    input: String,
}

impl fmt::Display for InvalidGitRefNameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid git ref name: {}", self.input)
    }
}

impl std::error::Error for InvalidGitRefNameError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GitWorktreeTarget {
    Head,
    Revision(GitRevision),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitWorktree {
    pub path: PathBuf,
    pub head: GitOid,
    pub branch: Option<GitRefName>,
    pub detached: bool,
}
