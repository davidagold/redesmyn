use std::env;

#[derive(Clone, Debug)]
pub struct LoggingConfig {
    pub format: LogFormat,
    pub filter_directives: Option<String>,
    pub payload_policy: PayloadPolicy,
}

impl LoggingConfig {
    pub fn from_env() -> Self {
        Self {
            format: LogFormat::from_env(),
            filter_directives: filter_directives_from_env(),
            payload_policy: PayloadPolicy::from_env(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LogFormat {
    Pretty,
    Json,
}

impl LogFormat {
    fn from_env() -> Self {
        let Ok(raw) = env::var("REDESMYN_LOG_FORMAT") else {
            return Self::Pretty;
        };

        match raw.trim().to_ascii_lowercase().as_str() {
            "pretty" | "human" | "dev" => Self::Pretty,
            "json" => Self::Json,
            _ => Self::Pretty,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PayloadPolicy {
    /// Do not emit payload contents by default.
    MetadataOnly,
    /// Emit a stable placeholder (useful when debugging routing without leaking contents).
    Redact,
    /// Emit a truncated `Debug` representation.
    Sampled { max_bytes: usize },
    /// Emit the full `Debug` representation (use sparingly).
    Full,
}

impl Default for PayloadPolicy {
    fn default() -> Self {
        Self::MetadataOnly
    }
}

impl PayloadPolicy {
    pub fn from_env() -> Self {
        let Ok(raw) = env::var("REDESMYN_LOG_PAYLOADS") else {
            return Self::MetadataOnly;
        };

        let raw = raw.trim().to_ascii_lowercase();
        if raw.is_empty() {
            return Self::MetadataOnly;
        }

        match raw.as_str() {
            "0" | "off" | "false" | "metadata" | "metadataonly" => Self::MetadataOnly,
            "redact" | "redacted" => Self::Redact,
            "1" | "on" | "true" | "full" => Self::Full,
            "sample" | "sampled" => Self::Sampled { max_bytes: 2048 },
            _ => {
                if let Some(rest) = raw.strip_prefix("sample:") {
                    if let Ok(max_bytes) = rest.parse::<usize>() {
                        return Self::Sampled { max_bytes };
                    }
                }
                Self::MetadataOnly
            }
        }
    }
}

fn filter_directives_from_env() -> Option<String> {
    if let Ok(filter) = env::var("REDESMYN_LOG") {
        return Some(filter);
    }
    env::var("RUST_LOG").ok()
}
