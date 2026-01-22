//! Incremental Codex output parser (T-33).
//!
//! This is a pure, deterministic state machine:
//! - Prefer structured JSONL output (`codex exec --json`).
//! - Fall back to prompt-tail heuristics for interactive sessions.

use std::time::Instant;

use crate::agent::ExternalSessionRef;

const DEFAULT_MAX_BUSY_S: f64 = 15.0 * 60.0;
const MAX_TEXT_TAIL_CHARS: usize = 4000;
const PROMPT_SCAN_TAIL_CHARS: usize = 400;
const MAX_JSONL_BUFFER_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodexTurnState {
    Unknown,
    Ready,
    Busy,
    Completed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodexTurnStatus {
    pub turn_state: CodexTurnState,
    pub detail: Option<String>,
}

impl Default for CodexTurnStatus {
    fn default() -> Self {
        Self {
            turn_state: CodexTurnState::Unknown,
            detail: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CodexParserCapabilities {
    pub can_stream_semantic_events: bool,
    pub can_detect_turn_complete: bool,
    pub can_resume_by_id: bool,
}

impl Default for CodexParserCapabilities {
    fn default() -> Self {
        Self {
            can_stream_semantic_events: false,
            can_detect_turn_complete: false,
            can_resume_by_id: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodexParserEvent {
    TurnStarted,
    TurnCompleted,
    AssistantMessage { text: String },
}

pub struct CodexParseOutput<'a> {
    pub events: Vec<CodexParserEvent>,
    pub capabilities: CodexParserCapabilities,
    pub status: &'a CodexTurnStatus,
    pub external_session_ref: &'a ExternalSessionRef,
}

#[derive(Debug, Default, serde::Deserialize)]
struct CodexJsonlItem {
    #[serde(default)]
    r#type: Option<String>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    command: Option<String>,
}

#[derive(Debug, Default, serde::Deserialize)]
struct CodexJsonlEvent {
    #[serde(rename = "type")]
    event_type: String,
    #[serde(default, alias = "threadId")]
    thread_id: Option<String>,
    #[serde(default, alias = "turnId")]
    turn_id: Option<String>,
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    item: Option<CodexJsonlItem>,
}

/// Pure Codex stdout parser.
pub struct CodexOutputParser {
    clock_s: Box<dyn Fn() -> f64 + Send + Sync>,
    max_busy_s: f64,

    capabilities: CodexParserCapabilities,
    status: CodexTurnStatus,
    external_session_ref: ExternalSessionRef,

    jsonl_buffer: String,
    text_tail: String,
    saw_structured_events: bool,

    output_since_prompt: bool,
    saw_prompt_once: bool,
    busy_since_s: Option<f64>,

    logged_structured_stream: bool,
}

impl Default for CodexOutputParser {
    fn default() -> Self {
        Self::new()
    }
}

impl CodexOutputParser {
    #[must_use]
    pub fn new() -> Self {
        let start = Instant::now();
        Self::with_clock(move || start.elapsed().as_secs_f64())
    }

    #[must_use]
    pub fn with_clock(clock_s: impl Fn() -> f64 + Send + Sync + 'static) -> Self {
        Self::with_clock_and_max_busy(clock_s, DEFAULT_MAX_BUSY_S)
    }

    #[must_use]
    pub fn with_clock_and_max_busy(
        clock_s: impl Fn() -> f64 + Send + Sync + 'static,
        max_busy_s: f64,
    ) -> Self {
        Self {
            clock_s: Box::new(clock_s),
            max_busy_s,
            capabilities: CodexParserCapabilities::default(),
            status: CodexTurnStatus::default(),
            external_session_ref: ExternalSessionRef::None,
            jsonl_buffer: String::new(),
            text_tail: String::new(),
            saw_structured_events: false,
            output_since_prompt: false,
            saw_prompt_once: false,
            busy_since_s: None,
            logged_structured_stream: false,
        }
    }

    #[must_use]
    pub fn capabilities(&self) -> CodexParserCapabilities {
        self.capabilities
    }

    #[must_use]
    pub fn status(&self) -> &CodexTurnStatus {
        &self.status
    }

    #[must_use]
    pub fn external_session_ref(&self) -> &ExternalSessionRef {
        &self.external_session_ref
    }

    pub fn seed_external_session_ref(&mut self, external_session_ref: ExternalSessionRef) {
        self.set_external_session_ref(external_session_ref);
    }

    #[must_use]
    pub fn consume_output(&mut self, text: &str) -> CodexParseOutput<'_> {
        let events = self.consume_output_events(text);
        CodexParseOutput {
            events,
            capabilities: self.capabilities,
            status: &self.status,
            external_session_ref: &self.external_session_ref,
        }
    }

    fn consume_output_events(&mut self, text: &str) -> Vec<CodexParserEvent> {
        // Empty text is treated as a tick for timeouts.
        if text.is_empty() {
            self.maybe_timeout();
            return Vec::new();
        }

        if self.jsonl_buffer.len() > MAX_JSONL_BUFFER_BYTES {
            redesmyn_logging::tracing::warn!(
                buffer_len = self.jsonl_buffer.len(),
                "codex jsonl buffer exceeded limit; clearing"
            );
            self.jsonl_buffer.clear();
        }

        let mut saw_structured_in_chunk = false;
        let mut emitted: Vec<CodexParserEvent> = Vec::new();

        self.jsonl_buffer.push_str(text);
        if let Some(last_newline) = self.jsonl_buffer.rfind('\n') {
            let remainder = self.jsonl_buffer.split_off(last_newline + 1);
            let complete = std::mem::replace(&mut self.jsonl_buffer, remainder);

            for raw_line in complete.split('\n') {
                let line = raw_line.trim();
                if line.is_empty() || !line.starts_with('{') || !line.ends_with('}') {
                    continue;
                }

                emitted.extend(self.consume_structured_line(line));
                if self.saw_structured_events {
                    saw_structured_in_chunk = true;
                }
            }
        }

        let cleaned = strip_ansi(text);
        self.consume_text_for_external_ids(&cleaned);

        // In structured exec mode, prompt heuristics are actively harmful: the
        // JSON stream itself is the semantic signal.
        if saw_structured_in_chunk && cleaned.trim_start().starts_with('{') {
            return emitted;
        }

        self.text_tail.push_str(&cleaned);
        truncate_to_tail_chars(&mut self.text_tail, MAX_TEXT_TAIL_CHARS);

        if !cleaned.trim().is_empty() {
            self.output_since_prompt = true;
        }

        if tail_looks_like_codex_prompt(&self.text_tail) {
            self.note_prompt();
        } else {
            if self.output_since_prompt
                && (matches!(
                    self.status.turn_state,
                    CodexTurnState::Ready | CodexTurnState::Completed
                ) || self.saw_prompt_once)
            {
                self.note_busy(None);
            }
            self.maybe_timeout();
        }

        emitted
    }

    fn note_saw_structured_event(&mut self) {
        if !self.saw_structured_events {
            self.saw_structured_events = true;
        }
        if !self.capabilities.can_stream_semantic_events {
            self.capabilities.can_stream_semantic_events = true;
        }
        if !self.logged_structured_stream {
            self.logged_structured_stream = true;
            redesmyn_logging::tracing::info!("codex parser detected structured jsonl stream");
        }
    }

    fn consume_structured_line(&mut self, line: &str) -> Vec<CodexParserEvent> {
        let event: CodexJsonlEvent = match serde_json::from_str(line) {
            Ok(event) => event,
            Err(_) => return Vec::new(),
        };

        let event_type = event.event_type.trim().to_owned();
        if event_type.is_empty() {
            return Vec::new();
        }

        self.note_saw_structured_event();

        match event_type.as_str() {
            "thread.started" => {
                if let Some(thread_id) = event.thread_id {
                    let preserved_turn_id = match &self.external_session_ref {
                        ExternalSessionRef::CodexThread { turn_id, .. } => turn_id.clone(),
                        _ => None,
                    };
                    self.set_external_session_ref(ExternalSessionRef::CodexThread {
                        thread_id,
                        turn_id: preserved_turn_id,
                    });
                }
                Vec::new()
            }
            "turn.started" => {
                if !self.capabilities.can_detect_turn_complete {
                    self.capabilities.can_detect_turn_complete = true;
                }

                if let (Some(turn_id), ExternalSessionRef::CodexThread { thread_id, .. }) =
                    (event.turn_id, &self.external_session_ref)
                {
                    self.set_external_session_ref(ExternalSessionRef::CodexThread {
                        thread_id: thread_id.clone(),
                        turn_id: Some(turn_id),
                    });
                }

                self.note_busy(None);
                vec![CodexParserEvent::TurnStarted]
            }
            "turn.completed" => {
                if !self.capabilities.can_detect_turn_complete {
                    self.capabilities.can_detect_turn_complete = true;
                }

                if let (Some(turn_id), ExternalSessionRef::CodexThread { thread_id, .. }) =
                    (event.turn_id, &self.external_session_ref)
                {
                    self.set_external_session_ref(ExternalSessionRef::CodexThread {
                        thread_id: thread_id.clone(),
                        turn_id: Some(turn_id),
                    });
                }

                self.set_turn_state(CodexTurnState::Completed, None);
                self.busy_since_s = None;
                self.output_since_prompt = false;
                vec![CodexParserEvent::TurnCompleted]
            }
            _ if event_type.starts_with("item.") => {
                self.consume_structured_item_event(&event_type, event)
            }
            _ => self.consume_structured_message_event(&event_type, event),
        }
    }

    fn consume_structured_item_event(
        &mut self,
        event_type: &str,
        event: CodexJsonlEvent,
    ) -> Vec<CodexParserEvent> {
        let item = event.item;
        let item_type = item
            .as_ref()
            .and_then(|item| item.r#type.as_deref())
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();

        match (event_type, item.as_ref()) {
            ("item.started", Some(item)) => {
                let mut detail: Option<String> = None;
                if item_type == "command_execution" {
                    let command = item.command.as_deref().unwrap_or_default().trim();
                    if !command.is_empty() {
                        detail = Some(format!("Running: {}", truncate_to_chars(command, 200)));
                    }
                }

                let was_busy = self.status.turn_state == CodexTurnState::Busy;
                self.note_busy(detail);
                if was_busy {
                    Vec::new()
                } else {
                    vec![CodexParserEvent::TurnStarted]
                }
            }
            ("item.completed", Some(item))
                if matches!(
                    item_type.as_str(),
                    "reasoning" | "agent_message" | "assistant_message"
                ) =>
            {
                let text_value = item
                    .text
                    .as_deref()
                    .or(item.message.as_deref())
                    .or(item.content.as_deref())
                    .unwrap_or_default()
                    .trim()
                    .to_owned();

                let mut emitted = Vec::new();
                if !text_value.is_empty() {
                    emitted.push(CodexParserEvent::AssistantMessage { text: text_value });
                    self.output_since_prompt = true;
                }
                self.note_busy(None);
                emitted
            }
            _ => {
                self.note_busy(None);
                Vec::new()
            }
        }
    }

    fn consume_structured_message_event(
        &mut self,
        event_type: &str,
        event: CodexJsonlEvent,
    ) -> Vec<CodexParserEvent> {
        let message_type = event_type.trim().to_ascii_lowercase();
        if !matches!(
            message_type.as_str(),
            "assistant.message" | "assistant_message" | "assistant" | "message"
        ) {
            return Vec::new();
        }

        let role = event
            .role
            .as_deref()
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        if message_type == "message" && role != "assistant" {
            return Vec::new();
        }

        let text_value = event.text.or(event.message).or(event.content);
        match text_value {
            Some(text) if !text.trim().is_empty() => {
                vec![CodexParserEvent::AssistantMessage { text }]
            }
            _ => Vec::new(),
        }
    }

    fn consume_text_for_external_ids(&mut self, text: &str) {
        let mut thread_id = extract_external_id(text, ExternalIdKey::Thread);
        let turn_id = extract_external_id(text, ExternalIdKey::Turn);

        let existing_codex = match &self.external_session_ref {
            ExternalSessionRef::CodexThread { thread_id, turn_id } => Some((thread_id, turn_id)),
            _ => None,
        };

        if thread_id.is_none() {
            if let Some((existing_thread_id, _)) = existing_codex {
                thread_id = Some(existing_thread_id.clone());
            }
        }

        let Some(thread_id) = thread_id else {
            return;
        };

        let resolved_turn_id = match turn_id {
            Some(turn_id) => Some(turn_id),
            None => existing_codex.and_then(|(_, turn_id)| turn_id.clone()),
        };

        self.set_external_session_ref(ExternalSessionRef::CodexThread {
            thread_id,
            turn_id: resolved_turn_id,
        });
    }

    fn set_external_session_ref(&mut self, external_session_ref: ExternalSessionRef) {
        let can_resume_by_id =
            matches!(external_session_ref, ExternalSessionRef::CodexThread { .. });
        if self.capabilities.can_resume_by_id != can_resume_by_id {
            self.capabilities.can_resume_by_id = can_resume_by_id;
        }
        self.external_session_ref = external_session_ref;
    }

    fn set_turn_state(&mut self, turn_state: CodexTurnState, detail: Option<String>) {
        if self.status.turn_state == turn_state && self.status.detail == detail {
            return;
        }
        self.status.turn_state = turn_state;
        self.status.detail = detail;
    }

    fn note_busy(&mut self, detail: Option<String>) {
        if self.busy_since_s.is_none() {
            self.busy_since_s = Some((self.clock_s)());
        }
        self.set_turn_state(CodexTurnState::Busy, detail);
    }

    fn note_prompt(&mut self) {
        let completed = self.status.turn_state == CodexTurnState::Busy
            || self.busy_since_s.is_some()
            || (self.saw_prompt_once && self.output_since_prompt);
        self.set_turn_state(
            if completed {
                CodexTurnState::Completed
            } else {
                CodexTurnState::Ready
            },
            None,
        );
        self.saw_prompt_once = true;
        self.output_since_prompt = false;
        self.busy_since_s = None;
    }

    fn maybe_timeout(&mut self) {
        let Some(busy_since_s) = self.busy_since_s else {
            return;
        };

        let now_s = (self.clock_s)();
        if now_s - busy_since_s < self.max_busy_s {
            return;
        }

        self.busy_since_s = None;
        self.output_since_prompt = false;
        self.set_turn_state(
            CodexTurnState::Unknown,
            Some("Timeout waiting for Codex turn completion; signals unavailable.".to_owned()),
        );
        redesmyn_logging::tracing::warn!("codex parser timed out waiting for turn completion");
    }
}

fn truncate_to_tail_chars(s: &mut String, max_chars: usize) {
    if s.chars().count() <= max_chars {
        return;
    }

    let keep_from = s
        .char_indices()
        .nth(s.chars().count().saturating_sub(max_chars))
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    s.drain(..keep_from);
}

fn truncate_to_chars(s: &str, max_chars: usize) -> String {
    let mut out = String::new();
    out.extend(s.chars().take(max_chars));
    out
}

fn tail_looks_like_codex_prompt(tail: &str) -> bool {
    let mut trimmed: String = tail
        .chars()
        .rev()
        .take(PROMPT_SCAN_TAIL_CHARS)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    trimmed.retain(|c| c != '\0');
    while trimmed.ends_with(' ') || trimmed.ends_with('\t') {
        trimmed.pop();
    }

    let candidate = trimmed.trim_end_matches(|c: char| c.is_whitespace());
    if candidate.is_empty() {
        return false;
    }

    let mut chars: Vec<char> = candidate.chars().collect();
    while let Some(last) = chars.last().copied() {
        if last.is_whitespace() {
            chars.pop();
            continue;
        }
        break;
    }
    if chars.is_empty() {
        return false;
    }

    let last = *chars.last().unwrap();
    if last == '>' {
        let idx = chars.len() - 1;
        return idx == 0 || matches!(chars[idx - 1], '\n' | '\r');
    }

    if last == '_' || last == '▌' {
        if chars.len() < 2 {
            return false;
        }
        let marker_idx = chars.len() - 1;
        let prompt_idx = marker_idx - 1;
        if chars[prompt_idx] != '>' {
            return false;
        }
        return prompt_idx == 0 || matches!(chars[prompt_idx - 1], '\n' | '\r');
    }

    false
}

fn strip_ansi(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut last_keep = 0;
    let mut idx = 0;

    while idx < bytes.len() {
        if bytes[idx] != 0x1b {
            idx += 1;
            continue;
        }

        let Some(next) = bytes.get(idx + 1).copied() else {
            idx += 1;
            continue;
        };

        match next {
            b'[' => {
                let mut end = idx + 2;
                while end < bytes.len() && (0x30..=0x3f).contains(&bytes[end]) {
                    end += 1;
                }
                while end < bytes.len() && (0x20..=0x2f).contains(&bytes[end]) {
                    end += 1;
                }
                if end < bytes.len() && (0x40..=0x7e).contains(&bytes[end]) {
                    end += 1;
                    out.push_str(&text[last_keep..idx]);
                    last_keep = end;
                    idx = end;
                    continue;
                }
            }
            b']' => {
                let mut end = idx + 2;
                let mut consumed: Option<usize> = None;
                while end < bytes.len() {
                    if bytes[end] == 0x07 {
                        consumed = Some(end + 1);
                        break;
                    }
                    if bytes[end] == 0x1b && bytes.get(end + 1) == Some(&b'\\') {
                        consumed = Some(end + 2);
                        break;
                    }
                    end += 1;
                }
                if let Some(end) = consumed {
                    out.push_str(&text[last_keep..idx]);
                    last_keep = end;
                    idx = end;
                    continue;
                }
            }
            _ => {}
        }

        idx += 1;
    }

    out.push_str(&text[last_keep..]);
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExternalIdKey {
    Thread,
    Turn,
}

fn extract_external_id(text: &str, key: ExternalIdKey) -> Option<String> {
    let needle = match key {
        ExternalIdKey::Thread => "thread",
        ExternalIdKey::Turn => "turn",
    };

    let bytes = text.as_bytes();
    let mut idx = 0;
    while idx + needle.len() <= bytes.len() {
        if bytes.get(idx..idx + needle.len()) != Some(needle.as_bytes()) {
            idx += 1;
            continue;
        }

        if idx > 0 && is_word_byte(bytes[idx - 1]) {
            idx += 1;
            continue;
        }

        let mut end = idx + needle.len();
        if let Some(sep) = bytes.get(end).copied() {
            if sep == b'-' || sep == b'_' || sep == b' ' {
                end += 1;
            }
        }

        if end + 2 > bytes.len() || bytes.get(end..end + 2) != Some(b"id") {
            idx += 1;
            continue;
        }
        end += 2;

        if end < bytes.len() && is_word_byte(bytes[end]) {
            idx += 1;
            continue;
        }

        let mut cursor = end;
        while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }

        let Some(assign) = bytes.get(cursor).copied() else {
            idx += 1;
            continue;
        };
        if assign != b':' && assign != b'=' {
            idx += 1;
            continue;
        }
        cursor += 1;
        while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }

        let start_val = cursor;
        while let Some(b) = bytes.get(cursor).copied() {
            let ok = (b'A'..=b'Z').contains(&b)
                || (b'a'..=b'z').contains(&b)
                || (b'0'..=b'9').contains(&b)
                || b == b'_'
                || b == b'-';
            if !ok {
                break;
            }
            cursor += 1;
        }
        if cursor > start_val {
            return Some(text[start_val..cursor].to_owned());
        }

        idx += 1;
    }

    None
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b >= 0x80
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::{CodexOutputParser, CodexParserEvent, CodexTurnState};
    use crate::agent::ExternalSessionRef;

    #[test]
    fn codex_parser_parses_structured_turn_events() {
        let now_s = Arc::new(Mutex::new(0.0));
        let clock_now_s = Arc::clone(&now_s);
        let mut parser =
            CodexOutputParser::with_clock_and_max_busy(move || *clock_now_s.lock().unwrap(), 5.0);

        assert!(!parser.capabilities().can_stream_semantic_events);
        assert!(!parser.capabilities().can_detect_turn_complete);
        assert!(!parser.capabilities().can_resume_by_id);

        let out = parser.consume_output("{\"type\":\"thread.started\",\"thread_id\":\"th_123\"}\n");
        assert!(out.events.is_empty());
        assert!(out.capabilities.can_stream_semantic_events);
        assert!(!out.capabilities.can_detect_turn_complete);
        assert!(out.capabilities.can_resume_by_id);
        assert_eq!(
            out.external_session_ref,
            &ExternalSessionRef::CodexThread {
                thread_id: "th_123".to_owned(),
                turn_id: None
            }
        );

        let out = parser.consume_output("{\"type\":\"turn.started\",\"turn_id\":\"tu_1\"}\n");
        assert!(
            out.events
                .iter()
                .any(|e| matches!(e, CodexParserEvent::TurnStarted))
        );
        assert!(out.capabilities.can_detect_turn_complete);
        assert_eq!(out.status.turn_state, CodexTurnState::Busy);
        assert_eq!(
            out.external_session_ref,
            &ExternalSessionRef::CodexThread {
                thread_id: "th_123".to_owned(),
                turn_id: Some("tu_1".to_owned())
            }
        );

        let out = parser.consume_output("{\"type\":\"turn.completed\",\"turn_id\":\"tu_1\"}\n");
        assert!(
            out.events
                .iter()
                .any(|e| matches!(e, CodexParserEvent::TurnCompleted))
        );
        assert_eq!(out.status.turn_state, CodexTurnState::Completed);
        assert_eq!(
            out.external_session_ref,
            &ExternalSessionRef::CodexThread {
                thread_id: "th_123".to_owned(),
                turn_id: Some("tu_1".to_owned())
            }
        );
    }

    #[test]
    fn codex_parser_uses_prompt_heuristics_for_turn_completion() {
        let now_s = Arc::new(Mutex::new(0.0));
        let clock_now_s = Arc::clone(&now_s);
        let mut parser =
            CodexOutputParser::with_clock_and_max_busy(move || *clock_now_s.lock().unwrap(), 5.0);

        let _ = parser.consume_output("OpenAI Codex\n> ");
        assert_eq!(parser.status().turn_state, CodexTurnState::Ready);

        let _ = parser.consume_output("> hello\n");
        assert_eq!(parser.status().turn_state, CodexTurnState::Busy);

        let _ = parser.consume_output("response line 1\nresponse line 2\n> ");
        assert_eq!(parser.status().turn_state, CodexTurnState::Completed);
    }

    #[test]
    fn codex_parser_degrades_to_unknown_on_timeout() {
        let now_s = Arc::new(Mutex::new(0.0));
        let clock_now_s = Arc::clone(&now_s);
        let mut parser =
            CodexOutputParser::with_clock_and_max_busy(move || *clock_now_s.lock().unwrap(), 1.0);

        let _ = parser.consume_output("OpenAI Codex\n> ");
        let _ = parser.consume_output("> do something\n");
        assert_eq!(parser.status().turn_state, CodexTurnState::Busy);

        *now_s.lock().unwrap() = 2.0;
        let _ = parser.consume_output("");
        assert_eq!(parser.status().turn_state, CodexTurnState::Unknown);
        assert!(parser.status().detail.is_some());
    }

    #[test]
    fn codex_parser_emits_assistant_message_events_from_structured_stream() {
        let mut parser = CodexOutputParser::new();
        let _ = parser.consume_output("{\"type\":\"thread.started\",\"thread_id\":\"th_123\"}\n");
        let out = parser.consume_output(
            "{\"type\":\"assistant.message\",\"role\":\"assistant\",\"text\":\"Hello there\"}\n",
        );
        assert!(
            out.events
                .iter()
                .any(|e| matches!(e, CodexParserEvent::AssistantMessage { .. }))
        );
    }

    #[test]
    fn codex_parser_emits_assistant_message_events_from_item_completed_reasoning() {
        let mut parser = CodexOutputParser::new();
        let out = parser.consume_output(
            "{\"type\":\"item.completed\",\"item\":{\"id\":\"item_88\",\"type\":\"reasoning\",\"text\":\"Hello world\"}}\n",
        );
        assert!(
            out.events
                .iter()
                .any(|e| matches!(e, CodexParserEvent::AssistantMessage { .. }))
        );
        assert!(out.capabilities.can_stream_semantic_events);
        assert!(!out.capabilities.can_detect_turn_complete);
    }

    #[test]
    fn codex_parser_treats_item_started_as_activity_signal() {
        let mut parser = CodexOutputParser::new();
        let out = parser.consume_output(
            "{\"type\":\"item.started\",\"item\":{\"id\":\"item_90\",\"type\":\"command_execution\",\"command\":\"/bin/zsh -lc \\\"echo hi\\\"\"}}\n",
        );
        assert!(
            out.events
                .iter()
                .any(|e| matches!(e, CodexParserEvent::TurnStarted))
        );
        assert_eq!(out.status.turn_state, CodexTurnState::Busy);
        assert!(out.status.detail.is_some());
        assert!(!out.capabilities.can_detect_turn_complete);
    }
}
