use std::time::Duration;

use redesmyn_exec::artifact_store::LocalArtifactStore;
use redesmyn_exec::parser::{OutputStream, SessionOutputParser};
use redesmyn_exec::supervisor::{
    ExecSessionSpec, ExecSessionSupervisor, ExecSessionSupervisorConfig, SessionControlError,
    StartSessionError,
};
use redesmyn_ids::TaskId;
use redesmyn_protocol::artifacts::StorageHint;
use redesmyn_protocol::daemon::DaemonMessage;
use redesmyn_protocol::session::{
    AssistantMessage, InterfaceMode, SessionEvent, SessionEventKind, SessionScope,
};
use tokio::sync::mpsc;

#[derive(Default)]
struct AssistantLineParser {
    buf: String,
}

impl SessionOutputParser for AssistantLineParser {
    fn push_chunk(&mut self, _stream: OutputStream, chunk: &str) -> Vec<SessionEventKind> {
        self.buf.push_str(chunk);
        self.drain_lines()
    }

    fn flush(&mut self) -> Vec<SessionEventKind> {
        let mut out = self.drain_lines();
        let tail = self.buf.trim();
        if !tail.is_empty() {
            out.extend(parse_line(tail));
        }
        self.buf.clear();
        out
    }
}

impl AssistantLineParser {
    fn drain_lines(&mut self) -> Vec<SessionEventKind> {
        let mut out = Vec::new();
        while let Some(pos) = self.buf.find('\n') {
            let line = self.buf[..pos].trim_end_matches('\r');
            out.extend(parse_line(line));
            self.buf.drain(..pos + 1);
        }
        out
    }
}

fn parse_line(line: &str) -> Vec<SessionEventKind> {
    let trimmed = line.trim();
    let Some(rest) = trimmed.strip_prefix("assistant:") else {
        return Vec::new();
    };

    let text = rest.trim_start().to_owned();
    vec![SessionEventKind::AssistantMessage(AssistantMessage {
        text,
        preview: String::new(),
        full_text_artifact: None,
    })]
}

async fn collect_session_records(
    rx: &mut mpsc::Receiver<redesmyn_protocol::daemon::DaemonFrame>,
    session_id: redesmyn_ids::SessionId,
    timeout: Duration,
) -> Vec<SessionEvent> {
    let frame = tokio::time::timeout(timeout, rx.recv())
        .await
        .expect("timeout waiting for daemon frame")
        .expect("frame channel closed");

    match frame.message {
        DaemonMessage::SessionEventBatch(batch) => batch
            .events
            .into_iter()
            .filter(|event| event.session_id == session_id)
            .collect(),
        _ => Vec::new(),
    }
}

async fn collect_until_session_ended(
    rx: &mut mpsc::Receiver<redesmyn_protocol::daemon::DaemonFrame>,
    session_id: redesmyn_ids::SessionId,
    timeout: Duration,
) -> Vec<SessionEvent> {
    let mut out = Vec::new();
    loop {
        let records = collect_session_records(rx, session_id, timeout).await;
        let ended = records
            .iter()
            .any(|event| matches!(event.kind, SessionEventKind::SessionEnded(_)));
        out.extend(records);
        if ended {
            return out;
        }
    }
}

#[tokio::test]
async fn interrupt_session_emits_parsed_message_events() {
    let (frames_tx, mut frames_rx) = mpsc::channel(256);
    let tmp = tempfile::tempdir().expect("tempdir");
    let artifact_store = LocalArtifactStore::new(tmp.path().to_path_buf());

    let supervisor = ExecSessionSupervisor::new(
        ExecSessionSupervisorConfig::default(),
        artifact_store,
        frames_tx,
    )
    .await
    .expect("supervisor");

    let task_id = TaskId::new();

    let scope = SessionScope::Task { task_id };
    let script = r#"
	set -euo pipefail
	trap 'exit 0' INT TERM
	echo "assistant: hello"
echo "assistant: world" >&2
while read -r _line; do :; done
"#;

    let spec = ExecSessionSpec {
        scope,
        interface_mode: InterfaceMode::Structured,
        argv: vec!["bash".to_owned(), "-c".to_owned(), script.to_owned()],
        env: Vec::new(),
        cwd: tmp.path().to_path_buf(),
        parser: Some(Box::new(AssistantLineParser::default())),
        allow_concurrent_for_task: false,
    };

    let session_id = supervisor
        .start_session(task_id, spec)
        .await
        .expect("start_session");

    // Wait for at least one assistant message before interrupting so we know output ingestion works.
    let mut collected = Vec::new();
    let mut saw_message = false;
    while !saw_message {
        let records =
            collect_session_records(&mut frames_rx, session_id, Duration::from_secs(5)).await;
        for event in &records {
            if matches!(event.kind, SessionEventKind::AssistantMessage(_)) {
                saw_message = true;
                break;
            }
        }
        collected.extend(records);
    }
    assert!(saw_message, "expected assistant message event");

    supervisor
        .interrupt_session(session_id)
        .await
        .expect("interrupt_session");

    collected.extend(
        collect_until_session_ended(&mut frames_rx, session_id, Duration::from_secs(5)).await,
    );
    assert!(
        collected
            .iter()
            .any(|r| matches!(r.kind, SessionEventKind::SessionStarted(_)))
    );
    assert!(collected.iter().any(|r| matches!(
        r.kind,
        SessionEventKind::StatusUpdate(ref u)
            if u.message.as_deref() == Some("interrupt_requested")
    )));
    assert!(
        collected
            .iter()
            .any(|r| matches!(r.kind, SessionEventKind::AssistantMessage(_)))
    );
    assert!(
        collected
            .iter()
            .any(|r| matches!(r.kind, SessionEventKind::SessionEnded(_)))
    );

    supervisor.shutdown().await;
}

#[tokio::test]
async fn enforces_one_active_session_per_task() {
    let (frames_tx, mut frames_rx) = mpsc::channel(256);
    let tmp = tempfile::tempdir().expect("tempdir");
    let artifact_store = LocalArtifactStore::new(tmp.path().to_path_buf());

    let config = ExecSessionSupervisorConfig {
        emit_log_artifact_threshold_bytes: 1024 * 1024,
        ..ExecSessionSupervisorConfig::default()
    };

    let supervisor = ExecSessionSupervisor::new(config, artifact_store, frames_tx)
        .await
        .expect("supervisor");

    let task_id = TaskId::new();
    let scope = SessionScope::Task { task_id };

    let script = r#"
set -euo pipefail
trap 'exit 0' INT TERM
echo "assistant: started"
while read -r _line; do :; done
"#;

    let spec = ExecSessionSpec {
        scope,
        interface_mode: InterfaceMode::Structured,
        argv: vec!["bash".to_owned(), "-c".to_owned(), script.to_owned()],
        env: Vec::new(),
        cwd: tmp.path().to_path_buf(),
        parser: Some(Box::new(AssistantLineParser::default())),
        allow_concurrent_for_task: false,
    };

    let session_id = supervisor
        .start_session(task_id, spec)
        .await
        .expect("start_session");

    let err = supervisor
        .start_session(
            task_id,
            ExecSessionSpec {
                scope,
                interface_mode: InterfaceMode::Structured,
                argv: vec!["bash".to_owned(), "-c".to_owned(), "exit 0".to_owned()],
                env: Vec::new(),
                cwd: tmp.path().to_path_buf(),
                parser: None,
                allow_concurrent_for_task: false,
            },
        )
        .await
        .unwrap_err();

    assert!(
        matches!(err, StartSessionError::TaskHasActiveSession { .. }),
        "unexpected error: {err:?}"
    );

    supervisor
        .stop_session(session_id)
        .await
        .expect("stop_session");

    let _records =
        collect_until_session_ended(&mut frames_rx, session_id, Duration::from_secs(5)).await;
    supervisor.shutdown().await;
}

#[tokio::test]
async fn emits_log_artifacts_for_large_output() {
    let (frames_tx, mut frames_rx) = mpsc::channel(256);
    let tmp = tempfile::tempdir().expect("tempdir");
    let artifact_store = LocalArtifactStore::new(tmp.path().to_path_buf());

    let config = ExecSessionSupervisorConfig {
        emit_log_artifact_threshold_bytes: 1024,
        ..ExecSessionSupervisorConfig::default()
    };

    let supervisor = ExecSessionSupervisor::new(config, artifact_store, frames_tx)
        .await
        .expect("supervisor");

    let task_id = TaskId::new();
    let scope = SessionScope::Task { task_id };

    let spec = ExecSessionSpec {
        scope,
        interface_mode: InterfaceMode::Structured,
        argv: vec![
            "bash".to_owned(),
            "-c".to_owned(),
            "head -c 4096 /dev/zero".to_owned(),
        ],
        env: Vec::new(),
        cwd: tmp.path().to_path_buf(),
        parser: None,
        allow_concurrent_for_task: false,
    };

    let session_id = supervisor
        .start_session(task_id, spec)
        .await
        .expect("start_session");

    let records =
        collect_until_session_ended(&mut frames_rx, session_id, Duration::from_secs(5)).await;
    let artifact_refs: Vec<_> = records
        .iter()
        .filter_map(|event| match &event.kind {
            SessionEventKind::ArtifactEmitted(ev) => Some(ev.artifact.clone()),
            _ => None,
        })
        .collect();

    assert!(
        !artifact_refs.is_empty(),
        "expected at least one ArtifactEmitted"
    );
    assert!(
        artifact_refs
            .iter()
            .all(|artifact| matches!(&artifact.storage_hint, Some(StorageHint::BlobKey { .. })))
    );

    supervisor.shutdown().await;
}

#[tokio::test]
async fn concurrent_sessions_do_not_clear_active_task_marker() {
    let (frames_tx, mut frames_rx) = mpsc::channel(256);
    let tmp = tempfile::tempdir().expect("tempdir");
    let artifact_store = LocalArtifactStore::new(tmp.path().to_path_buf());

    let config = ExecSessionSupervisorConfig {
        emit_log_artifact_threshold_bytes: 1024 * 1024,
        ..ExecSessionSupervisorConfig::default()
    };

    let supervisor = ExecSessionSupervisor::new(config, artifact_store, frames_tx)
        .await
        .expect("supervisor");

    let task_id = TaskId::new();
    let scope = SessionScope::Task { task_id };

    let script = r#"
set -euo pipefail
trap 'exit 0' INT TERM
while true; do sleep 0.1; done
"#;

    let session_1 = supervisor
        .start_session(
            task_id,
            ExecSessionSpec {
                scope,
                interface_mode: InterfaceMode::Structured,
                argv: vec!["bash".to_owned(), "-c".to_owned(), script.to_owned()],
                env: Vec::new(),
                cwd: tmp.path().to_path_buf(),
                parser: None,
                allow_concurrent_for_task: true,
            },
        )
        .await
        .expect("start_session 1");

    let session_2 = supervisor
        .start_session(
            task_id,
            ExecSessionSpec {
                scope,
                interface_mode: InterfaceMode::Structured,
                argv: vec!["bash".to_owned(), "-c".to_owned(), script.to_owned()],
                env: Vec::new(),
                cwd: tmp.path().to_path_buf(),
                parser: None,
                allow_concurrent_for_task: true,
            },
        )
        .await
        .expect("start_session 2");

    supervisor
        .stop_session(session_1)
        .await
        .expect("stop_session 1");

    let _records =
        collect_until_session_ended(&mut frames_rx, session_1, Duration::from_secs(5)).await;

    // Wait for the supervisor cleanup loop to drop session_1 so we cover the bookkeeping path.
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match supervisor.stop_session(session_1).await {
                Err(SessionControlError::UnknownSession { .. }) => break,
                Err(SessionControlError::SessionClosed { .. }) | Ok(()) => {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            }
        }
    })
    .await
    .expect("timeout waiting for supervisor cleanup");

    let err = supervisor
        .start_session(
            task_id,
            ExecSessionSpec {
                scope,
                interface_mode: InterfaceMode::Structured,
                argv: vec!["bash".to_owned(), "-c".to_owned(), "exit 0".to_owned()],
                env: Vec::new(),
                cwd: tmp.path().to_path_buf(),
                parser: None,
                allow_concurrent_for_task: false,
            },
        )
        .await
        .unwrap_err();

    assert!(
        matches!(err, StartSessionError::TaskHasActiveSession { .. }),
        "unexpected error: {err:?}"
    );

    supervisor
        .stop_session(session_2)
        .await
        .expect("stop_session 2");

    let _records =
        collect_until_session_ended(&mut frames_rx, session_2, Duration::from_secs(5)).await;

    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match supervisor.stop_session(session_2).await {
                Err(SessionControlError::UnknownSession { .. }) => break,
                Err(SessionControlError::SessionClosed { .. }) | Ok(()) => {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            }
        }
    })
    .await
    .expect("timeout waiting for supervisor cleanup");

    let session_3 = supervisor
        .start_session(
            task_id,
            ExecSessionSpec {
                scope,
                interface_mode: InterfaceMode::Structured,
                argv: vec!["bash".to_owned(), "-c".to_owned(), "exit 0".to_owned()],
                env: Vec::new(),
                cwd: tmp.path().to_path_buf(),
                parser: None,
                allow_concurrent_for_task: false,
            },
        )
        .await
        .expect("start_session 3");

    let _records =
        collect_until_session_ended(&mut frames_rx, session_3, Duration::from_secs(5)).await;

    supervisor.shutdown().await;
}
