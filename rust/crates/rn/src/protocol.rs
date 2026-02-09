use std::{
    fs,
    io::{self, Read, Write},
    path::PathBuf,
};

use clap::{ArgAction, Args, Subcommand, ValueEnum};
use redesmyn_logging::tracing;
use redesmyn_protocol::client::{ClientFrame, ClientMessage};
use redesmyn_protocol::{ErrorCategory, ErrorEnvelope};
use redesmyn_transport::client::codec as client_codec;
use redesmyn_transport::client::codec::Codec as _;
use serde::Serialize;

use crate::{CommandOutcome, Output, OutputFormat};

#[cfg(unix)]
use std::os::unix::net::UnixStream;

#[derive(Debug, Subcommand)]
pub(crate) enum ProtocolCommands {
    /// Decodes protocol frames for debugging.
    Decode(ProtocolDecodeArgs),

    /// Encodes protocol frames from JSON for testing.
    Encode(ProtocolEncodeArgs),

    /// Connects to the client API UDS and logs frames.
    Tap(ProtocolTapArgs),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "lower")]
enum ProtocolCodec {
    Json,
    Protobuf,
}

#[derive(Debug, Args)]
pub(crate) struct ProtocolDecodeArgs {
    /// Codec to use when decoding input.
    #[arg(long, value_enum, default_value_t = ProtocolCodec::Protobuf)]
    codec: ProtocolCodec,

    /// Path to read from (defaults to stdin).
    #[arg(long)]
    input: Option<PathBuf>,

    /// Input is a framed stream (u32 length prefix + payload bytes).
    #[arg(
        long,
        action = ArgAction::Set,
        num_args = 0..=1,
        default_value_t = true,
        default_missing_value = "true"
    )]
    framed: bool,

    /// Include full decoded payloads (otherwise prints concise summaries).
    #[arg(long, default_value_t = false)]
    verbose: bool,

    /// Pretty-print JSON output (only applies to `--verbose` + `--output human`).
    #[arg(
        long,
        action = ArgAction::Set,
        num_args = 0..=1,
        default_value_t = true,
        default_missing_value = "true"
    )]
    pretty: bool,

    /// Max number of bytes to print for `json_payload` fields (truncates and annotates).
    #[arg(long, default_value_t = 1024)]
    max_json_payload_bytes: usize,
}

#[derive(Debug, Args)]
pub(crate) struct ProtocolEncodeArgs {
    /// Codec to use when encoding output.
    #[arg(long, value_enum, default_value_t = ProtocolCodec::Protobuf)]
    codec: ProtocolCodec,

    /// Path to read JSON from (defaults to stdin).
    #[arg(long)]
    input: Option<PathBuf>,

    /// Write output to a file (defaults to stdout).
    #[arg(long)]
    out: Option<PathBuf>,

    /// Write a framed stream (u32 length prefix + payload bytes).
    #[arg(
        long,
        action = ArgAction::Set,
        num_args = 0..=1,
        default_value_t = true,
        default_missing_value = "true"
    )]
    framed: bool,
}

#[derive(Debug, Args)]
pub(crate) struct ProtocolTapArgs {
    /// Unix domain socket path for the control-plane client API.
    ///
    /// If omitted, uses the configured path (via `redesmyn_config`).
    #[arg(long)]
    uds: Option<PathBuf>,

    /// Codec to use for the connection (Protobuf by default; JSON for debug/diagnostics).
    #[arg(long, value_enum, default_value_t = ProtocolCodec::Protobuf)]
    codec: ProtocolCodec,

    /// Include full decoded payloads (otherwise prints concise summaries).
    #[arg(long, default_value_t = false)]
    verbose: bool,

    /// Pretty-print JSON output (only applies to `--verbose` + `--output human`).
    #[arg(
        long,
        action = ArgAction::Set,
        num_args = 0..=1,
        default_value_t = true,
        default_missing_value = "true"
    )]
    pretty: bool,

    /// Max number of bytes to print for `json_payload` fields (truncates and annotates).
    #[arg(long, default_value_t = 1024)]
    max_json_payload_bytes: usize,

    /// Send a `status` request on connect (recommended; exercises request/response).
    #[arg(
        long,
        action = ArgAction::Set,
        num_args = 0..=1,
        default_value_t = true,
        default_missing_value = "true"
    )]
    status: bool,

    /// Subscribe to the `event_log` stream on connect (recommended; exercises streaming).
    #[arg(
        long,
        action = ArgAction::Set,
        num_args = 0..=1,
        default_value_t = true,
        default_missing_value = "true"
    )]
    subscribe_event_log: bool,
}

pub(crate) fn protocol(cmd: ProtocolCommands, output: &Output) -> CommandOutcome {
    match cmd {
        ProtocolCommands::Decode(args) => protocol_decode(args, output),
        ProtocolCommands::Encode(args) => protocol_encode(args),
        ProtocolCommands::Tap(args) => protocol_tap(args, output),
    }
}

fn protocol_decode(args: ProtocolDecodeArgs, output: &Output) -> CommandOutcome {
    let mut reader = match open_input_reader(args.input) {
        Ok(reader) => reader,
        Err(err) => return CommandOutcome::Failure(err),
    };

    let mut frame_index: u64 = 0;
    let mut saw_any = false;

    if args.framed {
        loop {
            let payload = match read_next_framed_payload(&mut reader) {
                Ok(Some(bytes)) => bytes,
                Ok(None) => break,
                Err(err) => return CommandOutcome::Failure(err),
            };
            saw_any = true;

            let frame = match decode_client_frame(&args.codec, &payload) {
                Ok(frame) => frame,
                Err(err) => return CommandOutcome::Failure(err),
            };

            if let Err(err) = print_client_frame(
                output,
                &frame,
                payload.len(),
                frame_index,
                args.verbose,
                args.pretty,
                args.max_json_payload_bytes,
            ) {
                return CommandOutcome::Failure(err);
            }

            frame_index += 1;
        }
    } else {
        let payload = match read_all_reader(&mut reader) {
            Ok(bytes) => bytes,
            Err(err) => {
                return CommandOutcome::Failure(ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    format!("failed to read input: {err}"),
                ));
            }
        };
        if !payload.is_empty() {
            saw_any = true;
            let frame = match decode_client_frame(&args.codec, &payload) {
                Ok(frame) => frame,
                Err(err) => return CommandOutcome::Failure(err),
            };

            if let Err(err) = print_client_frame(
                output,
                &frame,
                payload.len(),
                frame_index,
                args.verbose,
                args.pretty,
                args.max_json_payload_bytes,
            ) {
                return CommandOutcome::Failure(err);
            }
        }
    }

    if !saw_any {
        return CommandOutcome::Failure(ErrorEnvelope::new(
            ErrorCategory::InvalidRequest,
            "no frames found in input",
        ));
    }

    CommandOutcome::Success
}

fn protocol_encode(args: ProtocolEncodeArgs) -> CommandOutcome {
    let input_bytes = match read_all_input(args.input) {
        Ok(bytes) => bytes,
        Err(err) => return CommandOutcome::Failure(err),
    };

    let json_value: serde_json::Value = match serde_json::from_slice(&input_bytes) {
        Ok(value) => value,
        Err(err) => {
            return CommandOutcome::Failure(ErrorEnvelope::new(
                ErrorCategory::InvalidRequest,
                format!("failed to parse JSON input: {err}"),
            ));
        }
    };

    let frames: Vec<ClientFrame> = match json_value {
        serde_json::Value::Array(items) => {
            let mut frames = Vec::with_capacity(items.len());
            for item in items {
                let frame: ClientFrame = match serde_json::from_value(item) {
                    Ok(frame) => frame,
                    Err(err) => {
                        return CommandOutcome::Failure(ErrorEnvelope::new(
                            ErrorCategory::InvalidRequest,
                            format!("invalid client frame JSON: {err}"),
                        ));
                    }
                };
                frames.push(frame);
            }
            frames
        }
        serde_json::Value::Object(_) => {
            let frame: ClientFrame = match serde_json::from_value(json_value) {
                Ok(frame) => frame,
                Err(err) => {
                    return CommandOutcome::Failure(ErrorEnvelope::new(
                        ErrorCategory::InvalidRequest,
                        format!("invalid client frame JSON: {err}"),
                    ));
                }
            };
            vec![frame]
        }
        _ => {
            return CommandOutcome::Failure(ErrorEnvelope::new(
                ErrorCategory::InvalidRequest,
                "expected a client frame object or an array of client frames",
            ));
        }
    };

    if frames.len() > 1 && !args.framed {
        return CommandOutcome::Failure(ErrorEnvelope::new(
            ErrorCategory::InvalidRequest,
            "encoding multiple frames requires --framed (otherwise output is ambiguous)",
        ));
    }

    let mut writer = match open_output_writer(args.out) {
        Ok(writer) => writer,
        Err(err) => return CommandOutcome::Failure(err),
    };

    for frame in &frames {
        let payload = match encode_client_frame(&args.codec, frame) {
            Ok(payload) => payload,
            Err(err) => return CommandOutcome::Failure(err),
        };

        if args.framed {
            if payload.len() > u32::MAX as usize {
                return CommandOutcome::Failure(ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    format!("frame too large: {} bytes", payload.len()),
                ));
            }
            if let Err(err) = writer.write_all(&(payload.len() as u32).to_be_bytes()) {
                return CommandOutcome::Failure(ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    format!("failed to write output: {err}"),
                ));
            }
        }

        if let Err(err) = writer.write_all(&payload) {
            return CommandOutcome::Failure(ErrorEnvelope::new(
                ErrorCategory::Internal,
                format!("failed to write output: {err}"),
            ));
        }
    }

    if let Err(err) = writer.flush() {
        return CommandOutcome::Failure(ErrorEnvelope::new(
            ErrorCategory::Internal,
            format!("failed to flush output: {err}"),
        ));
    }

    CommandOutcome::Success
}

fn protocol_tap(args: ProtocolTapArgs, output: &Output) -> CommandOutcome {
    #[cfg(not(unix))]
    {
        let _ = output;
        let _ = args;
        return CommandOutcome::Failure(ErrorEnvelope::new(
            ErrorCategory::Unavailable,
            "Unix domain sockets are not supported on this platform",
        ));
    }

    #[cfg(unix)]
    {
        let socket_path = match args.uds {
            Some(path) => path,
            None => match load_default_client_socket_path() {
                Ok(path) => path,
                Err(err) => return CommandOutcome::Failure(err),
            },
        };

        tracing::info!(
            socket_path = %socket_path.display(),
            codec = ?args.codec,
            "connecting protocol tap"
        );

        let mut stream = match UnixStream::connect(&socket_path) {
            Ok(stream) => stream,
            Err(err) => {
                return CommandOutcome::Failure(ErrorEnvelope::new(
                    ErrorCategory::Unavailable,
                    format!("failed to connect to {}: {err}", socket_path.display()),
                ));
            }
        };

        let mut frame_index: u64 = 0;

        if args.status {
            let frame = build_status_request_frame();
            let payload = match encode_client_frame(&args.codec, &frame) {
                Ok(payload) => payload,
                Err(err) => return CommandOutcome::Failure(err),
            };

            if let Err(err) = write_framed_payload(&mut stream, &payload) {
                return CommandOutcome::Failure(err);
            }

            if let Err(err) = print_tapped_client_frame(
                output,
                TapDirection::Send,
                &frame,
                payload.len(),
                frame_index,
                args.verbose,
                args.pretty,
                args.max_json_payload_bytes,
            ) {
                return CommandOutcome::Failure(err);
            }

            frame_index += 1;
        }

        if args.subscribe_event_log {
            let frame = build_event_log_subscribe_frame();
            let payload = match encode_client_frame(&args.codec, &frame) {
                Ok(payload) => payload,
                Err(err) => return CommandOutcome::Failure(err),
            };

            if let Err(err) = write_framed_payload(&mut stream, &payload) {
                return CommandOutcome::Failure(err);
            }

            if let Err(err) = print_tapped_client_frame(
                output,
                TapDirection::Send,
                &frame,
                payload.len(),
                frame_index,
                args.verbose,
                args.pretty,
                args.max_json_payload_bytes,
            ) {
                return CommandOutcome::Failure(err);
            }

            frame_index += 1;
        }

        loop {
            let payload = match read_next_framed_payload(&mut stream) {
                Ok(Some(bytes)) => bytes,
                Ok(None) => return CommandOutcome::Success,
                Err(err) => return CommandOutcome::Failure(err),
            };

            let frame = match decode_client_frame(&args.codec, &payload) {
                Ok(frame) => frame,
                Err(err) => return CommandOutcome::Failure(err),
            };

            if let Err(err) = print_tapped_client_frame(
                output,
                TapDirection::Recv,
                &frame,
                payload.len(),
                frame_index,
                args.verbose,
                args.pretty,
                args.max_json_payload_bytes,
            ) {
                return CommandOutcome::Failure(err);
            }

            frame_index += 1;
        }
    }
}

fn read_all_input(input: Option<PathBuf>) -> Result<Vec<u8>, ErrorEnvelope> {
    match input {
        Some(path) => fs::read(&path).map_err(|err| {
            ErrorEnvelope::new(
                ErrorCategory::Internal,
                format!("failed to read input file {}: {err}", path.display()),
            )
        }),
        None => {
            let mut bytes = Vec::new();
            io::stdin().read_to_end(&mut bytes).map_err(|err| {
                ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    format!("failed to read stdin: {err}"),
                )
            })?;
            Ok(bytes)
        }
    }
}

fn open_input_reader(input: Option<PathBuf>) -> Result<Box<dyn Read>, ErrorEnvelope> {
    match input {
        Some(path) => {
            let file = fs::File::open(&path).map_err(|err| {
                ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    format!("failed to open input file {}: {err}", path.display()),
                )
            })?;
            Ok(Box::new(io::BufReader::new(file)))
        }
        None => Ok(Box::new(io::BufReader::new(io::stdin()))),
    }
}

fn open_output_writer(out: Option<PathBuf>) -> Result<Box<dyn Write>, ErrorEnvelope> {
    match out {
        Some(path) => {
            let file = fs::File::create(&path).map_err(|err| {
                ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    format!("failed to create output file {}: {err}", path.display()),
                )
            })?;
            Ok(Box::new(io::BufWriter::new(file)))
        }
        None => Ok(Box::new(io::BufWriter::new(io::stdout()))),
    }
}

fn read_all_reader<R: Read>(reader: &mut R) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn read_exact_or_eof<R: Read>(reader: &mut R, buf: &mut [u8]) -> io::Result<bool> {
    let mut read_total = 0;
    while read_total < buf.len() {
        match reader.read(&mut buf[read_total..]) {
            Ok(0) => {
                if read_total == 0 {
                    return Ok(false);
                }
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "unexpected end of file",
                ));
            }
            Ok(n) => read_total += n,
            Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
            Err(err) => return Err(err),
        }
    }
    Ok(true)
}

fn read_next_framed_payload<R: Read>(reader: &mut R) -> Result<Option<Vec<u8>>, ErrorEnvelope> {
    let mut len_buf = [0_u8; 4];
    let has_more = read_exact_or_eof(reader, &mut len_buf).map_err(|err| {
        ErrorEnvelope::new(
            ErrorCategory::Internal,
            format!("failed to read frame length: {err}"),
        )
    })?;
    if !has_more {
        return Ok(None);
    }

    let len = u32::from_be_bytes(len_buf) as usize;
    if len > redesmyn_transport::client::framed::DEFAULT_MAX_FRAME_LEN {
        return Err(ErrorEnvelope::new(
            ErrorCategory::InvalidRequest,
            format!(
                "frame too large: {len} > {} bytes",
                redesmyn_transport::client::framed::DEFAULT_MAX_FRAME_LEN
            ),
        ));
    }

    let mut payload = vec![0_u8; len];
    read_exact_or_eof(reader, &mut payload).map_err(|err| {
        ErrorEnvelope::new(
            ErrorCategory::Internal,
            format!("failed to read frame payload: {err}"),
        )
    })?;
    Ok(Some(payload))
}

fn decode_client_frame(codec: &ProtocolCodec, bytes: &[u8]) -> Result<ClientFrame, ErrorEnvelope> {
    match codec {
        ProtocolCodec::Json => client_codec::JsonCodec::new()
            .decode_frame(bytes)
            .map_err(client_codec_error_to_envelope),
        ProtocolCodec::Protobuf => client_codec::ProtobufCodec::new()
            .decode_frame(bytes)
            .map_err(client_codec_error_to_envelope),
    }
}

fn encode_client_frame(
    codec: &ProtocolCodec,
    frame: &ClientFrame,
) -> Result<Vec<u8>, ErrorEnvelope> {
    match codec {
        ProtocolCodec::Json => client_codec::JsonCodec::new()
            .encode_frame(frame)
            .map_err(client_codec_error_to_envelope),
        ProtocolCodec::Protobuf => client_codec::ProtobufCodec::new()
            .encode_frame(frame)
            .map_err(client_codec_error_to_envelope),
    }
}

fn client_codec_error_to_envelope(err: client_codec::CodecError) -> ErrorEnvelope {
    match err {
        client_codec::CodecError::Frame(frame_err) => frame_err.error,
        other => ErrorEnvelope::new(ErrorCategory::InvalidRequest, other.to_string()),
    }
}

fn print_client_frame(
    output: &Output,
    frame: &ClientFrame,
    payload_len: usize,
    frame_index: u64,
    verbose: bool,
    pretty: bool,
    max_json_payload_bytes: usize,
) -> Result<(), ErrorEnvelope> {
    if !verbose {
        match output.format {
            OutputFormat::Human => {
                println!(
                    "{}",
                    client_frame_summary_line(frame, payload_len, frame_index)
                );
            }
            OutputFormat::Json => {
                let summary = client_frame_summary(frame, payload_len, frame_index);
                output
                    .print_json_stdout(&summary)
                    .map_err(|err| ErrorEnvelope::new(ErrorCategory::Internal, err.to_string()))?;
            }
        }
        return Ok(());
    }

    let mut value = serde_json::to_value(frame)
        .map_err(|err| ErrorEnvelope::new(ErrorCategory::Internal, err.to_string()))?;
    truncate_json_payload_fields(&mut value, max_json_payload_bytes);

    match output.format {
        OutputFormat::Human => {
            if pretty {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&value)
                        .unwrap_or_else(|_| "<failed to render JSON>".to_owned())
                );
            } else {
                println!(
                    "{}",
                    serde_json::to_string(&value)
                        .unwrap_or_else(|_| "<failed to render JSON>".to_owned())
                );
            }
        }
        OutputFormat::Json => {
            output
                .print_json_stdout(&value)
                .map_err(|err| ErrorEnvelope::new(ErrorCategory::Internal, err.to_string()))?;
        }
    }

    Ok(())
}

#[derive(Debug, Serialize)]
struct ClientFrameSummary {
    frame_index: u64,
    payload_len: usize,
    protocol_version: redesmyn_protocol::ProtocolVersion,
    msg_id: redesmyn_ids::MsgId,
    #[serde(skip_serializing_if = "Option::is_none")]
    correlation_id: Option<redesmyn_ids::MsgId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    trace_id: Option<redesmyn_protocol::TraceId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    scope: Option<redesmyn_protocol::Scope>,
    message: ClientMessageSummary,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ClientMessageSummary {
    Request {
        request_id: redesmyn_ids::RequestId,
        method: redesmyn_protocol::client::ClientMethod,
    },
    Response {
        request_id: redesmyn_ids::RequestId,
        status: redesmyn_protocol::client::ResponseStatus,
    },
    Subscribe {
        subscription_id: redesmyn_ids::SubscriptionId,
        topic: redesmyn_protocol::client::SubscriptionTopic,
    },
    Event {
        subscription_id: redesmyn_ids::SubscriptionId,
        event: ClientEventSummary,
    },
    Unsubscribe {
        subscription_id: redesmyn_ids::SubscriptionId,
    },
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ClientEventSummary {
    Subscribed {
        topic: redesmyn_protocol::client::SubscriptionTopic,
    },
    EventLog {
        event_id: redesmyn_ids::EventId,
        event_type: String,
        json_payload_len: usize,
    },
    SessionEvent {
        session_id: redesmyn_ids::SessionId,
        session_event_id: redesmyn_ids::SessionEventId,
        event_type: String,
    },
    SessionLiveEvent {
        session_id: redesmyn_ids::SessionId,
        event_type: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        turn_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        item_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_name: Option<String>,
        delta_len: usize,
    },
    Error {
        category: ErrorCategory,
        message: String,
    },
}

fn session_event_kind_label(kind: &redesmyn_protocol::session::SessionEventKind) -> String {
    match kind {
        redesmyn_protocol::session::SessionEventKind::SessionStarted(_) => "session_started",
        redesmyn_protocol::session::SessionEventKind::SessionEnded(_) => "session_ended",
        redesmyn_protocol::session::SessionEventKind::TurnStarted(_) => "turn_started",
        redesmyn_protocol::session::SessionEventKind::TurnCompleted(_) => "turn_completed",
        redesmyn_protocol::session::SessionEventKind::UserMessage(_) => "user_message",
        redesmyn_protocol::session::SessionEventKind::AssistantMessage(_) => "assistant_message",
        redesmyn_protocol::session::SessionEventKind::AssistantReasoning(_) => {
            "assistant_reasoning"
        }
        redesmyn_protocol::session::SessionEventKind::ToolInvocation(_) => "tool_invocation",
        redesmyn_protocol::session::SessionEventKind::ToolResult(_) => "tool_result",
        redesmyn_protocol::session::SessionEventKind::StatusUpdate(_) => "status_update",
        redesmyn_protocol::session::SessionEventKind::TaskAgentMessageSent(_) => {
            "task_agent_message_sent"
        }
        redesmyn_protocol::session::SessionEventKind::PermissionsModeChanged(_) => {
            "permissions_mode_changed"
        }
        redesmyn_protocol::session::SessionEventKind::CodexApprovalPolicyChanged(_) => {
            "codex_approval_policy_changed"
        }
        redesmyn_protocol::session::SessionEventKind::CodexSandboxPolicyChanged(_) => {
            "codex_sandbox_policy_changed"
        }
        redesmyn_protocol::session::SessionEventKind::PermissionRequested(_) => {
            "permission_requested"
        }
        redesmyn_protocol::session::SessionEventKind::PermissionDecided(_) => "permission_decided",
        redesmyn_protocol::session::SessionEventKind::ArtifactEmitted(_) => "artifact_emitted",
        redesmyn_protocol::session::SessionEventKind::SessionModelChanged(_) => {
            "session_model_changed"
        }
        redesmyn_protocol::session::SessionEventKind::Unknown(unknown) => &unknown.event_type,
    }
    .to_owned()
}

fn session_live_event_kind_summary(
    event: &redesmyn_protocol::session_live::SessionLiveEvent,
) -> (String, Option<String>, usize) {
    match &event.kind {
        redesmyn_protocol::session_live::SessionLiveEventKind::AssistantMessageDelta(delta) => (
            "assistant_message_delta".to_owned(),
            None,
            delta.delta.len(),
        ),
        redesmyn_protocol::session_live::SessionLiveEventKind::AssistantReasoningSummaryPartAdded(
            _,
        ) => ("assistant_reasoning_summary_part_added".to_owned(), None, 0),
        redesmyn_protocol::session_live::SessionLiveEventKind::AssistantReasoningSummaryDelta(
            delta,
        ) => (
            "assistant_reasoning_summary_delta".to_owned(),
            None,
            delta.delta.len(),
        ),
        redesmyn_protocol::session_live::SessionLiveEventKind::AssistantReasoningRawDelta(delta) => (
            "assistant_reasoning_raw_delta".to_owned(),
            None,
            delta.delta.len(),
        ),
        redesmyn_protocol::session_live::SessionLiveEventKind::ToolOutputDelta(delta) => (
            "tool_output_delta".to_owned(),
            Some(delta.tool_name.clone()),
            delta.delta.len(),
        ),
        redesmyn_protocol::session_live::SessionLiveEventKind::Unknown(unknown) => (
            unknown.event_type.clone(),
            None,
            unknown.json_payload.len(),
        ),
    }
}

fn client_frame_summary(
    frame: &ClientFrame,
    payload_len: usize,
    frame_index: u64,
) -> ClientFrameSummary {
    let envelope = &frame.envelope;
    let message = match &frame.message {
        ClientMessage::Request(req) => ClientMessageSummary::Request {
            request_id: req.request_id,
            method: req.method(),
        },
        ClientMessage::Response(resp) => ClientMessageSummary::Response {
            request_id: resp.request_id,
            status: resp.status(),
        },
        ClientMessage::Subscribe(sub) => ClientMessageSummary::Subscribe {
            subscription_id: sub.subscription_id,
            topic: sub.topic(),
        },
        ClientMessage::Event(ev) => {
            let event = match &ev.event {
                redesmyn_protocol::client::SubscriptionEvent::Subscribed(subscribed) => {
                    ClientEventSummary::Subscribed {
                        topic: subscribed.topic,
                    }
                }
                redesmyn_protocol::client::SubscriptionEvent::EventLog(event_log) => {
                    ClientEventSummary::EventLog {
                        event_id: event_log.event_id,
                        event_type: event_log.event_type.clone(),
                        json_payload_len: event_log.json_payload.len(),
                    }
                }
                redesmyn_protocol::client::SubscriptionEvent::SessionEvent(session_event) => {
                    ClientEventSummary::SessionEvent {
                        session_id: session_event.session_id,
                        session_event_id: session_event.session_event_id,
                        event_type: session_event_kind_label(&session_event.kind),
                    }
                }
                redesmyn_protocol::client::SubscriptionEvent::SessionLiveEvent(live_event) => {
                    let (event_type, tool_name, delta_len) =
                        session_live_event_kind_summary(live_event);
                    ClientEventSummary::SessionLiveEvent {
                        session_id: live_event.session_id,
                        event_type,
                        turn_id: live_event.turn_id.clone(),
                        item_id: live_event.item_id.clone(),
                        tool_name,
                        delta_len,
                    }
                }
                redesmyn_protocol::client::SubscriptionEvent::Error(err) => {
                    ClientEventSummary::Error {
                        category: err.category,
                        message: err.message.clone(),
                    }
                }
            };

            ClientMessageSummary::Event {
                subscription_id: ev.subscription_id,
                event,
            }
        }
        ClientMessage::Unsubscribe(unsub) => ClientMessageSummary::Unsubscribe {
            subscription_id: unsub.subscription_id,
        },
    };

    ClientFrameSummary {
        frame_index,
        payload_len,
        protocol_version: envelope.protocol_version(),
        msg_id: envelope.msg_id,
        correlation_id: envelope.correlation_id,
        trace_id: envelope.trace_id,
        scope: envelope.scope,
        message,
    }
}

fn client_frame_summary_line(frame: &ClientFrame, payload_len: usize, frame_index: u64) -> String {
    let summary = client_frame_summary(frame, payload_len, frame_index);
    let scope = match summary.scope {
        Some(scope) => match scope {
            redesmyn_protocol::Scope::Repo { repo } => {
                format!("repo({}/{})", repo.workspace_id, repo.repo_id)
            }
            redesmyn_protocol::Scope::Unknown => "unknown".to_string(),
            _ => "unknown".to_string(),
        },
        None => "-".to_string(),
    };

    let ids = match (summary.correlation_id, summary.trace_id) {
        (Some(correlation_id), Some(trace_id)) => {
            format!(
                "msg_id={} corr_id={} trace_id={}",
                summary.msg_id, correlation_id, trace_id
            )
        }
        (Some(correlation_id), None) => {
            format!("msg_id={} corr_id={}", summary.msg_id, correlation_id)
        }
        (None, Some(trace_id)) => format!("msg_id={} trace_id={}", summary.msg_id, trace_id),
        (None, None) => format!("msg_id={}", summary.msg_id),
    };

    let msg = match summary.message {
        ClientMessageSummary::Request { request_id, method } => {
            format!("request {method:?} request_id={request_id}")
        }
        ClientMessageSummary::Response { request_id, status } => {
            format!("response {status:?} request_id={request_id}")
        }
        ClientMessageSummary::Subscribe {
            subscription_id,
            topic,
        } => {
            format!("subscribe {topic:?} subscription_id={subscription_id}")
        }
        ClientMessageSummary::Event {
            subscription_id,
            event,
        } => {
            let event_desc = match event {
                ClientEventSummary::Subscribed { topic } => format!("subscribed {topic:?}"),
                ClientEventSummary::EventLog {
                    event_id,
                    event_type,
                    json_payload_len,
                } => format!(
                    "event_log event_id={event_id} event_type={event_type} json_payload_len={json_payload_len}"
                ),
                ClientEventSummary::SessionEvent {
                    session_id,
                    session_event_id,
                    event_type,
                } => format!(
                    "session_event session_id={session_id} session_event_id={session_event_id} event_type={event_type}"
                ),
                ClientEventSummary::SessionLiveEvent {
                    session_id,
                    event_type,
                    turn_id,
                    item_id,
                    tool_name,
                    delta_len,
                } => {
                    let tool = tool_name.map(|name| format!(" tool_name={name}"));
                    let turn = turn_id.map(|id| format!(" turn_id={id}"));
                    let item = item_id.map(|id| format!(" item_id={id}"));
                    format!(
                        "session_live_event session_id={session_id} event_type={event_type} delta_len={delta_len}{}{}{}",
                        tool.as_deref().unwrap_or(""),
                        turn.as_deref().unwrap_or(""),
                        item.as_deref().unwrap_or(""),
                    )
                }
                ClientEventSummary::Error { category, message } => {
                    format!("error {category}: {message}")
                }
            };

            format!("event subscription_id={subscription_id} {event_desc}")
        }
        ClientMessageSummary::Unsubscribe { subscription_id } => {
            format!("unsubscribe subscription_id={subscription_id}")
        }
    };

    format!(
        "frame={frame_index} v={} {msg} {ids} scope={scope} len={payload_len}",
        summary.protocol_version
    )
}

fn truncate_json_payload_fields(value: &mut serde_json::Value, max_bytes: usize) {
    match value {
        serde_json::Value::Object(map) => {
            if let Some(serde_json::Value::Array(bytes)) = map.get_mut("json_payload") {
                let byte_len = bytes.len();
                if byte_len > max_bytes {
                    bytes.truncate(max_bytes);
                    map.insert(
                        "json_payload_byte_len".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(byte_len as u64)),
                    );
                    map.insert(
                        "json_payload_truncated".to_string(),
                        serde_json::Value::Bool(true),
                    );
                }
            }

            for value in map.values_mut() {
                truncate_json_payload_fields(value, max_bytes);
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                truncate_json_payload_fields(value, max_bytes);
            }
        }
        _ => {}
    }
}

#[cfg(unix)]
fn load_default_client_socket_path() -> Result<PathBuf, ErrorEnvelope> {
    let config = redesmyn_config::load_rust_config(Default::default()).map_err(|err| {
        ErrorEnvelope::new(
            ErrorCategory::Unavailable,
            format!("failed to load config: {err}"),
        )
    })?;
    Ok(config.control_plane.api.client_socket_path)
}

#[cfg(not(unix))]
fn load_default_client_socket_path() -> Result<PathBuf, ErrorEnvelope> {
    Err(ErrorEnvelope::new(
        ErrorCategory::Unavailable,
        "Unix domain sockets are not supported on this platform",
    ))
}

fn build_status_request_frame() -> ClientFrame {
    let envelope = redesmyn_protocol::ProtocolEnvelope::new();
    let request = redesmyn_protocol::client::Request {
        request_id: redesmyn_ids::RequestId::new(),
        payload: redesmyn_protocol::client::RequestPayload::Status(
            redesmyn_protocol::client::StatusRequest {},
        ),
    };
    ClientFrame::new(envelope, ClientMessage::Request(request))
}

fn build_event_log_subscribe_frame() -> ClientFrame {
    let envelope = redesmyn_protocol::ProtocolEnvelope::new();
    let subscribe = redesmyn_protocol::client::Subscribe {
        subscription_id: redesmyn_ids::SubscriptionId::new(),
        filter: redesmyn_protocol::client::SubscriptionFilter::EventLog(
            redesmyn_protocol::client::EventLogFilter {
                after_event_id: None,
            },
        ),
    };
    ClientFrame::new(envelope, ClientMessage::Subscribe(subscribe))
}

fn write_framed_payload<W: Write>(writer: &mut W, payload: &[u8]) -> Result<(), ErrorEnvelope> {
    if payload.len() > redesmyn_transport::client::framed::DEFAULT_MAX_FRAME_LEN {
        return Err(ErrorEnvelope::new(
            ErrorCategory::InvalidRequest,
            format!(
                "frame too large: {} > {} bytes",
                payload.len(),
                redesmyn_transport::client::framed::DEFAULT_MAX_FRAME_LEN
            ),
        ));
    }
    if payload.len() > u32::MAX as usize {
        return Err(ErrorEnvelope::new(
            ErrorCategory::InvalidRequest,
            format!("frame too large: {} bytes", payload.len()),
        ));
    }

    writer
        .write_all(&(payload.len() as u32).to_be_bytes())
        .map_err(|err| {
            ErrorEnvelope::new(
                ErrorCategory::Unavailable,
                format!("failed to write frame: {err}"),
            )
        })?;
    writer.write_all(payload).map_err(|err| {
        ErrorEnvelope::new(
            ErrorCategory::Unavailable,
            format!("failed to write frame: {err}"),
        )
    })?;
    writer.flush().map_err(|err| {
        ErrorEnvelope::new(
            ErrorCategory::Unavailable,
            format!("failed to flush frame: {err}"),
        )
    })?;
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum TapDirection {
    Send,
    Recv,
}

#[derive(Debug, Serialize)]
struct TappedFrameSummary {
    direction: TapDirection,
    #[serde(flatten)]
    frame: ClientFrameSummary,
}

#[derive(Debug, Serialize)]
struct TappedFrameVerbose {
    direction: TapDirection,
    frame_index: u64,
    payload_len: usize,
    frame: serde_json::Value,
}

fn print_tapped_client_frame(
    output: &Output,
    direction: TapDirection,
    frame: &ClientFrame,
    payload_len: usize,
    frame_index: u64,
    verbose: bool,
    pretty: bool,
    max_json_payload_bytes: usize,
) -> Result<(), ErrorEnvelope> {
    if !verbose {
        match output.format {
            OutputFormat::Human => {
                println!(
                    "{direction:?} {}",
                    client_frame_summary_line(frame, payload_len, frame_index)
                );
            }
            OutputFormat::Json => {
                output
                    .print_json_stdout(&TappedFrameSummary {
                        direction,
                        frame: client_frame_summary(frame, payload_len, frame_index),
                    })
                    .map_err(|err| ErrorEnvelope::new(ErrorCategory::Internal, err.to_string()))?;
            }
        }
        return Ok(());
    }

    let mut value = serde_json::to_value(frame)
        .map_err(|err| ErrorEnvelope::new(ErrorCategory::Internal, err.to_string()))?;
    truncate_json_payload_fields(&mut value, max_json_payload_bytes);

    let wrapper = TappedFrameVerbose {
        direction,
        frame_index,
        payload_len,
        frame: value,
    };

    match output.format {
        OutputFormat::Human => {
            if pretty {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&wrapper)
                        .unwrap_or_else(|_| "<failed to render JSON>".to_owned())
                );
            } else {
                println!(
                    "{}",
                    serde_json::to_string(&wrapper)
                        .unwrap_or_else(|_| "<failed to render JSON>".to_owned())
                );
            }
        }
        OutputFormat::Json => {
            output
                .print_json_stdout(&wrapper)
                .map_err(|err| ErrorEnvelope::new(ErrorCategory::Internal, err.to_string()))?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use clap::Parser;
    use std::io::Cursor;

    #[test]
    fn protobuf_framed_roundtrip_decodes_single_frame() {
        let frame = build_status_request_frame();
        let payload = encode_client_frame(&ProtocolCodec::Protobuf, &frame).unwrap();

        let mut stream = Vec::new();
        stream.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        stream.extend_from_slice(&payload);

        let mut cursor = Cursor::new(stream);
        let read_payload = read_next_framed_payload(&mut cursor).unwrap().unwrap();
        assert_eq!(read_payload, payload);

        let decoded = decode_client_frame(&ProtocolCodec::Protobuf, &read_payload).unwrap();
        assert_eq!(decoded, frame);

        assert!(read_next_framed_payload(&mut cursor).unwrap().is_none());
    }

    #[test]
    fn truncate_json_payload_fields_truncates_and_annotates() {
        let mut value = serde_json::json!({
            "event": {
                "json_payload": [1, 2, 3, 4],
            }
        });

        truncate_json_payload_fields(&mut value, 2);

        let event = value.get("event").unwrap().as_object().unwrap();
        let payload = event.get("json_payload").unwrap().as_array().unwrap();
        assert_eq!(payload.len(), 2);
        assert_eq!(
            event
                .get("json_payload_byte_len")
                .unwrap()
                .as_u64()
                .unwrap(),
            4
        );
        assert_eq!(
            event
                .get("json_payload_truncated")
                .unwrap()
                .as_bool()
                .unwrap(),
            true
        );
    }

    #[test]
    fn cli_defaults_match_intended_protocol_tooling_behavior() {
        let cli = crate::Cli::try_parse_from(["rn-rs", "protocol", "decode"]).unwrap();
        match cli.command {
            crate::Commands::Protocol(ProtocolCommands::Decode(args)) => {
                assert!(args.framed);
                assert!(args.pretty);
                assert!(!args.verbose);
            }
            _ => panic!("expected `rn-rs protocol decode`"),
        }

        let cli = crate::Cli::try_parse_from(["rn-rs", "protocol", "tap"]).unwrap();
        match cli.command {
            crate::Commands::Protocol(ProtocolCommands::Tap(args)) => {
                assert!(args.pretty);
                assert!(args.status);
                assert!(args.subscribe_event_log);
                assert!(!args.verbose);
            }
            _ => panic!("expected `rn-rs protocol tap`"),
        }
    }
}
