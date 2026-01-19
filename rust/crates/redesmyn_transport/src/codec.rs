use prost::Message;
use redesmyn_ids::{CommandId, HostId, IdBytesLengthError, MsgId, RunId};
use redesmyn_protocol::daemon::{
    CommandAck, CommandAckStatus, DAEMON_PROTOCOL_VERSION, DaemonCommand, DaemonEvent, DaemonFrame,
    DaemonMessage, DispatchCommand, Heartbeat, HelloRequest, HelloResponse, MessageEnvelope,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecKind {
    Json,
    Protobuf,
}

pub trait Codec: Send + Sync {
    fn kind(&self) -> CodecKind;
    fn encode_frame(&self, frame: &DaemonFrame) -> Result<Vec<u8>, CodecError>;
    fn decode_frame(&self, bytes: &[u8]) -> Result<DaemonFrame, CodecError>;
}

#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Protobuf(#[from] prost::DecodeError),
    #[error(transparent)]
    InvalidIdBytes(#[from] IdBytesLengthError),
    #[error("missing protobuf envelope")]
    MissingEnvelope,
    #[error("missing protobuf message")]
    MissingMessage,
    #[error("unsupported protocol version: {version}")]
    UnsupportedProtocolVersion { version: u32 },
    #[error("unknown enum value for {enum_name}: {value}")]
    UnknownEnumValue { enum_name: &'static str, value: i32 },
}

#[derive(Debug, Default, Clone, Copy)]
pub struct JsonCodec;

impl JsonCodec {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Codec for JsonCodec {
    fn kind(&self) -> CodecKind {
        CodecKind::Json
    }

    fn encode_frame(&self, frame: &DaemonFrame) -> Result<Vec<u8>, CodecError> {
        Ok(serde_json::to_vec(frame)?)
    }

    fn decode_frame(&self, bytes: &[u8]) -> Result<DaemonFrame, CodecError> {
        let frame: DaemonFrame = serde_json::from_slice(bytes)?;
        if frame.envelope.protocol_version != DAEMON_PROTOCOL_VERSION {
            return Err(CodecError::UnsupportedProtocolVersion {
                version: frame.envelope.protocol_version,
            });
        }
        Ok(frame)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ProtobufCodec;

impl ProtobufCodec {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Codec for ProtobufCodec {
    fn kind(&self) -> CodecKind {
        CodecKind::Protobuf
    }

    fn encode_frame(&self, frame: &DaemonFrame) -> Result<Vec<u8>, CodecError> {
        let proto_frame = to_proto_frame(frame);
        Ok(proto_frame.encode_to_vec())
    }

    fn decode_frame(&self, bytes: &[u8]) -> Result<DaemonFrame, CodecError> {
        let proto_frame = proto::Frame::decode(bytes)?;
        from_proto_frame(proto_frame)
    }
}

fn to_proto_frame(frame: &DaemonFrame) -> proto::Frame {
    let envelope = Some(to_proto_envelope(&frame.envelope));
    let message = Some(match &frame.message {
        DaemonMessage::HelloRequest(req) => {
            proto::frame::Message::HelloRequest(proto::HelloRequest {
                client_name: req.client_name.clone(),
            })
        }
        DaemonMessage::HelloResponse(resp) => {
            proto::frame::Message::HelloResponse(proto::HelloResponse {
                daemon_name: resp.daemon_name.clone(),
            })
        }
        DaemonMessage::Heartbeat(heartbeat) => proto::frame::Message::Heartbeat(proto::Heartbeat {
            host_id: heartbeat.host_id.to_bytes().to_vec(),
        }),
        DaemonMessage::DispatchCommand(dispatch) => {
            proto::frame::Message::DispatchCommand(proto::DispatchCommand {
                command: to_proto_daemon_command(&dispatch.command) as i32,
            })
        }
        DaemonMessage::CommandAck(ack) => proto::frame::Message::CommandAck(proto::CommandAck {
            status: to_proto_command_ack_status(ack.status) as i32,
        }),
        DaemonMessage::Event(event) => proto::frame::Message::Event(proto::Event {
            event: to_proto_daemon_event(*event) as i32,
        }),
    });

    proto::Frame { envelope, message }
}

fn from_proto_frame(frame: proto::Frame) -> Result<DaemonFrame, CodecError> {
    let envelope = frame.envelope.ok_or(CodecError::MissingEnvelope)?;
    if envelope.protocol_version != DAEMON_PROTOCOL_VERSION {
        return Err(CodecError::UnsupportedProtocolVersion {
            version: envelope.protocol_version,
        });
    }

    let envelope = from_proto_envelope(envelope)?;
    let message = match frame.message.ok_or(CodecError::MissingMessage)? {
        proto::frame::Message::HelloRequest(req) => DaemonMessage::HelloRequest(HelloRequest {
            client_name: req.client_name,
        }),
        proto::frame::Message::HelloResponse(resp) => DaemonMessage::HelloResponse(HelloResponse {
            daemon_name: resp.daemon_name,
        }),
        proto::frame::Message::Heartbeat(heartbeat) => DaemonMessage::Heartbeat(Heartbeat {
            host_id: HostId::try_from_bytes_slice(&heartbeat.host_id)?,
        }),
        proto::frame::Message::DispatchCommand(dispatch) => {
            DaemonMessage::DispatchCommand(DispatchCommand {
                command: from_proto_daemon_command(dispatch.command)?,
            })
        }
        proto::frame::Message::CommandAck(ack) => DaemonMessage::CommandAck(CommandAck {
            status: from_proto_command_ack_status(ack.status)?,
        }),
        proto::frame::Message::Event(event) => {
            DaemonMessage::Event(from_proto_daemon_event(event.event)?)
        }
    };

    Ok(DaemonFrame { envelope, message })
}

fn to_proto_envelope(envelope: &MessageEnvelope) -> proto::Envelope {
    proto::Envelope {
        protocol_version: envelope.protocol_version,
        msg_id: envelope.msg_id.to_bytes().to_vec(),
        in_reply_to: envelope.in_reply_to.map(|id| id.to_bytes().to_vec()),
        command_id: envelope.command_id.map(|id| id.to_bytes().to_vec()),
        run_id: envelope.run_id.map(|id| id.to_bytes().to_vec()),
    }
}

fn from_proto_envelope(envelope: proto::Envelope) -> Result<MessageEnvelope, CodecError> {
    Ok(MessageEnvelope {
        protocol_version: envelope.protocol_version,
        msg_id: MsgId::try_from_bytes_slice(&envelope.msg_id)?,
        in_reply_to: envelope
            .in_reply_to
            .as_deref()
            .map(MsgId::try_from_bytes_slice)
            .transpose()?,
        command_id: envelope
            .command_id
            .as_deref()
            .map(CommandId::try_from_bytes_slice)
            .transpose()?,
        run_id: envelope
            .run_id
            .as_deref()
            .map(RunId::try_from_bytes_slice)
            .transpose()?,
    })
}

fn to_proto_daemon_command(command: &DaemonCommand) -> proto::DaemonCommand {
    match command {
        DaemonCommand::Noop => proto::DaemonCommand::Noop,
    }
}

fn from_proto_daemon_command(value: i32) -> Result<DaemonCommand, CodecError> {
    match proto::DaemonCommand::try_from(value) {
        Ok(proto::DaemonCommand::Noop) => Ok(DaemonCommand::Noop),
        Err(_) => Err(CodecError::UnknownEnumValue {
            enum_name: "DaemonCommand",
            value,
        }),
    }
}

fn to_proto_command_ack_status(status: CommandAckStatus) -> proto::CommandAckStatus {
    match status {
        CommandAckStatus::Accepted => proto::CommandAckStatus::Accepted,
        CommandAckStatus::Rejected => proto::CommandAckStatus::Rejected,
    }
}

fn from_proto_command_ack_status(value: i32) -> Result<CommandAckStatus, CodecError> {
    match proto::CommandAckStatus::try_from(value) {
        Ok(proto::CommandAckStatus::Accepted) => Ok(CommandAckStatus::Accepted),
        Ok(proto::CommandAckStatus::Rejected) => Ok(CommandAckStatus::Rejected),
        Err(_) => Err(CodecError::UnknownEnumValue {
            enum_name: "CommandAckStatus",
            value,
        }),
    }
}

fn to_proto_daemon_event(event: DaemonEvent) -> proto::DaemonEvent {
    match event {
        DaemonEvent::Noop => proto::DaemonEvent::Noop,
    }
}

fn from_proto_daemon_event(value: i32) -> Result<DaemonEvent, CodecError> {
    match proto::DaemonEvent::try_from(value) {
        Ok(proto::DaemonEvent::Noop) => Ok(DaemonEvent::Noop),
        Err(_) => Err(CodecError::UnknownEnumValue {
            enum_name: "DaemonEvent",
            value,
        }),
    }
}

// Temporary (Domain 0 scaffolding): inline `prost` message definitions.
//
// This will be replaced by the T-10 Protobuf schema + codegen pipeline so the
// wire format is driven by .proto files instead of hand-written Rust structs.
mod proto {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Envelope {
        #[prost(uint32, tag = "1")]
        pub protocol_version: u32,
        #[prost(bytes = "vec", tag = "2")]
        pub msg_id: Vec<u8>,
        #[prost(bytes = "vec", optional, tag = "3")]
        pub in_reply_to: Option<Vec<u8>>,
        #[prost(bytes = "vec", optional, tag = "4")]
        pub command_id: Option<Vec<u8>>,
        #[prost(bytes = "vec", optional, tag = "5")]
        pub run_id: Option<Vec<u8>>,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct HelloRequest {
        #[prost(string, tag = "1")]
        pub client_name: String,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct HelloResponse {
        #[prost(string, tag = "1")]
        pub daemon_name: String,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Heartbeat {
        #[prost(bytes = "vec", tag = "1")]
        pub host_id: Vec<u8>,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum DaemonCommand {
        Noop = 0,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct DispatchCommand {
        #[prost(enumeration = "DaemonCommand", tag = "1")]
        pub command: i32,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum CommandAckStatus {
        Accepted = 0,
        Rejected = 1,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct CommandAck {
        #[prost(enumeration = "CommandAckStatus", tag = "1")]
        pub status: i32,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum DaemonEvent {
        Noop = 0,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Event {
        #[prost(enumeration = "DaemonEvent", tag = "1")]
        pub event: i32,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Frame {
        #[prost(message, optional, tag = "1")]
        pub envelope: Option<Envelope>,
        #[prost(oneof = "frame::Message", tags = "2, 3, 4, 5, 6, 7")]
        pub message: Option<frame::Message>,
    }

    pub mod frame {
        #[derive(Clone, PartialEq, ::prost::Oneof)]
        pub enum Message {
            #[prost(message, tag = "2")]
            HelloRequest(super::HelloRequest),
            #[prost(message, tag = "3")]
            HelloResponse(super::HelloResponse),
            #[prost(message, tag = "4")]
            Heartbeat(super::Heartbeat),
            #[prost(message, tag = "5")]
            DispatchCommand(super::DispatchCommand),
            #[prost(message, tag = "6")]
            CommandAck(super::CommandAck),
            #[prost(message, tag = "7")]
            Event(super::Event),
        }
    }
}
