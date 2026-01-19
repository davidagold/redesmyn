# `redesmyn_transport`

Typed transports and codecs for the daemon ↔ control plane boundary.

## Transports

- `in_proc::InProcEndpoint`: in-memory channels (no serialization).
- `framed::FramedEndpoint`: length-delimited frames over an `AsyncRead + AsyncWrite` stream.

## Codecs

- `codec::ProtobufCodec`: default for remote transports.
- `codec::JsonCodec`: opt-in debug/diagnostic mode.

## Framed wire format (stable contract)

When using `FramedEndpoint`, the byte stream is a sequence of frames:

- `u32` big-endian length prefix (4 bytes)
- followed by exactly `len` bytes of payload

The payload bytes are produced/consumed by the selected codec.

## Wiretap

`wiretap` provides helpers to decode raw frame payload bytes into JSON for inspection.

