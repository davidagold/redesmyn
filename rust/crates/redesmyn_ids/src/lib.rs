//! ULID-backed identifier newtypes.
//!
//! # Encoding strategy
//!
//! - In memory: `#[repr(transparent)]` newtypes over [`ulid::Ulid`].
//! - JSON (serde): canonical ULID string (e.g. `"01ARZ3NDEKTSV4RRFFQ69G5FAV"`).
//! - Protobuf: prefer 16 raw bytes (see [`WorkspaceId::to_bytes`] / [`WorkspaceId::from_bytes`]).
//! - SQLite (`sqlx`): `BLOB(16)` when the crate feature `sqlx` is enabled.

#![forbid(unsafe_code)]

use std::{fmt, str::FromStr};

use ulid::Ulid;

/// An error produced when parsing an ID from a ULID string.
#[derive(Debug)]
pub struct ParseIdError {
    id_type: &'static str,
    input: String,
    source: ulid::DecodeError,
}

impl ParseIdError {
    fn new(id_type: &'static str, input: &str, source: ulid::DecodeError) -> Self {
        Self {
            id_type,
            input: input.to_owned(),
            source,
        }
    }
}

impl fmt::Display for ParseIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid {id_type} ULID: {input} ({source})",
            id_type = self.id_type,
            input = self.input,
            source = self.source
        )
    }
}

impl std::error::Error for ParseIdError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// An error produced when decoding an ID from a byte slice of the wrong length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdBytesLengthError {
    id_type: &'static str,
    actual_len: usize,
}

impl IdBytesLengthError {
    fn new(id_type: &'static str, actual_len: usize) -> Self {
        Self {
            id_type,
            actual_len,
        }
    }

    #[must_use]
    pub fn id_type(&self) -> &'static str {
        self.id_type
    }

    #[must_use]
    pub fn actual_len(&self) -> usize {
        self.actual_len
    }
}

impl fmt::Display for IdBytesLengthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid {id_type} bytes length: expected 16, got {actual_len}",
            id_type = self.id_type,
            actual_len = self.actual_len
        )
    }
}

impl std::error::Error for IdBytesLengthError {}

macro_rules! ulid_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        #[repr(transparent)]
        pub struct $name(Ulid);

        impl $name {
            pub const BYTE_LEN: usize = 16;

            #[must_use]
            pub fn new() -> Self {
                Self(Ulid::new())
            }

            #[must_use]
            pub fn from_ulid(value: Ulid) -> Self {
                Self(value)
            }

            #[must_use]
            pub fn as_ulid(&self) -> &Ulid {
                &self.0
            }

            #[must_use]
            pub fn into_ulid(self) -> Ulid {
                self.0
            }

            #[must_use]
            pub fn from_bytes(bytes: [u8; 16]) -> Self {
                Self(Ulid::from_bytes(bytes))
            }

            #[must_use]
            pub fn to_bytes(self) -> [u8; 16] {
                self.0.to_bytes()
            }

            pub fn try_from_bytes_slice(bytes: &[u8]) -> Result<Self, IdBytesLengthError> {
                if bytes.len() != 16 {
                    return Err(IdBytesLengthError::new(stringify!($name), bytes.len()));
                }

                let mut array = [0_u8; 16];
                array.copy_from_slice(bytes);
                Ok(Self::from_bytes(array))
            }
        }

        impl TryFrom<&[u8]> for $name {
            type Error = IdBytesLengthError;

            fn try_from(value: &[u8]) -> Result<Self, Self::Error> {
                Self::try_from_bytes_slice(value)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({})", stringify!($name), self.0)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }

        impl FromStr for $name {
            type Err = ParseIdError;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                Ulid::from_str(s)
                    .map(Self)
                    .map_err(|err| ParseIdError::new(stringify!($name), s, err))
            }
        }

        impl From<Ulid> for $name {
            fn from(value: Ulid) -> Self {
                Self(value)
            }
        }

        impl From<$name> for Ulid {
            fn from(value: $name) -> Self {
                value.0
            }
        }

        impl serde::Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                serializer.collect_str(&self.0)
            }
        }

        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                let s = <std::borrow::Cow<'de, str> as serde::Deserialize<'de>>::deserialize(
                    deserializer,
                )?;
                s.parse().map_err(serde::de::Error::custom)
            }
        }

        #[cfg(feature = "sqlx")]
        impl sqlx::Type<sqlx::sqlite::Sqlite> for $name {
            fn type_info() -> sqlx::sqlite::SqliteTypeInfo {
                <Vec<u8> as sqlx::Type<sqlx::sqlite::Sqlite>>::type_info()
            }

            fn compatible(ty: &sqlx::sqlite::SqliteTypeInfo) -> bool {
                <Vec<u8> as sqlx::Type<sqlx::sqlite::Sqlite>>::compatible(ty)
            }
        }

        #[cfg(feature = "sqlx")]
        impl<'q> sqlx::Encode<'q, sqlx::sqlite::Sqlite> for $name {
            fn encode_by_ref(
                &self,
                buf: &mut <sqlx::sqlite::Sqlite as sqlx::Database>::ArgumentBuffer<'q>,
            ) -> Result<sqlx::encode::IsNull, sqlx::error::BoxDynError> {
                use sqlx::sqlite::SqliteArgumentValue;
                buf.push(SqliteArgumentValue::Blob(std::borrow::Cow::Owned(
                    self.to_bytes().to_vec(),
                )));
                Ok(sqlx::encode::IsNull::No)
            }

            fn size_hint(&self) -> usize {
                Self::BYTE_LEN
            }
        }

        #[cfg(feature = "sqlx")]
        impl<'r> sqlx::Decode<'r, sqlx::sqlite::Sqlite> for $name {
            fn decode(
                value: sqlx::sqlite::SqliteValueRef<'r>,
            ) -> Result<Self, sqlx::error::BoxDynError> {
                let bytes = <&[u8] as sqlx::Decode<sqlx::sqlite::Sqlite>>::decode(value)?;
                Ok(Self::try_from_bytes_slice(bytes)?)
            }
        }
    };
}

ulid_id!(/// Identifier for a workspace. Workspace is the top-level container in Rust-land.
WorkspaceId);
ulid_id!(/// Identifier for a repo registered with the daemon/control plane.
RepoId);
ulid_id!(/// Identifier for an epic (graph root) within a workspace.
EpicId);
ulid_id!(/// Identifier for a task node within an epic.
TaskId);
ulid_id!(/// Identifier for a command/run lifecycle execution.
RunId);
ulid_id!(/// Identifier for a host (machine) in a multi-host topology.
HostId);
ulid_id!(/// Identifier for a command (idempotency key and/or persisted command record).
CommandId);
ulid_id!(/// Identifier for an event (append-only event log record).
EventId);
ulid_id!(/// Identifier for a protocol message (idempotency + dedupe key).
MsgId);

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! assert_roundtrip {
        ($ty:ty) => {{
            let id = <$ty>::new();
            let s = id.to_string();
            let parsed: $ty = s.parse().unwrap();
            assert_eq!(parsed, id);
        }};
    }

    #[test]
    fn parse_format_roundtrips() {
        assert_roundtrip!(WorkspaceId);
        assert_roundtrip!(RepoId);
        assert_roundtrip!(EpicId);
        assert_roundtrip!(TaskId);
        assert_roundtrip!(RunId);
        assert_roundtrip!(HostId);
        assert_roundtrip!(CommandId);
        assert_roundtrip!(EventId);
        assert_roundtrip!(MsgId);
    }

    #[test]
    fn serde_roundtrip_json_string() {
        let id = TaskId::new();

        let json = serde_json::to_string(&id).unwrap();
        assert!(
            json.starts_with('\"') && json.ends_with('\"'),
            "json={json}"
        );

        let parsed: TaskId = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, id);
    }

    #[test]
    fn bytes_roundtrip() {
        let id = WorkspaceId::new();
        let bytes = id.to_bytes();
        let decoded = WorkspaceId::try_from_bytes_slice(&bytes).unwrap();
        assert_eq!(decoded, id);
    }

    #[cfg(feature = "sqlx")]
    mod sqlx_tests {
        use super::*;

        #[tokio::test]
        async fn sqlx_roundtrip_blob16() {
            let pool = sqlx::sqlite::SqlitePoolOptions::new()
                .max_connections(1)
                .connect(":memory:")
                .await
                .unwrap();

            sqlx::query("CREATE TABLE ids (id BLOB(16) PRIMARY KEY NOT NULL)")
                .execute(&pool)
                .await
                .unwrap();

            let id = WorkspaceId::new();
            sqlx::query("INSERT INTO ids (id) VALUES (?)")
                .bind(id)
                .execute(&pool)
                .await
                .unwrap();

            let (loaded_id,): (WorkspaceId,) = sqlx::query_as("SELECT id FROM ids LIMIT 1")
                .fetch_one(&pool)
                .await
                .unwrap();

            assert_eq!(loaded_id, id);
        }
    }
}
