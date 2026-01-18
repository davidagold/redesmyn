use std::fmt;

use crate::PayloadPolicy;

/// Debug wrapper that enforces the current payload logging policy.
///
/// This intentionally defaults to omitting payload contents.
pub struct Payload<'a, T: ?Sized> {
    value: &'a T,
}

impl<'a, T: ?Sized> Payload<'a, T> {
    pub fn new(value: &'a T) -> Self {
        Self { value }
    }
}

pub fn payload<T: ?Sized>(value: &T) -> Payload<'_, T> {
    Payload::new(value)
}

impl<T: fmt::Debug + ?Sized> fmt::Debug for Payload<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match crate::payload_policy() {
            PayloadPolicy::MetadataOnly => f.write_str("<payload omitted>"),
            PayloadPolicy::Redact => f.write_str("<payload redacted>"),
            PayloadPolicy::Sampled { max_bytes } => {
                let mut rendered = format!("{:?}", self.value);
                let was_truncated = truncate_to_char_boundary(&mut rendered, max_bytes);
                if was_truncated {
                    rendered.push('…');
                }
                f.write_str(&rendered)
            }
            PayloadPolicy::Full => write!(f, "{:?}", self.value),
        }
    }
}

fn truncate_to_char_boundary(s: &mut String, max_bytes: usize) -> bool {
    if s.len() <= max_bytes {
        return false;
    }

    let mut boundary = max_bytes;
    while boundary > 0 && !s.is_char_boundary(boundary) {
        boundary -= 1;
    }

    s.truncate(boundary);
    true
}
