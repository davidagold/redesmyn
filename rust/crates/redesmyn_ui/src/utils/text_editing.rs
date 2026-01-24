fn clamp_to_char_boundary(text: &str, mut offset: usize) -> usize {
    offset = offset.min(text.len());
    while offset > 0 && !text.is_char_boundary(offset) {
        offset -= 1;
    }
    offset
}

fn is_word_char(ch: char) -> bool {
    ch.is_alphanumeric() || ch == '_'
}

/// Returns the start byte offset of the previous "word" relative to `offset`.
///
/// Semantics are deterministic and editor-like:
/// - "word" chars are `char::is_alphanumeric()` plus `_`
/// - everything else is a separator
///
/// The algorithm:
/// 1) skip separators to the left
/// 2) then skip word chars to the left
pub(crate) fn previous_word_boundary(text: &str, offset: usize) -> usize {
    let mut offset = clamp_to_char_boundary(text, offset);
    if offset == 0 {
        return 0;
    }

    while offset > 0 {
        let ch = text[..offset].chars().next_back().unwrap();
        if is_word_char(ch) {
            break;
        }
        offset -= ch.len_utf8();
    }

    while offset > 0 {
        let ch = text[..offset].chars().next_back().unwrap();
        if !is_word_char(ch) {
            break;
        }
        offset -= ch.len_utf8();
    }

    offset
}

/// Returns the end byte offset of the next "word" relative to `offset`.
///
/// See `previous_word_boundary` for the word/separator definition.
pub(crate) fn next_word_boundary(text: &str, offset: usize) -> usize {
    let len = text.len();
    let mut offset = clamp_to_char_boundary(text, offset);
    if offset >= len {
        return len;
    }

    while offset < len {
        let ch = text[offset..].chars().next().unwrap();
        if is_word_char(ch) {
            break;
        }
        offset += ch.len_utf8();
    }

    while offset < len {
        let ch = text[offset..].chars().next().unwrap();
        if !is_word_char(ch) {
            break;
        }
        offset += ch.len_utf8();
    }

    offset
}

pub(crate) fn line_start_offset(text: &str, offset: usize) -> usize {
    let offset = clamp_to_char_boundary(text, offset);
    text[..offset].rfind('\n').map(|ix| ix + 1).unwrap_or(0)
}

pub(crate) fn line_end_offset(text: &str, offset: usize) -> usize {
    let offset = clamp_to_char_boundary(text, offset);
    text[offset..]
        .find('\n')
        .map(|ix| offset + ix)
        .unwrap_or(text.len())
}

#[cfg(test)]
mod tests {
    use super::{line_end_offset, line_start_offset, next_word_boundary, previous_word_boundary};
    use std::ops::Range;

    fn delete_word_backward(text: &str, cursor: usize) -> (String, usize) {
        let start = previous_word_boundary(text, cursor);
        let mut text = text.to_string();
        text.replace_range(start..cursor, "");
        (text, start)
    }

    fn delete_word_forward(text: &str, cursor: usize) -> (String, usize) {
        let end = next_word_boundary(text, cursor);
        let mut text = text.to_string();
        text.replace_range(cursor..end, "");
        (text, cursor)
    }

    fn extend_selection(
        mut range: Range<usize>,
        mut reversed: bool,
        offset: usize,
    ) -> (Range<usize>, bool) {
        if reversed {
            range.start = offset;
        } else {
            range.end = offset;
        }

        if range.end < range.start {
            reversed = !reversed;
            range = range.end..range.start;
        }

        (range, reversed)
    }

    #[test]
    fn word_boundaries_ascii() {
        let text = "hello world";
        assert_eq!(next_word_boundary(text, 0), 5);
        assert_eq!(next_word_boundary(text, 5), 11);
        assert_eq!(previous_word_boundary(text, 11), 6);
        assert_eq!(previous_word_boundary(text, 6), 0);
    }

    #[test]
    fn word_boundaries_treat_underscore_as_word() {
        let text = "foo_bar baz";
        assert_eq!(next_word_boundary(text, 0), "foo_bar".len());
        assert_eq!(previous_word_boundary(text, "foo_bar".len()), 0);
    }

    #[test]
    fn word_boundaries_unicode_alphanumeric() {
        let text = "héllo 世界";
        let world_start = text.find('世').unwrap();
        assert_eq!(previous_word_boundary(text, world_start), 0);
        assert_eq!(next_word_boundary(text, 0), text.find(' ').unwrap());
    }

    #[test]
    fn delete_word_backward_and_forward() {
        let text = "hello world";
        assert_eq!(delete_word_backward(text, 11).0, "hello ");
        assert_eq!(delete_word_backward(text, 6).0, "world");

        assert_eq!(delete_word_forward(text, 0).0, " world");
        assert_eq!(delete_word_forward(text, 5).0, "hello");
    }

    #[test]
    fn line_boundaries_respect_newlines() {
        let text = "abc def\nghi jkl\nmn";

        assert_eq!(line_start_offset(text, 0), 0);
        assert_eq!(line_end_offset(text, 0), 7);
        assert_eq!(line_start_offset(text, 4), 0);
        assert_eq!(line_end_offset(text, 4), 7);

        // Cursor at newline belongs to the preceding line.
        assert_eq!(line_start_offset(text, 7), 0);
        assert_eq!(line_end_offset(text, 7), 7);

        assert_eq!(line_start_offset(text, 8), 8);
        assert_eq!(line_end_offset(text, 8), 15);

        assert_eq!(line_start_offset(text, 16), 16);
        assert_eq!(line_end_offset(text, 16), 18);
    }

    #[test]
    fn selection_extension_by_word_and_line() {
        let text = "hello world\nabc def";
        let line2_start = text.find('\n').unwrap() + 1;

        // Shift-alt-left at end selects the final word.
        let cursor = text.len();
        let (range, reversed) =
            extend_selection(cursor..cursor, false, previous_word_boundary(text, cursor));
        assert_eq!(range, (cursor - "def".len())..cursor);
        assert!(reversed);

        // Shift-cmd-left on the second line selects to line start.
        let cursor = line2_start + "abc".len();
        let (range, reversed) =
            extend_selection(cursor..cursor, false, line_start_offset(text, cursor));
        assert_eq!(range, line2_start..cursor);
        assert!(reversed);
    }
}
