use unicode_segmentation::UnicodeSegmentation;

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

pub(crate) fn word_range_at(text: &str, offset: usize) -> std::ops::Range<usize> {
    if text.is_empty() {
        return 0..0;
    }

    let mut offset = clamp_to_char_boundary(text, offset);
    if offset == text.len() {
        offset = clamp_to_char_boundary(text, offset.saturating_sub(1));
    }

    let Some(ch) = text[offset..].chars().next() else {
        return 0..0;
    };
    if ch == '\n' {
        return offset..offset;
    }

    let target_is_word = is_word_char(ch);

    let mut start = offset;
    while start > 0 {
        let prev = text[..start].chars().next_back().unwrap();
        if prev == '\n' || is_word_char(prev) != target_is_word {
            break;
        }
        start -= prev.len_utf8();
    }

    let mut end = offset;
    while end < text.len() {
        let next = text[end..].chars().next().unwrap();
        if next == '\n' || is_word_char(next) != target_is_word {
            break;
        }
        end += next.len_utf8();
    }

    start..end
}

pub(crate) fn previous_grapheme_boundary(text: &str, offset: usize) -> usize {
    let offset = clamp_to_char_boundary(text, offset);
    if offset == 0 {
        return 0;
    }

    text.grapheme_indices(true)
        .take_while(|(ix, _)| *ix < offset)
        .last()
        .map(|(ix, _)| ix)
        .unwrap_or(0)
}

pub(crate) fn next_grapheme_boundary(text: &str, offset: usize) -> usize {
    let len = text.len();
    let offset = clamp_to_char_boundary(text, offset);
    if offset >= len {
        return len;
    }

    let mut cluster_start = 0usize;
    let mut cluster_len = 0usize;
    for (start, cluster) in text.grapheme_indices(true) {
        if start > offset {
            break;
        }
        cluster_start = start;
        cluster_len = cluster.len();
    }

    (cluster_start + cluster_len).min(len)
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
    use super::{
        line_end_offset, line_start_offset, next_grapheme_boundary, next_word_boundary,
        previous_grapheme_boundary, previous_word_boundary,
    };
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

    #[test]
    fn grapheme_boundaries_combining_mark() {
        let text = "e\u{301}";
        assert_eq!(next_grapheme_boundary(text, 0), text.len());
        assert_eq!(next_grapheme_boundary(text, "e".len()), text.len());
        assert_eq!(previous_grapheme_boundary(text, text.len()), 0);
        assert_eq!(previous_grapheme_boundary(text, "e".len()), 0);
    }

    #[test]
    fn grapheme_boundaries_zwj_emoji() {
        let emoji = "👩‍👩‍👧‍👧";
        assert_eq!(next_grapheme_boundary(emoji, 0), emoji.len());
        assert_eq!(next_grapheme_boundary(emoji, "👩".len()), emoji.len());
        assert_eq!(previous_grapheme_boundary(emoji, emoji.len()), 0);
        assert_eq!(previous_grapheme_boundary(emoji, "👩".len()), 0);
    }

    #[test]
    fn grapheme_boundaries_multiple_clusters() {
        let emoji = "👩‍👩‍👧‍👧";
        let text = format!("a{emoji}b");
        let emoji_start = "a".len();
        let emoji_end = emoji_start + emoji.len();

        assert_eq!(next_grapheme_boundary(&text, 0), 1);
        assert_eq!(next_grapheme_boundary(&text, emoji_start), emoji_end);
        assert_eq!(next_grapheme_boundary(&text, emoji_end), text.len());

        assert_eq!(previous_grapheme_boundary(&text, text.len()), emoji_end);
        assert_eq!(previous_grapheme_boundary(&text, emoji_end), emoji_start);
        assert_eq!(previous_grapheme_boundary(&text, emoji_start), 0);
    }
}
