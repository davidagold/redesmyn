//! Minimal markdown parsing for GPUI surfaces.
//!
//! This crate is intentionally GPUI-free so it can be unit tested and reused by UI crates without
//! pulling in rendering dependencies.

#![forbid(unsafe_code)]

use std::ops::Range;

use pulldown_cmark::{CodeBlockKind, Event, HeadingLevel, Parser, Tag, TagEnd};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MarkdownParseOptions {
    pub max_input_len: usize,
}

impl Default for MarkdownParseOptions {
    fn default() -> Self {
        Self {
            max_input_len: 200_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarkdownTruncation {
    pub original_len: usize,
    pub rendered_len: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarkdownDoc {
    pub blocks: Vec<MarkdownBlock>,
    pub truncation: Option<MarkdownTruncation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MarkdownSourceRange {
    pub start: u32,
    pub end: u32,
}

impl MarkdownSourceRange {
    fn from_offset_range(range: &Range<usize>) -> Option<Self> {
        let start = u32::try_from(range.start).ok()?;
        let end = u32::try_from(range.end).ok()?;
        Some(Self { start, end })
    }

    fn union(self, other: Self) -> Self {
        Self {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MarkdownBlock {
    Paragraph {
        range: Option<MarkdownSourceRange>,
        content: Vec<MarkdownInline>,
    },
    Heading {
        range: Option<MarkdownSourceRange>,
        level: u8,
        content: Vec<MarkdownInline>,
    },
    CodeBlock {
        range: Option<MarkdownSourceRange>,
        language: Option<String>,
        info_raw: Option<String>,
        code: String,
    },
    BlockQuote {
        range: Option<MarkdownSourceRange>,
        content: Vec<MarkdownBlock>,
    },
    List {
        range: Option<MarkdownSourceRange>,
        ordered: bool,
        start: Option<u64>,
        items: Vec<MarkdownListItem>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarkdownListItem {
    pub range: Option<MarkdownSourceRange>,
    pub blocks: Vec<MarkdownBlock>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MarkdownInline {
    Text {
        range: Option<MarkdownSourceRange>,
        text: String,
    },
    Emphasis {
        range: Option<MarkdownSourceRange>,
        content: Vec<MarkdownInline>,
    },
    Strong {
        range: Option<MarkdownSourceRange>,
        content: Vec<MarkdownInline>,
    },
    Code {
        range: Option<MarkdownSourceRange>,
        code: String,
    },
    Link {
        range: Option<MarkdownSourceRange>,
        destination: String,
        content: Vec<MarkdownInline>,
    },
    SoftBreak {
        range: Option<MarkdownSourceRange>,
    },
    HardBreak {
        range: Option<MarkdownSourceRange>,
    },
}

pub fn parse_markdown(input: &str, options: MarkdownParseOptions) -> MarkdownDoc {
    let (prefix, truncation) = truncate(input, options.max_input_len);
    let parser = Parser::new(prefix).into_offset_iter();
    MarkdownDoc {
        blocks: parse_blocks(parser),
        truncation,
    }
}

fn truncate(input: &str, max_len: usize) -> (&str, Option<MarkdownTruncation>) {
    if input.len() <= max_len {
        return (input, None);
    }

    let mut cut = max_len.min(input.len());
    while cut > 0 && !input.is_char_boundary(cut) {
        cut -= 1;
    }

    (
        &input[..cut],
        Some(MarkdownTruncation {
            original_len: input.len(),
            rendered_len: cut,
        }),
    )
}

#[derive(Debug, Default, Clone, Copy)]
struct RangeAccumulator {
    range: Option<MarkdownSourceRange>,
}

impl RangeAccumulator {
    fn observe_offset_range(&mut self, range: &Range<usize>) {
        if let Some(span) = MarkdownSourceRange::from_offset_range(range) {
            self.observe(span);
        }
    }

    fn observe(&mut self, span: MarkdownSourceRange) {
        self.range = Some(match self.range {
            Some(existing) => existing.union(span),
            None => span,
        });
    }

    fn finish(self) -> Option<MarkdownSourceRange> {
        self.range
    }
}

fn parse_blocks<'a>(
    events: impl IntoIterator<Item = (Event<'a>, Range<usize>)>,
) -> Vec<MarkdownBlock> {
    let mut stack = vec![BlockContext::Root {
        blocks: Vec::new(),
        tight_inlines: None,
    }];

    for (event, event_range) in events {
        match event {
            Event::Start(tag) => {
                if tag_is_block_boundary(&tag)
                    && let Some(parent) = stack.last_mut()
                {
                    parent.flush_tight_paragraph();
                }

                match tag {
                    Tag::Paragraph => {
                        let mut inlines = InlineBuilder::new();
                        inlines.observe_offset_range(&event_range);
                        stack.push(BlockContext::Paragraph { inlines });
                    }
                    Tag::Heading { level, .. } => {
                        let mut inlines = InlineBuilder::new();
                        inlines.observe_offset_range(&event_range);
                        stack.push(BlockContext::Heading {
                            level: heading_level(level),
                            inlines,
                        });
                    }
                    Tag::BlockQuote(_) => {
                        let mut range = RangeAccumulator::default();
                        range.observe_offset_range(&event_range);
                        stack.push(BlockContext::BlockQuote {
                            range,
                            blocks: Vec::new(),
                            tight_inlines: None,
                        });
                    }
                    Tag::List(start) => {
                        let mut range = RangeAccumulator::default();
                        range.observe_offset_range(&event_range);
                        stack.push(BlockContext::List {
                            range,
                            ordered: start.is_some(),
                            start,
                            items: Vec::new(),
                        });
                    }
                    Tag::Item => {
                        let mut range = RangeAccumulator::default();
                        range.observe_offset_range(&event_range);
                        stack.push(BlockContext::ListItem {
                            range,
                            blocks: Vec::new(),
                            tight_inlines: None,
                        });
                    }
                    Tag::CodeBlock(kind) => {
                        let info = code_block_info(kind);
                        let mut range = RangeAccumulator::default();
                        range.observe_offset_range(&event_range);
                        stack.push(BlockContext::CodeBlock {
                            range,
                            language: info.language,
                            info_raw: info.info_raw,
                            code: String::new(),
                        });
                    }
                    Tag::Emphasis => {
                        if let Some(builder) = stack
                            .last_mut()
                            .and_then(BlockContext::ensure_inline_builder_mut)
                        {
                            builder.start_emphasis(&event_range);
                        }
                    }
                    Tag::Strong => {
                        if let Some(builder) = stack
                            .last_mut()
                            .and_then(BlockContext::ensure_inline_builder_mut)
                        {
                            builder.start_strong(&event_range);
                        }
                    }
                    Tag::Link { dest_url, .. } => {
                        if let Some(builder) = stack
                            .last_mut()
                            .and_then(BlockContext::ensure_inline_builder_mut)
                        {
                            builder.start_link(dest_url.to_string(), &event_range);
                        }
                    }
                    _ => {}
                }
            }
            Event::End(tag_end) => match tag_end {
                TagEnd::Paragraph => {
                    if let Some(BlockContext::Paragraph { mut inlines }) = stack.pop() {
                        inlines.observe_offset_range(&event_range);
                        let (content, range) = inlines.finish_with_range();
                        push_block(
                            &mut stack,
                            MarkdownBlock::Paragraph { range, content },
                        );
                    }
                }
                TagEnd::Heading(_) => {
                    if let Some(BlockContext::Heading { level, mut inlines }) = stack.pop() {
                        inlines.observe_offset_range(&event_range);
                        let (content, range) = inlines.finish_with_range();
                        push_block(
                            &mut stack,
                            MarkdownBlock::Heading {
                                range,
                                level,
                                content,
                            },
                        );
                    }
                }
                TagEnd::BlockQuote(_) => {
                    if let Some(BlockContext::BlockQuote {
                        mut range,
                        mut blocks,
                        mut tight_inlines,
                    }) = stack.pop()
                    {
                        range.observe_offset_range(&event_range);
                        if let Some(flushed) = flush_tight_inlines(&mut blocks, &mut tight_inlines)
                        {
                            range.observe(flushed);
                        }
                        push_block(
                            &mut stack,
                            MarkdownBlock::BlockQuote {
                                range: range.finish(),
                                content: blocks,
                            },
                        );
                    }
                }
                TagEnd::List(_) => {
                    if let Some(BlockContext::List {
                        mut range,
                        ordered,
                        start,
                        items,
                    }) = stack.pop()
                    {
                        range.observe_offset_range(&event_range);
                        push_block(
                            &mut stack,
                            MarkdownBlock::List {
                                range: range.finish(),
                                ordered,
                                start,
                                items,
                            },
                        );
                    }
                }
                TagEnd::Item => {
                    if let Some(BlockContext::ListItem {
                        mut range,
                        mut blocks,
                        mut tight_inlines,
                    }) = stack.pop()
                    {
                        range.observe_offset_range(&event_range);
                        if let Some(flushed) = flush_tight_inlines(&mut blocks, &mut tight_inlines)
                        {
                            range.observe(flushed);
                        }
                        push_item(
                            &mut stack,
                            MarkdownListItem {
                                range: range.finish(),
                                blocks,
                            },
                        );
                    }
                }
                TagEnd::CodeBlock => {
                    if let Some(BlockContext::CodeBlock {
                        mut range,
                        language,
                        info_raw,
                        code,
                    }) = stack.pop()
                    {
                        range.observe_offset_range(&event_range);
                        push_block(
                            &mut stack,
                            MarkdownBlock::CodeBlock {
                                range: range.finish(),
                                language,
                                info_raw,
                                code,
                            },
                        );
                    }
                }
                TagEnd::Emphasis => {
                    if let Some(builder) = stack
                        .last_mut()
                        .and_then(BlockContext::ensure_inline_builder_mut)
                    {
                        builder.end_emphasis(&event_range);
                    }
                }
                TagEnd::Strong => {
                    if let Some(builder) = stack
                        .last_mut()
                        .and_then(BlockContext::ensure_inline_builder_mut)
                    {
                        builder.end_strong(&event_range);
                    }
                }
                TagEnd::Link => {
                    if let Some(builder) = stack
                        .last_mut()
                        .and_then(BlockContext::ensure_inline_builder_mut)
                    {
                        builder.end_link(&event_range);
                    }
                }
                _ => {}
            },
            Event::Text(text) => match stack.last_mut() {
                Some(BlockContext::CodeBlock { range, code, .. }) => {
                    range.observe_offset_range(&event_range);
                    code.push_str(text.as_ref());
                }
                Some(ctx) => ctx.push_inline(MarkdownInline::Text {
                    range: MarkdownSourceRange::from_offset_range(&event_range),
                    text: text.to_string(),
                }),
                None => {}
            },
            Event::Code(code) => {
                if let Some(ctx) = stack.last_mut() {
                    ctx.push_inline(MarkdownInline::Code {
                        range: MarkdownSourceRange::from_offset_range(&event_range),
                        code: code.to_string(),
                    });
                }
            }
            Event::SoftBreak => {
                if let Some(ctx) = stack.last_mut() {
                    ctx.push_inline(MarkdownInline::SoftBreak {
                        range: MarkdownSourceRange::from_offset_range(&event_range),
                    });
                }
            }
            Event::HardBreak => {
                if let Some(ctx) = stack.last_mut() {
                    ctx.push_inline(MarkdownInline::HardBreak {
                        range: MarkdownSourceRange::from_offset_range(&event_range),
                    });
                }
            }
            Event::Html(html) | Event::InlineHtml(html) => {
                let _ = html;
            }
            Event::InlineMath(math) | Event::DisplayMath(math) => {
                if let Some(ctx) = stack.last_mut() {
                    ctx.push_inline(MarkdownInline::Text {
                        range: MarkdownSourceRange::from_offset_range(&event_range),
                        text: math.to_string(),
                    });
                }
            }
            Event::Rule | Event::FootnoteReference(_) | Event::TaskListMarker(_) => {}
        }
    }

    while stack.len() > 1 {
        match stack.pop() {
            Some(BlockContext::Paragraph { mut inlines }) => {
                let (content, range) = inlines.finish_with_range();
                push_block(&mut stack, MarkdownBlock::Paragraph { range, content });
            }
            Some(BlockContext::Heading { level, mut inlines }) => {
                let (content, range) = inlines.finish_with_range();
                push_block(
                    &mut stack,
                    MarkdownBlock::Heading {
                        range,
                        level,
                        content,
                    },
                );
            }
            Some(BlockContext::CodeBlock {
                range,
                language,
                info_raw,
                code,
            }) => {
                push_block(
                    &mut stack,
                    MarkdownBlock::CodeBlock {
                        range: range.finish(),
                        language,
                        info_raw,
                        code,
                    },
                );
            }
            Some(BlockContext::BlockQuote {
                mut range,
                mut blocks,
                mut tight_inlines,
            }) => {
                if let Some(flushed) = flush_tight_inlines(&mut blocks, &mut tight_inlines) {
                    range.observe(flushed);
                }
                push_block(
                    &mut stack,
                    MarkdownBlock::BlockQuote {
                        range: range.finish(),
                        content: blocks,
                    },
                );
            }
            Some(BlockContext::List {
                range,
                ordered,
                start,
                items,
            }) => {
                push_block(
                    &mut stack,
                    MarkdownBlock::List {
                        range: range.finish(),
                        ordered,
                        start,
                        items,
                    },
                );
            }
            Some(BlockContext::ListItem {
                mut range,
                mut blocks,
                mut tight_inlines,
            }) => {
                if let Some(flushed) = flush_tight_inlines(&mut blocks, &mut tight_inlines) {
                    range.observe(flushed);
                }
                push_item(
                    &mut stack,
                    MarkdownListItem {
                        range: range.finish(),
                        blocks,
                    },
                );
            }
            _ => {}
        }
    }

    match stack.pop() {
        Some(BlockContext::Root {
            mut blocks,
            mut tight_inlines,
        }) => {
            flush_tight_inlines(&mut blocks, &mut tight_inlines);
            blocks
        }
        _ => Vec::new(),
    }
}

fn tag_is_block_boundary(tag: &Tag<'_>) -> bool {
    matches!(
        tag,
        Tag::Paragraph
            | Tag::Heading { .. }
            | Tag::BlockQuote(_)
            | Tag::List(_)
            | Tag::Item
            | Tag::CodeBlock(_)
    )
}

fn flush_tight_inlines(
    blocks: &mut Vec<MarkdownBlock>,
    tight_inlines: &mut Option<InlineBuilder>,
) -> Option<MarkdownSourceRange> {
    let Some(mut inlines) = tight_inlines.take() else {
        return None;
    };

    let (inlines, range) = inlines.finish_with_range();
    if inlines.is_empty() {
        return None;
    }

    blocks.push(MarkdownBlock::Paragraph {
        range,
        content: inlines,
    });
    range
}

fn heading_level(level: HeadingLevel) -> u8 {
    match level {
        HeadingLevel::H1 => 1,
        HeadingLevel::H2 => 2,
        HeadingLevel::H3 => 3,
        HeadingLevel::H4 => 4,
        HeadingLevel::H5 => 5,
        HeadingLevel::H6 => 6,
    }
}

#[derive(Debug)]
struct MarkdownCodeBlockInfo {
    language: Option<String>,
    info_raw: Option<String>,
}

fn code_block_info(kind: CodeBlockKind<'_>) -> MarkdownCodeBlockInfo {
    match kind {
        CodeBlockKind::Indented => MarkdownCodeBlockInfo {
            language: None,
            info_raw: None,
        },
        CodeBlockKind::Fenced(info_raw) => {
            let info_raw = info_raw.to_string();
            let trimmed = info_raw.trim();
            if trimmed.is_empty() {
                return MarkdownCodeBlockInfo {
                    language: None,
                    info_raw: None,
                };
            }

            MarkdownCodeBlockInfo {
                language: extract_fenced_language(trimmed),
                info_raw: Some(trimmed.to_string()),
            }
        }
    }
}

fn extract_fenced_language(info_raw: &str) -> Option<String> {
    let first_token = info_raw.split_whitespace().next().unwrap_or_default();
    let primary = first_token
        .split_once(',')
        .map_or(first_token, |(prefix, _)| prefix)
        .trim();
    if primary.is_empty() {
        return None;
    }

    Some(primary.to_ascii_lowercase())
}

fn block_source_range(block: &MarkdownBlock) -> Option<MarkdownSourceRange> {
    match block {
        MarkdownBlock::Paragraph { range, .. }
        | MarkdownBlock::Heading { range, .. }
        | MarkdownBlock::CodeBlock { range, .. }
        | MarkdownBlock::BlockQuote { range, .. }
        | MarkdownBlock::List { range, .. } => *range,
    }
}

fn inline_source_range(inline: &MarkdownInline) -> Option<MarkdownSourceRange> {
    match inline {
        MarkdownInline::Text { range, .. }
        | MarkdownInline::Emphasis { range, .. }
        | MarkdownInline::Strong { range, .. }
        | MarkdownInline::Code { range, .. }
        | MarkdownInline::Link { range, .. }
        | MarkdownInline::SoftBreak { range }
        | MarkdownInline::HardBreak { range } => *range,
    }
}

fn push_block(stack: &mut [BlockContext], block: MarkdownBlock) {
    let child_range = block_source_range(&block);
    if let Some(parent) = stack.last_mut() {
        match parent {
            BlockContext::Root { blocks, .. } => blocks.push(block),
            BlockContext::BlockQuote { range, blocks, .. } => {
                blocks.push(block);
                if let Some(child_range) = child_range {
                    range.observe(child_range);
                }
            }
            BlockContext::ListItem { range, blocks, .. } => {
                blocks.push(block);
                if let Some(child_range) = child_range {
                    range.observe(child_range);
                }
            }
            _ => {}
        }
    }
}

fn push_item(stack: &mut [BlockContext], item: MarkdownListItem) {
    let item_range = item.range;
    if let Some(parent) = stack.last_mut() {
        match parent {
            BlockContext::List { range, items, .. } => {
                items.push(item);
                if let Some(item_range) = item_range {
                    range.observe(item_range);
                }
            }
            BlockContext::ListItem { range, blocks, .. } => {
                blocks.extend(item.blocks);
                if let Some(item_range) = item_range {
                    range.observe(item_range);
                }
            }
            BlockContext::BlockQuote { range, blocks, .. } => {
                blocks.extend(item.blocks);
                if let Some(item_range) = item_range {
                    range.observe(item_range);
                }
            }
            BlockContext::Root { blocks, .. } => blocks.extend(item.blocks),
            _ => {}
        }
    }
}

enum BlockContext {
    Root {
        blocks: Vec<MarkdownBlock>,
        tight_inlines: Option<InlineBuilder>,
    },
    Paragraph {
        inlines: InlineBuilder,
    },
    Heading {
        level: u8,
        inlines: InlineBuilder,
    },
    CodeBlock {
        range: RangeAccumulator,
        language: Option<String>,
        info_raw: Option<String>,
        code: String,
    },
    BlockQuote {
        range: RangeAccumulator,
        blocks: Vec<MarkdownBlock>,
        tight_inlines: Option<InlineBuilder>,
    },
    List {
        range: RangeAccumulator,
        ordered: bool,
        start: Option<u64>,
        items: Vec<MarkdownListItem>,
    },
    ListItem {
        range: RangeAccumulator,
        blocks: Vec<MarkdownBlock>,
        tight_inlines: Option<InlineBuilder>,
    },
}

impl BlockContext {
    fn inline_builder_mut(&mut self) -> Option<&mut InlineBuilder> {
        match self {
            BlockContext::Paragraph { inlines } | BlockContext::Heading { inlines, .. } => {
                Some(inlines)
            }
            _ => None,
        }
    }

    fn ensure_inline_builder_mut(&mut self) -> Option<&mut InlineBuilder> {
        match self {
            BlockContext::Paragraph { .. } | BlockContext::Heading { .. } => self.inline_builder_mut(),
            BlockContext::Root { tight_inlines, .. }
            | BlockContext::BlockQuote { tight_inlines, .. }
            | BlockContext::ListItem { tight_inlines, .. } => {
                if tight_inlines.is_none() {
                    *tight_inlines = Some(InlineBuilder::new());
                }
                tight_inlines.as_mut()
            }
            _ => None,
        }
    }

    fn push_inline(&mut self, inline: MarkdownInline) {
        if let Some(builder) = self.ensure_inline_builder_mut() {
            builder.push(inline);
        }
    }

    fn flush_tight_paragraph(&mut self) {
        match self {
            BlockContext::Root {
                blocks,
                tight_inlines,
            }
            => {
                flush_tight_inlines(blocks, tight_inlines);
            }
            BlockContext::BlockQuote {
                range,
                blocks,
                tight_inlines,
            }
            | BlockContext::ListItem {
                range,
                blocks,
                tight_inlines,
            } => {
                if let Some(span) = flush_tight_inlines(blocks, tight_inlines) {
                    range.observe(span);
                }
            }
            _ => {}
        }
    }
}

#[derive(Default)]
struct InlineBuilder {
    stack: Vec<InlineFrame>,
}

impl InlineBuilder {
    fn new() -> Self {
        Self {
            stack: vec![InlineFrame::new(InlineFrameKind::Root)],
        }
    }

    fn observe_offset_range(&mut self, range: &Range<usize>) {
        if let Some(frame) = self.stack.last_mut() {
            frame.range.observe_offset_range(range);
        }
    }

    fn push(&mut self, inline: MarkdownInline) {
        if let Some(frame) = self.stack.last_mut() {
            if let Some(span) = inline_source_range(&inline) {
                frame.range.observe(span);
            }
            frame.content.push(inline);
        }
    }

    fn start_emphasis(&mut self, range: &Range<usize>) {
        let mut frame = InlineFrame::new(InlineFrameKind::Emphasis);
        frame.range.observe_offset_range(range);
        self.stack.push(frame);
    }

    fn end_emphasis(&mut self, range: &Range<usize>) {
        self.end_frame(range, |range, content| MarkdownInline::Emphasis { range, content });
    }

    fn start_strong(&mut self, range: &Range<usize>) {
        let mut frame = InlineFrame::new(InlineFrameKind::Strong);
        frame.range.observe_offset_range(range);
        self.stack.push(frame);
    }

    fn end_strong(&mut self, range: &Range<usize>) {
        self.end_frame(range, |range, content| MarkdownInline::Strong { range, content });
    }

    fn start_link(&mut self, destination: String, range: &Range<usize>) {
        let mut frame = InlineFrame::new(InlineFrameKind::Link { destination });
        frame.range.observe_offset_range(range);
        self.stack.push(frame);
    }

    fn end_link(&mut self, range: &Range<usize>) {
        let mut frame = match self.stack.pop() {
            Some(frame) => frame,
            None => return,
        };

        let InlineFrameKind::Link { destination } = frame.kind else {
            self.stack.push(frame);
            return;
        };

        frame.range.observe_offset_range(range);

        self.push(MarkdownInline::Link {
            range: frame.range.finish(),
            destination,
            content: frame.content,
        });
    }

    fn end_frame(
        &mut self,
        range: &Range<usize>,
        wrap: impl FnOnce(Option<MarkdownSourceRange>, Vec<MarkdownInline>) -> MarkdownInline,
    ) {
        let mut frame = match self.stack.pop() {
            Some(frame) => frame,
            None => return,
        };

        match frame.kind {
            InlineFrameKind::Root | InlineFrameKind::Link { .. } => {
                self.stack.push(frame);
            }
            InlineFrameKind::Emphasis | InlineFrameKind::Strong => {
                frame.range.observe_offset_range(range);
                self.push(wrap(frame.range.finish(), frame.content));
            }
        }
    }

    fn finish_with_range(&mut self) -> (Vec<MarkdownInline>, Option<MarkdownSourceRange>) {
        while self.stack.len() > 1 {
            match self.stack.pop() {
                Some(InlineFrame {
                    kind: InlineFrameKind::Link { destination },
                    content,
                    range,
                }) => self.push(MarkdownInline::Link {
                    range: range.finish(),
                    destination,
                    content,
                }),
                Some(InlineFrame {
                    kind: InlineFrameKind::Emphasis,
                    content,
                    range,
                }) => self.push(MarkdownInline::Emphasis {
                    range: range.finish(),
                    content,
                }),
                Some(InlineFrame {
                    kind: InlineFrameKind::Strong,
                    content,
                    range,
                }) => self.push(MarkdownInline::Strong {
                    range: range.finish(),
                    content,
                }),
                _ => {}
            }
        }

        match self.stack.pop() {
            Some(frame) => (frame.content, frame.range.finish()),
            None => (Vec::new(), None),
        }
    }
}

struct InlineFrame {
    kind: InlineFrameKind,
    content: Vec<MarkdownInline>,
    range: RangeAccumulator,
}

impl InlineFrame {
    fn new(kind: InlineFrameKind) -> Self {
        Self {
            kind,
            content: Vec::new(),
            range: RangeAccumulator::default(),
        }
    }
}

enum InlineFrameKind {
    Root,
    Emphasis,
    Strong,
    Link { destination: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_soft_and_hard_breaks() {
        let doc = parse_markdown("Hello\nWorld  \nAgain", MarkdownParseOptions::default());

        let [MarkdownBlock::Paragraph { content, .. }] = doc.blocks.as_slice() else {
            panic!("expected single paragraph");
        };

        assert_eq!(content.len(), 5);
        assert!(matches!(&content[0], MarkdownInline::Text { text, .. } if text == "Hello"));
        assert!(matches!(&content[1], MarkdownInline::SoftBreak { .. }));
        assert!(matches!(&content[2], MarkdownInline::Text { text, .. } if text == "World"));
        assert!(matches!(&content[3], MarkdownInline::HardBreak { .. }));
        assert!(matches!(&content[4], MarkdownInline::Text { text, .. } if text == "Again"));
    }

    #[test]
    fn parses_headings_and_inline_spans() {
        let doc = parse_markdown(
            "# Title\n\nHello *world* and **bold** and `code`.",
            MarkdownParseOptions::default(),
        );

        let MarkdownBlock::Heading {
            level, content, ..
        } = &doc.blocks[0]
        else {
            panic!("expected heading");
        };
        assert_eq!(*level, 1);
        assert!(matches!(
            content.as_slice(),
            [MarkdownInline::Text { text, .. }] if text == "Title"
        ));

        let MarkdownBlock::Paragraph { content: inlines, .. } = &doc.blocks[1] else {
            panic!("expected paragraph");
        };

        assert!(inlines.iter().any(|inline| matches!(
            inline,
            MarkdownInline::Emphasis { .. }
        )));
        assert!(inlines.iter().any(|inline| matches!(inline, MarkdownInline::Strong { .. })));
        assert!(inlines.iter().any(|inline| matches!(inline, MarkdownInline::Code { .. })));
    }

    #[test]
    fn parses_fenced_code_blocks_with_language() {
        let doc = parse_markdown(
            "```rust\nfn main() {}\n```",
            MarkdownParseOptions::default(),
        );

        let [MarkdownBlock::CodeBlock {
            language,
            info_raw,
            code,
            ..
        }] = doc.blocks.as_slice()
        else {
            panic!("expected single code block");
        };
        assert_eq!(language.as_deref(), Some("rust"));
        assert_eq!(info_raw.as_deref(), Some("rust"));
        assert_eq!(code, "fn main() {}\n");
    }

    #[test]
    fn normalizes_fenced_language_tokens() {
        let doc = parse_markdown(
            "```Rust,no_run\nfn main() {}\n```",
            MarkdownParseOptions::default(),
        );

        let [MarkdownBlock::CodeBlock {
            language,
            info_raw,
            code,
            ..
        }] = doc.blocks.as_slice()
        else {
            panic!("expected single code block");
        };

        assert_eq!(language.as_deref(), Some("rust"));
        assert_eq!(info_raw.as_deref(), Some("Rust,no_run"));
        assert_eq!(code, "fn main() {}\n");
    }

    #[test]
    fn extracts_fenced_language_from_comma_followed_by_space() {
        let doc = parse_markdown(
            "```rust, no_run\nfn main() {}\n```",
            MarkdownParseOptions::default(),
        );

        let [MarkdownBlock::CodeBlock {
            language, info_raw, ..
        }] = doc.blocks.as_slice()
        else {
            panic!("expected single code block");
        };

        assert_eq!(language.as_deref(), Some("rust"));
        assert_eq!(info_raw.as_deref(), Some("rust, no_run"));
    }

    #[test]
    fn parses_block_quotes_and_lists() {
        let doc = parse_markdown(
            "> quoted\n> - item\n>   - nested\n",
            MarkdownParseOptions::default(),
        );

        let [MarkdownBlock::BlockQuote { content: blocks, .. }] = doc.blocks.as_slice() else {
            panic!("expected single block quote");
        };

        assert!(blocks.iter().any(|block| matches!(block, MarkdownBlock::Paragraph { .. })));
        assert!(blocks.iter().any(|block| matches!(block, MarkdownBlock::List { .. })));
    }

    #[test]
    fn truncates_large_inputs() {
        let input = "a".repeat(50);
        let doc = parse_markdown(
            &input,
            MarkdownParseOptions {
                max_input_len: 10,
            },
        );

        assert_eq!(
            doc.truncation,
            Some(MarkdownTruncation {
                original_len: 50,
                rendered_len: 10,
            })
        );
    }

    #[test]
    fn records_source_ranges_for_simple_text() {
        let doc = parse_markdown("Hello", MarkdownParseOptions::default());

        let [MarkdownBlock::Paragraph {
            range: Some(range),
            content,
        }] = doc.blocks.as_slice()
        else {
            panic!("expected a ranged paragraph");
        };

        assert!(range.end > range.start);

        let [MarkdownInline::Text {
            range: Some(text_range),
            text,
        }] = content.as_slice()
        else {
            panic!("expected a ranged text inline");
        };

        assert_eq!(text, "Hello");
        assert!(text_range.end > text_range.start);
        assert!(text_range.start >= range.start);
        assert!(text_range.end <= range.end);
    }

    #[test]
    fn projects_nested_lists_code_blocks_and_links() {
        let doc = parse_markdown(
            r#"1. First
   - nested A
     - nested B
2. Second with [link](https://example.com) and `inline`.

```rust,no_run
fn main() {}
```

```bash
echo "hi"
```"#,
            MarkdownParseOptions::default(),
        );

        let list = doc
            .blocks
            .iter()
            .find_map(|block| match block {
                MarkdownBlock::List { ordered, items, .. } if *ordered => Some(items),
                _ => None,
            })
            .expect("expected ordered list block");
        assert_eq!(list.len(), 2);

        let nested_list = list[0]
            .blocks
            .iter()
            .find_map(|block| match block {
                MarkdownBlock::List { ordered, items, .. } if !*ordered => Some(items),
                _ => None,
            })
            .expect("expected nested bullet list");
        assert!(
            nested_list[0]
                .blocks
                .iter()
                .any(|block| matches!(block, MarkdownBlock::Paragraph { .. })),
            "expected paragraph inside nested list item"
        );
        assert!(
            nested_list[0]
                .blocks
                .iter()
                .any(|block| matches!(block, MarkdownBlock::List { .. })),
            "expected nested list inside nested list item"
        );

        let second_paragraph_inlines = list[1]
            .blocks
            .iter()
            .find_map(|block| match block {
                MarkdownBlock::Paragraph { content, .. } => Some(content),
                _ => None,
            })
            .expect("expected paragraph in second list item");
        assert!(
            second_paragraph_inlines.iter().any(|inline| matches!(
                inline,
                MarkdownInline::Link { destination, .. } if destination == "https://example.com"
            )),
            "expected https link inline"
        );

        let code_blocks: Vec<_> = doc
            .blocks
            .iter()
            .filter_map(|block| match block {
                MarkdownBlock::CodeBlock {
                    language,
                    info_raw,
                    code,
                    ..
                } => Some((language.as_deref(), info_raw.as_deref(), code.as_str())),
                _ => None,
            })
            .collect();

        assert!(
            code_blocks.iter().any(|(language, info_raw, code)| {
                *language == Some("rust")
                    && *info_raw == Some("rust,no_run")
                    && *code == "fn main() {}\n"
            }),
            "expected rust fenced code block"
        );
        assert!(
            code_blocks.iter().any(|(language, info_raw, code)| {
                *language == Some("bash") && *info_raw == Some("bash") && *code == "echo \"hi\"\n"
            }),
            "expected bash fenced code block"
        );
    }
}
