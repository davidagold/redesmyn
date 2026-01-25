//! Minimal markdown parsing for GPUI surfaces.
//!
//! This crate is intentionally GPUI-free so it can be unit tested and reused by UI crates without
//! pulling in rendering dependencies.

#![forbid(unsafe_code)]

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MarkdownBlock {
    Paragraph(Vec<MarkdownInline>),
    Heading {
        level: u8,
        content: Vec<MarkdownInline>,
    },
    CodeBlock {
        language: Option<String>,
        code: String,
    },
    BlockQuote(Vec<MarkdownBlock>),
    List {
        ordered: bool,
        start: Option<u64>,
        items: Vec<MarkdownListItem>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarkdownListItem {
    pub blocks: Vec<MarkdownBlock>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MarkdownInline {
    Text(String),
    Emphasis(Vec<MarkdownInline>),
    Strong(Vec<MarkdownInline>),
    Code(String),
    Link {
        destination: String,
        content: Vec<MarkdownInline>,
    },
    SoftBreak,
    HardBreak,
}

pub fn parse_markdown(input: &str, options: MarkdownParseOptions) -> MarkdownDoc {
    let (prefix, truncation) = truncate(input, options.max_input_len);
    let parser = Parser::new(prefix);
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

fn parse_blocks<'a>(events: impl IntoIterator<Item = Event<'a>>) -> Vec<MarkdownBlock> {
    let mut stack = vec![BlockContext::Root { blocks: Vec::new() }];

    for event in events {
        match event {
            Event::Start(tag) => match tag {
                Tag::Paragraph => stack.push(BlockContext::Paragraph {
                    inlines: InlineBuilder::new(),
                }),
                Tag::Heading { level, .. } => stack.push(BlockContext::Heading {
                    level: heading_level(level),
                    inlines: InlineBuilder::new(),
                }),
                Tag::BlockQuote(_) => stack.push(BlockContext::BlockQuote { blocks: Vec::new() }),
                Tag::List(start) => stack.push(BlockContext::List {
                    ordered: start.is_some(),
                    start,
                    items: Vec::new(),
                }),
                Tag::Item => stack.push(BlockContext::ListItem { blocks: Vec::new() }),
                Tag::CodeBlock(kind) => stack.push(BlockContext::CodeBlock {
                    language: code_block_language(kind),
                    code: String::new(),
                }),
                Tag::Emphasis => {
                    if let Some(builder) = stack
                        .last_mut()
                        .and_then(BlockContext::inline_builder_mut)
                    {
                        builder.start_emphasis();
                    }
                }
                Tag::Strong => {
                    if let Some(builder) = stack
                        .last_mut()
                        .and_then(BlockContext::inline_builder_mut)
                    {
                        builder.start_strong();
                    }
                }
                Tag::Link { dest_url, .. } => {
                    if let Some(builder) = stack
                        .last_mut()
                        .and_then(BlockContext::inline_builder_mut)
                    {
                        builder.start_link(dest_url.to_string());
                    }
                }
                _ => {}
            },
            Event::End(tag_end) => match tag_end {
                TagEnd::Paragraph => {
                    if let Some(BlockContext::Paragraph { mut inlines }) = stack.pop() {
                        push_block(&mut stack, MarkdownBlock::Paragraph(inlines.finish()));
                    }
                }
                TagEnd::Heading(_) => {
                    if let Some(BlockContext::Heading { level, mut inlines }) = stack.pop() {
                        push_block(
                            &mut stack,
                            MarkdownBlock::Heading {
                                level,
                                content: inlines.finish(),
                            },
                        );
                    }
                }
                TagEnd::BlockQuote(_) => {
                    if let Some(BlockContext::BlockQuote { blocks }) = stack.pop() {
                        push_block(&mut stack, MarkdownBlock::BlockQuote(blocks));
                    }
                }
                TagEnd::List(_) => {
                    if let Some(BlockContext::List {
                        ordered,
                        start,
                        items,
                    }) = stack.pop()
                    {
                        push_block(
                            &mut stack,
                            MarkdownBlock::List {
                                ordered,
                                start,
                                items,
                            },
                        );
                    }
                }
                TagEnd::Item => {
                    if let Some(BlockContext::ListItem { blocks }) = stack.pop() {
                        push_item(&mut stack, MarkdownListItem { blocks });
                    }
                }
                TagEnd::CodeBlock => {
                    if let Some(BlockContext::CodeBlock { language, code }) = stack.pop() {
                        push_block(
                            &mut stack,
                            MarkdownBlock::CodeBlock { language, code },
                        );
                    }
                }
                TagEnd::Emphasis => {
                    if let Some(builder) = stack
                        .last_mut()
                        .and_then(BlockContext::inline_builder_mut)
                    {
                        builder.end_emphasis();
                    }
                }
                TagEnd::Strong => {
                    if let Some(builder) = stack
                        .last_mut()
                        .and_then(BlockContext::inline_builder_mut)
                    {
                        builder.end_strong();
                    }
                }
                TagEnd::Link => {
                    if let Some(builder) = stack
                        .last_mut()
                        .and_then(BlockContext::inline_builder_mut)
                    {
                        builder.end_link();
                    }
                }
                _ => {}
            },
            Event::Text(text) => match stack.last_mut() {
                Some(BlockContext::CodeBlock { code, .. }) => code.push_str(text.as_ref()),
                Some(ctx) => ctx.push_inline(MarkdownInline::Text(text.to_string())),
                None => {}
            },
            Event::Code(code) => {
                if let Some(ctx) = stack.last_mut() {
                    ctx.push_inline(MarkdownInline::Code(code.to_string()));
                }
            }
            Event::SoftBreak => {
                if let Some(ctx) = stack.last_mut() {
                    ctx.push_inline(MarkdownInline::SoftBreak);
                }
            }
            Event::HardBreak => {
                if let Some(ctx) = stack.last_mut() {
                    ctx.push_inline(MarkdownInline::HardBreak);
                }
            }
            Event::Html(html) | Event::InlineHtml(html) => {
                if let Some(ctx) = stack.last_mut() {
                    ctx.push_inline(MarkdownInline::Text(html.to_string()));
                }
            }
            Event::InlineMath(math) | Event::DisplayMath(math) => {
                if let Some(ctx) = stack.last_mut() {
                    ctx.push_inline(MarkdownInline::Text(math.to_string()));
                }
            }
            Event::Rule | Event::FootnoteReference(_) | Event::TaskListMarker(_) => {}
        }
    }

    while stack.len() > 1 {
        match stack.pop() {
            Some(BlockContext::Paragraph { mut inlines }) => {
                push_block(&mut stack, MarkdownBlock::Paragraph(inlines.finish()));
            }
            Some(BlockContext::Heading { level, mut inlines }) => {
                push_block(
                    &mut stack,
                    MarkdownBlock::Heading {
                        level,
                        content: inlines.finish(),
                    },
                );
            }
            Some(BlockContext::CodeBlock { language, code }) => {
                push_block(&mut stack, MarkdownBlock::CodeBlock { language, code });
            }
            Some(BlockContext::BlockQuote { blocks }) => {
                push_block(&mut stack, MarkdownBlock::BlockQuote(blocks));
            }
            Some(BlockContext::List {
                ordered,
                start,
                items,
            }) => {
                push_block(
                    &mut stack,
                    MarkdownBlock::List {
                        ordered,
                        start,
                        items,
                    },
                );
            }
            Some(BlockContext::ListItem { blocks }) => {
                push_item(&mut stack, MarkdownListItem { blocks });
            }
            _ => {}
        }
    }

    match stack.pop() {
        Some(BlockContext::Root { blocks }) => blocks,
        _ => Vec::new(),
    }
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

fn code_block_language(kind: CodeBlockKind<'_>) -> Option<String> {
    match kind {
        CodeBlockKind::Indented => None,
        CodeBlockKind::Fenced(lang) => {
            let language = lang.trim();
            if language.is_empty() {
                None
            } else {
                Some(language.to_string())
            }
        }
    }
}

fn push_block(stack: &mut [BlockContext], block: MarkdownBlock) {
    if let Some(parent) = stack.last_mut() {
        match parent {
            BlockContext::Root { blocks } => blocks.push(block),
            BlockContext::BlockQuote { blocks } => blocks.push(block),
            BlockContext::ListItem { blocks } => blocks.push(block),
            _ => {}
        }
    }
}

fn push_item(stack: &mut [BlockContext], item: MarkdownListItem) {
    if let Some(parent) = stack.last_mut() {
        match parent {
            BlockContext::List { items, .. } => items.push(item),
            BlockContext::ListItem { blocks } => blocks.extend(item.blocks),
            BlockContext::BlockQuote { blocks } => blocks.extend(item.blocks),
            BlockContext::Root { blocks } => blocks.extend(item.blocks),
            _ => {}
        }
    }
}

enum BlockContext {
    Root {
        blocks: Vec<MarkdownBlock>,
    },
    Paragraph {
        inlines: InlineBuilder,
    },
    Heading {
        level: u8,
        inlines: InlineBuilder,
    },
    CodeBlock {
        language: Option<String>,
        code: String,
    },
    BlockQuote {
        blocks: Vec<MarkdownBlock>,
    },
    List {
        ordered: bool,
        start: Option<u64>,
        items: Vec<MarkdownListItem>,
    },
    ListItem {
        blocks: Vec<MarkdownBlock>,
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

    fn push_inline(&mut self, inline: MarkdownInline) {
        if let Some(builder) = self.inline_builder_mut() {
            builder.push(inline);
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

    fn push(&mut self, inline: MarkdownInline) {
        if let Some(frame) = self.stack.last_mut() {
            frame.content.push(inline);
        }
    }

    fn start_emphasis(&mut self) {
        self.stack.push(InlineFrame::new(InlineFrameKind::Emphasis));
    }

    fn end_emphasis(&mut self) {
        self.end_frame(|content| MarkdownInline::Emphasis(content));
    }

    fn start_strong(&mut self) {
        self.stack.push(InlineFrame::new(InlineFrameKind::Strong));
    }

    fn end_strong(&mut self) {
        self.end_frame(|content| MarkdownInline::Strong(content));
    }

    fn start_link(&mut self, destination: String) {
        self.stack
            .push(InlineFrame::new(InlineFrameKind::Link { destination }));
    }

    fn end_link(&mut self) {
        let frame = match self.stack.pop() {
            Some(frame) => frame,
            None => return,
        };

        let InlineFrameKind::Link { destination } = frame.kind else {
            self.stack.push(frame);
            return;
        };

        self.push(MarkdownInline::Link {
            destination,
            content: frame.content,
        });
    }

    fn end_frame(&mut self, wrap: impl FnOnce(Vec<MarkdownInline>) -> MarkdownInline) {
        let frame = match self.stack.pop() {
            Some(frame) => frame,
            None => return,
        };

        match frame.kind {
            InlineFrameKind::Root | InlineFrameKind::Link { .. } => {
                self.stack.push(frame);
            }
            InlineFrameKind::Emphasis | InlineFrameKind::Strong => self.push(wrap(frame.content)),
        }
    }

    fn finish(&mut self) -> Vec<MarkdownInline> {
        while self.stack.len() > 1 {
            match self.stack.pop() {
                Some(InlineFrame {
                    kind: InlineFrameKind::Link { destination },
                    content,
                }) => self.push(MarkdownInline::Link {
                    destination,
                    content,
                }),
                Some(InlineFrame {
                    kind: InlineFrameKind::Emphasis,
                    content,
                }) => self.push(MarkdownInline::Emphasis(content)),
                Some(InlineFrame {
                    kind: InlineFrameKind::Strong,
                    content,
                }) => self.push(MarkdownInline::Strong(content)),
                _ => {}
            }
        }

        self.stack
            .pop()
            .map(|frame| frame.content)
            .unwrap_or_default()
    }
}

struct InlineFrame {
    kind: InlineFrameKind,
    content: Vec<MarkdownInline>,
}

impl InlineFrame {
    fn new(kind: InlineFrameKind) -> Self {
        Self {
            kind,
            content: Vec::new(),
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

        assert_eq!(
            doc.blocks,
            vec![MarkdownBlock::Paragraph(vec![
                MarkdownInline::Text("Hello".into()),
                MarkdownInline::SoftBreak,
                MarkdownInline::Text("World".into()),
                MarkdownInline::HardBreak,
                MarkdownInline::Text("Again".into()),
            ])]
        );
    }

    #[test]
    fn parses_headings_and_inline_spans() {
        let doc = parse_markdown(
            "# Title\n\nHello *world* and **bold** and `code`.",
            MarkdownParseOptions::default(),
        );

        assert_eq!(
            doc.blocks[0],
            MarkdownBlock::Heading {
                level: 1,
                content: vec![MarkdownInline::Text("Title".into())],
            }
        );

        let MarkdownBlock::Paragraph(inlines) = &doc.blocks[1] else {
            panic!("expected paragraph");
        };

        assert!(inlines.iter().any(|inline| matches!(
            inline,
            MarkdownInline::Emphasis(_)
        )));
        assert!(inlines.iter().any(|inline| matches!(inline, MarkdownInline::Strong(_))));
        assert!(inlines.iter().any(|inline| matches!(inline, MarkdownInline::Code(_))));
    }

    #[test]
    fn parses_fenced_code_blocks_with_language() {
        let doc = parse_markdown(
            "```rust\nfn main() {}\n```",
            MarkdownParseOptions::default(),
        );

        assert_eq!(
            doc.blocks,
            vec![MarkdownBlock::CodeBlock {
                language: Some("rust".into()),
                code: "fn main() {}\n".into(),
            }]
        );
    }

    #[test]
    fn parses_block_quotes_and_lists() {
        let doc = parse_markdown(
            "> quoted\n> - item\n>   - nested\n",
            MarkdownParseOptions::default(),
        );

        let [MarkdownBlock::BlockQuote(blocks)] = doc.blocks.as_slice() else {
            panic!("expected single block quote");
        };

        assert!(blocks.iter().any(|block| matches!(block, MarkdownBlock::Paragraph(_))));
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
}
