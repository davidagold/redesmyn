use std::sync::Arc;

use gpui::{
    AnyElement, App, ClipboardItem, ClickEvent, ElementId, FontStyle, FontWeight, RenderOnce,
    StyledText, TextRun, UnderlineStyle, Window, div, px,
};

use gpui::prelude::*;

use redesmyn_markdown::{MarkdownBlock, MarkdownDoc, MarkdownInline};

use crate::utils::{OpenExternalUrl as _, theme_for_window};

use super::{ButtonKind, TextButton};

#[derive(IntoElement)]
pub struct MarkdownView {
    id: ElementId,
    doc: Arc<MarkdownDoc>,
    show_truncation_notice: bool,
}

impl MarkdownView {
    pub fn new(id: impl Into<ElementId>, doc: Arc<MarkdownDoc>) -> Self {
        Self {
            id: id.into(),
            doc,
            show_truncation_notice: true,
        }
    }

    pub fn show_truncation_notice(mut self, show: bool) -> Self {
        self.show_truncation_notice = show;
        self
    }
}

impl RenderOnce for MarkdownView {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let theme = theme_for_window(window, cx);

        let base_id = self.id.clone();

        let mut container = div()
            .id(base_id.clone())
            .flex()
            .flex_col()
            .gap(theme.spacing.sm)
            .w_full()
            .text_color(theme.colors.foreground)
            .text_size(theme.typography.body.size);

        for (ix, block) in self.doc.blocks.iter().enumerate() {
            let block_id: ElementId = (base_id.clone(), format!("block-{ix}")).into();
            container = container.child(render_block(block_id, block, window, cx));
        }

        if self.show_truncation_notice
            && let Some(truncation) = self.doc.truncation.as_ref()
        {
            container = container.child(
                div()
                    .id((base_id, "truncation"))
                    .text_xs()
                    .text_color(theme.colors.foreground_muted)
                    .child(format!(
                        "Truncated for rendering ({} / {} bytes).",
                        truncation.rendered_len, truncation.original_len
                    )),
            );
        }

        container
    }
}

fn render_block(
    id: ElementId,
    block: &MarkdownBlock,
    window: &mut Window,
    cx: &mut App,
) -> AnyElement {
    let theme = theme_for_window(window, cx);

    match block {
        MarkdownBlock::Paragraph { content, .. } => render_inline_flow(id, content, window, cx),
        MarkdownBlock::Heading { level, content, .. } => {
            let text_size = match level {
                1 => div().text_2xl(),
                2 => div().text_xl(),
                3 => div().text_lg(),
                _ => div().text_base(),
            };

            let inline_id: ElementId = (id.clone(), "inline").into();
            text_size
                .id(id)
                .font_weight(FontWeight::SEMIBOLD)
                .child(render_inline_segments(
                    inline_id,
                    content,
                    InlineSegmentsStyle::default().as_heading(*level),
                    window,
                    cx,
                ))
                .into_any_element()
        }
        MarkdownBlock::CodeBlock {
            language,
            code,
            ..
        } => CodeBlockView::new(id, language, code)
            .render(window, cx)
            .into_any_element(),
        MarkdownBlock::BlockQuote { content: blocks, .. } => {
            let mut quote = div()
                .id(id.clone())
                .flex()
                .flex_row()
                .gap(theme.spacing.sm)
                .px(theme.spacing.md)
                .py(theme.spacing.sm)
                .rounded_md()
                .bg(theme.colors.surface_elevated.opacity(0.25));

            quote = quote.child(
                div()
                    .w(px(3.0))
                    .rounded_sm()
                    .bg(theme.colors.border.opacity(0.7)),
            );

            let content_id: ElementId = (id, "content").into();
            quote = quote.child(render_blocks_column(content_id, blocks, window, cx));

            quote.into_any_element()
        }
        MarkdownBlock::List {
            ordered,
            start,
            items,
            ..
        } => {
            let mut list = div().id(id.clone()).flex().flex_col().gap(theme.spacing.xs);

            let start_ix = start.unwrap_or(1);
            for (ix, item) in items.iter().enumerate() {
                let marker = if *ordered {
                    format!("{}.", start_ix.saturating_add(ix as u64))
                } else {
                    "•".to_string()
                };

                let item_id: ElementId = (id.clone(), format!("item-{ix}")).into();
                list = list.child(render_list_item(item_id, marker, item, window, cx));
            }

            list.into_any_element()
        }
    }
}

fn render_blocks_column(
    id: ElementId,
    blocks: &[MarkdownBlock],
    window: &mut Window,
    cx: &mut App,
) -> AnyElement {
    let theme = theme_for_window(window, cx);
    let mut col = div()
        .id(id.clone())
        .flex()
        .flex_col()
        .gap(theme.spacing.sm)
        .flex_1()
        .min_w_0();

    for (ix, block) in blocks.iter().enumerate() {
        let block_id: ElementId = (id.clone(), format!("block-{ix}")).into();
        col = col.child(render_block(block_id, block, window, cx));
    }

    col.into_any_element()
}

fn render_list_item(
    id: ElementId,
    marker: String,
    item: &redesmyn_markdown::MarkdownListItem,
    window: &mut Window,
    cx: &mut App,
) -> AnyElement {
    let theme = theme_for_window(window, cx);

    let content_id: ElementId = (id.clone(), "content").into();
    div()
        .id(id)
        .flex()
        .flex_row()
        .gap(theme.spacing.sm)
        .items_start()
        .child(
            div()
                .w(px(28.0))
                .text_color(theme.colors.foreground_muted)
                .text_right()
                .child(marker),
        )
        .child(render_blocks_column(content_id, &item.blocks, window, cx))
        .into_any_element()
}

fn render_inline_flow(
    id: ElementId,
    inlines: &[MarkdownInline],
    window: &mut Window,
    cx: &mut App,
) -> AnyElement {
    render_inline_segments(id, inlines, InlineSegmentsStyle::default(), window, cx)
}

#[derive(Debug, Clone, Copy)]
struct InlineSegmentsStyle {
    link_color_is_ring: bool,
}

impl Default for InlineSegmentsStyle {
    fn default() -> Self {
        Self {
            link_color_is_ring: true,
        }
    }
}

impl InlineSegmentsStyle {
    fn as_heading(self, _level: u8) -> Self {
        // Headings typically inherit weight/size from the parent container; keep link styling
        // consistent.
        self
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct InlineStyle {
    bold: bool,
    italic: bool,
    code: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct InlineAtom {
    text: String,
    style: InlineStyle,
    link: Option<String>,
}

fn render_inline_segments(
    id: ElementId,
    inlines: &[MarkdownInline],
    style: InlineSegmentsStyle,
    window: &mut Window,
    cx: &mut App,
) -> AnyElement {
    let items = flatten_inlines(inlines);
    let lines = split_inline_items_into_lines(items);

    let mut flow = div()
        .id(id.clone())
        .flex()
        .flex_col()
        .min_w_0()
        .w_full();

    for (line_ix, atoms) in lines.into_iter().enumerate() {
        let line_id: ElementId = (id.clone(), format!("line-{line_ix}")).into();

        if atoms.is_empty() {
            flow = flow.child(styled_text_div(
                line_id,
                vec![InlineAtom {
                    text: " ".to_string(),
                    style: InlineStyle::default(),
                    link: None,
                }],
                TextFlavor::Body,
                window,
                cx,
            ));
            continue;
        }

        let has_links = atoms.iter().any(|atom| atom.link.is_some());
        if !has_links {
            // Render entire non-link lines as a single `StyledText` to avoid GPUI flex-wrap
            // layout/paint artifacts that can show up when we emit many inline chunks.
            flow = flow.child(styled_text_block(line_id, atoms, TextFlavor::Body, window, cx));
            continue;
        }

        let chunks = chunk_atoms(atoms);
        let mut line = div()
            .id(line_id.clone())
            .flex()
            .flex_row()
            .flex_wrap()
            .min_w_0()
            .w_full();

        for (chunk_ix, chunk) in chunks.into_iter().enumerate() {
            let chunk_id: ElementId = (line_id.clone(), format!("chunk-{chunk_ix}")).into();
            line = line.child(render_inline_chunk(chunk_id, chunk, style, window, cx));
        }

        flow = flow.child(line);
    }

    flow.into_any_element()
}

fn split_inline_items_into_lines(items: Vec<InlineItem>) -> Vec<Vec<InlineAtom>> {
    let mut lines = Vec::new();
    let mut current = Vec::new();

    for item in items {
        match item {
            InlineItem::Atom(atom) => current.push(atom),
            InlineItem::HardBreak => {
                lines.push(std::mem::take(&mut current));
            }
        }
    }

    lines.push(current);
    lines
}

enum InlineChunk {
    Text(Vec<InlineAtom>),
    Link { destination: String, atoms: Vec<InlineAtom> },
}

fn render_inline_chunk(
    id: ElementId,
    chunk: InlineChunk,
    style: InlineSegmentsStyle,
    window: &mut Window,
    cx: &mut App,
) -> AnyElement {
    match chunk {
        InlineChunk::Text(atoms) => {
            let element = styled_text_div(id, atoms, TextFlavor::Body, window, cx);
            element.into_any_element()
        }
        InlineChunk::Link { destination, atoms } => {
            let theme = theme_for_window(window, cx);
            let url = destination.clone();
            let link_color = if style.link_color_is_ring {
                theme.colors.ring
            } else {
                theme.colors.foreground
            };

            div()
                .id(id.clone())
                .cursor_pointer()
                .focusable()
                .hover(|this| this.text_color(link_color.opacity(0.9)))
                .focus(|mut style| {
                    style
                        .text
                        .get_or_insert_with(Default::default)
                        .color = Some(link_color);
                    style
                })
                .on_click(move |event: &ClickEvent, _window, cx| {
                    if event.standard_click() {
                        if !cx.open_external_url(&url) {
                            cx.write_to_clipboard(ClipboardItem::new_string(url.clone()));
                        }
                    }
                })
                .child(styled_text_div(
                    (id, "text").into(),
                    atoms,
                    TextFlavor::Link(link_color),
                    window,
                    cx,
                ))
                .into_any_element()
        }
    }
}

enum TextFlavor {
    Body,
    Link(gpui::Hsla),
}

fn styled_text_div(
    id: ElementId,
    atoms: Vec<InlineAtom>,
    flavor: TextFlavor,
    window: &mut Window,
    cx: &mut App,
) -> impl IntoElement {
    let theme = theme_for_window(window, cx);

    let mut text = String::new();
    let mut runs = Vec::new();

    let (default_color, underline) = match flavor {
        TextFlavor::Body => (theme.colors.foreground, None),
        TextFlavor::Link(color) => (
            color,
            Some(UnderlineStyle {
                color: Some(color),
                thickness: px(1.0),
                wavy: false,
            }),
        ),
    };

    for atom in atoms {
        if atom.text.is_empty() {
            continue;
        }

        let mut font = if atom.style.code {
            theme.typography.mono.font.clone()
        } else {
            theme.typography.body.font.clone()
        };

        if atom.style.bold {
            font.weight = FontWeight::BOLD;
        }
        if atom.style.italic {
            font.style = FontStyle::Italic;
        }

        let background_color = if atom.style.code {
            Some(theme.colors.accent.opacity(0.65))
        } else {
            None
        };

        runs.push(TextRun {
            len: atom.text.len(),
            font,
            color: default_color,
            background_color,
            underline,
            strikethrough: None,
        });
        text.push_str(&atom.text);
    }

    div()
        .id(id)
        .min_w_0()
        .flex_shrink()
        .child(StyledText::new(text).with_runs(runs))
}

fn styled_text_block(
    id: ElementId,
    atoms: Vec<InlineAtom>,
    flavor: TextFlavor,
    window: &mut Window,
    cx: &mut App,
) -> impl IntoElement {
    // Similar to `styled_text_div`, but forces the line to take full width (which helps keep GPUI's
    // layout stable when inline spans mix styles).
    let theme = theme_for_window(window, cx);

    let mut text = String::new();
    let mut runs = Vec::new();

    let (default_color, underline) = match flavor {
        TextFlavor::Body => (theme.colors.foreground, None),
        TextFlavor::Link(color) => (
            color,
            Some(UnderlineStyle {
                color: Some(color),
                thickness: px(1.0),
                wavy: false,
            }),
        ),
    };

    for atom in atoms {
        if atom.text.is_empty() {
            continue;
        }

        let mut font = if atom.style.code {
            theme.typography.mono.font.clone()
        } else {
            theme.typography.body.font.clone()
        };

        if atom.style.bold {
            font.weight = FontWeight::BOLD;
        }
        if atom.style.italic {
            font.style = FontStyle::Italic;
        }

        let background_color = if atom.style.code {
            Some(theme.colors.accent.opacity(0.65))
        } else {
            None
        };

        runs.push(TextRun {
            len: atom.text.len(),
            font,
            color: default_color,
            background_color,
            underline,
            strikethrough: None,
        });
        text.push_str(&atom.text);
    }

    div()
        .id(id)
        .min_w_0()
        .w_full()
        .child(StyledText::new(text).with_runs(runs))
}

enum InlineItem {
    Atom(InlineAtom),
    HardBreak,
}

fn flatten_inlines(inlines: &[MarkdownInline]) -> Vec<InlineItem> {
    let mut out = Vec::new();
    for inline in inlines {
        flatten_inline(inline, InlineStyle::default(), None, &mut out);
    }

    out
}

fn flatten_inline(
    inline: &MarkdownInline,
    style: InlineStyle,
    link: Option<&str>,
    out: &mut Vec<InlineItem>,
) {
    match inline {
        MarkdownInline::Text { text, .. } => push_atom(text, style, link, out),
        MarkdownInline::Emphasis { content: children, .. } => {
            let mut style = style;
            style.italic = true;
            for child in children {
                flatten_inline(child, style, link, out);
            }
        }
        MarkdownInline::Strong { content: children, .. } => {
            let mut style = style;
            style.bold = true;
            for child in children {
                flatten_inline(child, style, link, out);
            }
        }
        MarkdownInline::Code { code, .. } => {
            let mut style = style;
            style.code = true;
            push_atom(code, style, link, out);
        }
        MarkdownInline::Link {
            destination,
            content,
            ..
        } => {
            for child in content {
                flatten_inline(child, style, Some(destination.as_str()), out);
            }
        }
        MarkdownInline::SoftBreak { .. } => push_atom(" ", style, link, out),
        MarkdownInline::HardBreak { .. } => out.push(InlineItem::HardBreak),
    }
}

fn push_atom(text: &str, style: InlineStyle, link: Option<&str>, out: &mut Vec<InlineItem>) {
    if text.is_empty() {
        return;
    }

    let link = link.map(str::to_string);
    if let Some(last) = out.last_mut() {
        if let InlineItem::Atom(last) = last
            && last.style == style
            && last.link == link
        {
            last.text.push_str(text);
            return;
        }
    }

    out.push(InlineItem::Atom(InlineAtom {
        text: text.to_string(),
        style,
        link,
    }));
}

fn chunk_atoms(atoms: Vec<InlineAtom>) -> Vec<InlineChunk> {
    let mut chunks = Vec::new();
    let mut current: Option<InlineChunk> = None;

    for atom in atoms {
        match (&mut current, atom.link.as_deref()) {
            (Some(InlineChunk::Text(atoms)), None) => atoms.push(atom),
            (Some(InlineChunk::Link { destination, atoms }), Some(dest))
                if destination == dest =>
            {
                atoms.push(atom);
            }
            (Some(_), _) => {
                if let Some(chunk) = current.take() {
                    chunks.push(chunk);
                }
                current = Some(new_chunk(atom));
            }
            (None, _) => current = Some(new_chunk(atom)),
        }
    }

    if let Some(chunk) = current {
        chunks.push(chunk);
    }

    chunks
}

fn new_chunk(mut atom: InlineAtom) -> InlineChunk {
    match atom.link.take() {
        Some(destination) => InlineChunk::Link {
            destination,
            atoms: vec![atom],
        },
        None => InlineChunk::Text(vec![atom]),
    }
}

struct CodeBlockView {
    id: ElementId,
    language: Option<String>,
    code: String,
}

impl CodeBlockView {
    fn new(id: ElementId, language: &Option<String>, code: &str) -> Self {
        Self {
            id,
            language: language.clone(),
            code: code.to_string(),
        }
    }

    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let theme = theme_for_window(window, cx);
        let id = self.id.clone();

        let code = self.code.clone();
        let copy_button = TextButton::new((id.clone(), "copy"), "Copy code block")
            .kind(ButtonKind::Ghost)
            .on_click(move |event: &ClickEvent, _window, cx| {
                if event.standard_click() {
                    cx.write_to_clipboard(ClipboardItem::new_string(code.clone()));
                }
            });

        let header = div()
            .flex()
            .flex_row()
            .items_center()
            .justify_between()
            .w_full()
            .px(theme.spacing.md)
            .py(theme.spacing.sm)
            .border_b_1()
            .border_color(theme.colors.border.opacity(0.5))
            .child(
                div()
                    .text_sm()
                    .text_color(theme.colors.foreground_muted)
                    .child(self.language.unwrap_or_else(|| "code".into())),
            )
            .child(copy_button);

        let body = div()
            .id((id.clone(), "body"))
            .overflow_x_scroll()
            .scrollbar_width(px(10.0))
            .px(theme.spacing.md)
            .py(theme.spacing.md)
            .font(theme.typography.mono.font.clone())
            .text_size(theme.typography.mono.size)
            .text_color(theme.colors.foreground)
            .whitespace_nowrap()
            .child(self.code);

        div()
            .id(id)
            .flex()
            .flex_col()
            .rounded_md()
            .bg(theme.colors.surface_elevated.opacity(0.35))
            .border_1()
            .border_color(theme.colors.border.opacity(0.6))
            .child(header)
            .child(body)
    }
}
