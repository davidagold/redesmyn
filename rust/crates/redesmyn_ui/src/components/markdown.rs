use std::sync::Arc;

use gpui::{
    AbsoluteLength, AnyElement, App, ClickEvent, ClipboardItem, ElementId, FontStyle, FontWeight,
    Hsla, RenderOnce, ScrollHandle, TextRun, UnderlineStyle, Window, div, px,
};

use gpui::prelude::*;

use redesmyn_markdown::{MarkdownBlock, MarkdownDoc, MarkdownInline};

use crate::utils::{OpenExternalUrl as _, theme_for_window};

use super::rounded_styled_text::RoundedTextCopySpan;
use super::{
    ButtonKind, RoundedBackgroundStyle, RoundedStyledText, ScrollbarAxis, StyledScrollbar,
    TextButton,
};

#[derive(IntoElement)]
pub struct MarkdownView {
    id: ElementId,
    doc: Arc<MarkdownDoc>,
    show_truncation_notice: bool,
    text_color: Option<Hsla>,
    text_size: Option<AbsoluteLength>,
}

impl MarkdownView {
    pub fn new(id: impl Into<ElementId>, doc: Arc<MarkdownDoc>) -> Self {
        Self {
            id: id.into(),
            doc,
            show_truncation_notice: true,
            text_color: None,
            text_size: None,
        }
    }

    pub fn show_truncation_notice(mut self, show: bool) -> Self {
        self.show_truncation_notice = show;
        self
    }

    pub fn text_color(mut self, color: Hsla) -> Self {
        self.text_color = Some(color);
        self
    }

    pub fn text_size(mut self, size: AbsoluteLength) -> Self {
        self.text_size = Some(size);
        self
    }
}

impl RenderOnce for MarkdownView {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let theme = theme_for_window(window, cx);

        let base_id = self.id.clone();
        let base_text_color = self.text_color.unwrap_or(theme.colors.foreground);
        let base_text_size = self.text_size.unwrap_or(theme.typography.body.size);

        let mut container = div()
            .id(base_id.clone())
            .flex()
            .flex_col()
            .gap(theme.spacing.sm)
            .w_full()
            .text_color(base_text_color)
            .text_size(base_text_size);

        let mut previous_block: Option<&MarkdownBlock> = None;
        for (ix, block) in self.doc.blocks.iter().enumerate() {
            let mut copy_prefixes = CopyLinePrefixes::default();
            if let Some(previous) = previous_block
                && needs_extra_block_separator(previous)
            {
                copy_prefixes = copy_prefixes.prepend_first_line("\n");
            }

            let block_id: ElementId = (base_id.clone(), format!("block-{ix}")).into();
            container = container.child(render_block(
                block_id,
                block,
                base_text_color,
                &base_id,
                0,
                copy_prefixes,
                window,
                cx,
            ));
            previous_block = Some(block);
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

#[derive(Clone, Debug)]
pub struct MarkdownInlineSingleLineContent {
    atoms: Arc<[InlineAtom]>,
    plain_text: gpui::SharedString,
}

impl MarkdownInlineSingleLineContent {
    pub fn from_doc(doc: &MarkdownDoc) -> Self {
        let mut atoms = first_markdown_inline_single_line_atoms(doc);
        for atom in &mut atoms {
            atom.link = None;
        }

        if atoms.is_empty() {
            atoms.push(InlineAtom {
                text: " ".to_string(),
                style: InlineStyle::default(),
                link: None,
            });
        }

        let mut plain_text = String::new();
        for atom in &atoms {
            plain_text.push_str(&atom.text);
        }

        Self {
            atoms: Arc::from(atoms.into_boxed_slice()),
            plain_text: gpui::SharedString::new(plain_text),
        }
    }

    pub fn plain_text(&self) -> gpui::SharedString {
        self.plain_text.clone()
    }
}

#[derive(IntoElement)]
pub struct MarkdownInlineSingleLineView {
    id: ElementId,
    content: MarkdownInlineSingleLineContent,
    color: Option<gpui::Hsla>,
}

impl MarkdownInlineSingleLineView {
    pub fn new(id: impl Into<ElementId>, content: MarkdownInlineSingleLineContent) -> Self {
        Self {
            id: id.into(),
            content,
            color: None,
        }
    }

    pub fn from_doc(id: impl Into<ElementId>, doc: &MarkdownDoc) -> Self {
        Self::new(id, MarkdownInlineSingleLineContent::from_doc(doc))
    }

    pub fn text_color(mut self, color: gpui::Hsla) -> Self {
        self.color = Some(color);
        self
    }
}

impl RenderOnce for MarkdownInlineSingleLineView {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let theme = theme_for_window(window, cx);

        let atoms = self.content.atoms.as_ref();
        let base_text_color = self.color.unwrap_or(theme.colors.foreground);
        let (text, runs) = build_inline_single_line_styled_text(atoms, base_text_color, window, cx);

        div()
            .id(self.id)
            .min_w_0()
            .w_full()
            .whitespace_nowrap()
            .truncate()
            .child(
                RoundedStyledText::new(text)
                    .with_runs(runs)
                    .background_style(RoundedBackgroundStyle {
                        corner_radius: theme.radius.sm,
                        trim_horizontal: true,
                        padding_x: px(5.0),
                        padding_y: px(2.5),
                        ..Default::default()
                    }),
            )
    }
}

fn render_block(
    id: ElementId,
    block: &MarkdownBlock,
    base_text_color: Hsla,
    selection_scope: &ElementId,
    list_depth: usize,
    copy_prefixes: CopyLinePrefixes,
    window: &mut Window,
    cx: &mut App,
) -> AnyElement {
    let theme = theme_for_window(window, cx);

    match block {
        MarkdownBlock::Paragraph { content, .. } => render_inline_flow(
            id,
            content,
            base_text_color,
            selection_scope,
            copy_prefixes,
            window,
            cx,
        ),
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
                    base_text_color,
                    selection_scope,
                    copy_prefixes,
                    window,
                    cx,
                ))
                .into_any_element()
        }
        MarkdownBlock::CodeBlock { language, code, .. } => CodeBlockView::new(id, language, code)
            .render(window, cx)
            .into_any_element(),
        MarkdownBlock::BlockQuote {
            content: blocks, ..
        } => {
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
            let nested_prefixes = if copy_prefixes.is_empty() {
                None
            } else {
                Some(copy_prefixes)
            };
            quote = quote.child(render_blocks_column(
                content_id,
                blocks,
                base_text_color,
                selection_scope,
                list_depth,
                nested_prefixes,
                window,
                cx,
            ));

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
                let display_marker = if *ordered {
                    format!("{}.", start_ix.saturating_add(ix as u64))
                } else {
                    "•".to_string()
                };
                let copy_marker = if *ordered {
                    format!("{}.", start_ix.saturating_add(ix as u64))
                } else {
                    "-".to_string()
                };

                let item_id: ElementId = (id.clone(), format!("item-{ix}")).into();
                let leading_copy_prefix = if ix == 0 {
                    copy_prefixes.first_line.as_deref()
                } else {
                    None
                };
                list = list.child(render_list_item(
                    item_id,
                    display_marker,
                    copy_marker,
                    item,
                    base_text_color,
                    selection_scope,
                    list_depth,
                    leading_copy_prefix,
                    window,
                    cx,
                ));
            }

            list.into_any_element()
        }
    }
}

fn needs_extra_block_separator(previous: &MarkdownBlock) -> bool {
    matches!(previous, MarkdownBlock::List { .. })
}

fn first_markdown_inline_single_line_atoms(doc: &MarkdownDoc) -> Vec<InlineAtom> {
    let mut out: Vec<InlineAtom> = Vec::new();

    for block in &doc.blocks {
        let inlines = match block {
            MarkdownBlock::Paragraph { content, .. } => content.as_slice(),
            MarkdownBlock::Heading { content, .. } => content.as_slice(),
            _ => continue,
        };

        let items = flatten_inlines(inlines);
        for item in items {
            match item {
                InlineItem::HardBreak => break,
                InlineItem::Atom(mut atom) => {
                    if atom.text.is_empty() {
                        continue;
                    }

                    if let Some(newline_ix) = atom.text.find('\n') {
                        let prefix = atom.text[..newline_ix].to_string();
                        if !prefix.is_empty() {
                            atom.text = prefix;
                            out.push(atom);
                        }
                        break;
                    }

                    out.push(atom);
                }
            }
        }

        break;
    }

    out
}

fn build_inline_single_line_styled_text(
    atoms: &[InlineAtom],
    base_text_color: Hsla,
    window: &mut Window,
    cx: &mut App,
) -> (String, Vec<TextRun>) {
    let theme = theme_for_window(window, cx);
    let code_bg = match theme.mode {
        crate::styles::ThemeMode::Dark => {
            // In dark mode we invert user message bubbles (light background + dark text). Use a
            // darker chip background there so inline code remains visible.
            if base_text_color == theme.colors.surface {
                theme.colors.border.opacity(0.45)
            } else {
                theme.colors.accent_foreground.opacity(0.12)
            }
        }
        crate::styles::ThemeMode::Light => {
            if base_text_color == theme.colors.surface {
                theme.colors.border.opacity(0.35)
            } else {
                theme.colors.accent.opacity(0.65)
            }
        }
    };

    let mut text = String::new();
    let mut runs = Vec::new();

    let default_color = base_text_color;

    let code_color = if base_text_color == theme.colors.surface {
        default_color.blend(theme.colors.foreground_muted.alpha(0.25))
    } else {
        default_color.blend(theme.colors.foreground_muted.alpha(0.45))
    };

    for atom in atoms {
        if atom.text.is_empty() {
            continue;
        }

        let InlineAtom {
            text: atom_text,
            style: atom_style,
            link: _,
        } = atom;

        let mut font = if atom_style.code {
            theme.typography.mono.font.clone()
        } else {
            theme.typography.body.font.clone()
        };

        if atom_style.bold {
            font.weight = FontWeight::BOLD;
        }
        if atom_style.italic {
            font.style = FontStyle::Italic;
        }

        let background_color = atom_style.code.then_some(code_bg);
        let color = if atom_style.code {
            code_color
        } else {
            default_color
        };

        runs.push(TextRun {
            len: atom_text.len(),
            font,
            color,
            background_color,
            underline: None,
            strikethrough: None,
        });
        text.push_str(atom_text);
    }

    (text, runs)
}

fn render_blocks_column(
    id: ElementId,
    blocks: &[MarkdownBlock],
    base_text_color: Hsla,
    selection_scope: &ElementId,
    list_depth: usize,
    first_block_copy_prefixes: Option<CopyLinePrefixes>,
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

    let subsequent_block_prefixes = first_block_copy_prefixes
        .as_ref()
        .map(CopyLinePrefixes::continuation_only);

    for (ix, block) in blocks.iter().enumerate() {
        let block_id: ElementId = (id.clone(), format!("block-{ix}")).into();
        let copy_prefixes = if ix == 0 {
            first_block_copy_prefixes.clone().unwrap_or_default()
        } else {
            subsequent_block_prefixes.clone().unwrap_or_default()
        };
        col = col.child(render_block(
            block_id,
            block,
            base_text_color,
            selection_scope,
            list_depth,
            copy_prefixes,
            window,
            cx,
        ));
    }

    col.into_any_element()
}

fn render_list_item(
    id: ElementId,
    marker: String,
    copy_marker: String,
    item: &redesmyn_markdown::MarkdownListItem,
    base_text_color: Hsla,
    selection_scope: &ElementId,
    list_depth: usize,
    leading_copy_prefix: Option<&str>,
    window: &mut Window,
    cx: &mut App,
) -> AnyElement {
    let theme = theme_for_window(window, cx);

    let content_id: ElementId = (id.clone(), "content").into();
    let mut copy_prefixes = CopyLinePrefixes::for_list_marker(list_depth, &copy_marker);
    if let Some(prefix) = leading_copy_prefix {
        copy_prefixes = copy_prefixes.prepend_first_line(prefix);
    }
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
        .child(render_blocks_column(
            content_id,
            &item.blocks,
            base_text_color,
            selection_scope,
            list_depth.saturating_add(1),
            Some(copy_prefixes),
            window,
            cx,
        ))
        .into_any_element()
}

fn render_inline_flow(
    id: ElementId,
    inlines: &[MarkdownInline],
    base_text_color: Hsla,
    selection_scope: &ElementId,
    copy_prefixes: CopyLinePrefixes,
    window: &mut Window,
    cx: &mut App,
) -> AnyElement {
    render_inline_segments(
        id,
        inlines,
        InlineSegmentsStyle::default(),
        base_text_color,
        selection_scope,
        copy_prefixes,
        window,
        cx,
    )
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

#[derive(Debug, Clone, Default)]
struct CopyLinePrefixes {
    first_line: Option<String>,
    continuation: Option<String>,
}

impl CopyLinePrefixes {
    fn is_empty(&self) -> bool {
        self.first_line.is_none() && self.continuation.is_none()
    }

    fn for_line(&self, line_ix: usize) -> Option<String> {
        if line_ix == 0 {
            self.first_line.clone()
        } else {
            self.continuation.clone()
        }
    }

    fn continuation_only(&self) -> Self {
        Self {
            first_line: self.continuation.clone(),
            continuation: self.continuation.clone(),
        }
    }

    fn prepend_first_line(mut self, prefix: &str) -> Self {
        if prefix.is_empty() {
            return self;
        }

        match self.first_line {
            Some(first_line) => {
                self.first_line = Some(format!("{prefix}{first_line}"));
            }
            None => {
                self.first_line = Some(prefix.to_string());
            }
        }

        self
    }

    fn for_list_marker(list_depth: usize, marker: &str) -> Self {
        let indent = "  ".repeat(list_depth);
        let first_line = format!("{indent}{marker} ");
        let continuation = " ".repeat(first_line.len());
        Self {
            first_line: Some(first_line),
            continuation: Some(continuation),
        }
    }
}

fn render_inline_segments(
    id: ElementId,
    inlines: &[MarkdownInline],
    style: InlineSegmentsStyle,
    base_text_color: Hsla,
    selection_scope: &ElementId,
    copy_prefixes: CopyLinePrefixes,
    window: &mut Window,
    cx: &mut App,
) -> AnyElement {
    let items = flatten_inlines(inlines);
    let lines = split_inline_items_into_lines(items);

    let mut flow = div().id(id.clone()).flex().flex_col().min_w_0().w_full();

    for (line_ix, atoms) in lines.into_iter().enumerate() {
        let line_id: ElementId = (id.clone(), format!("line-{line_ix}")).into();
        let line_copy_prefix = copy_prefixes.for_line(line_ix);

        if atoms.is_empty() {
            flow = flow.child(styled_text_div(
                line_id,
                vec![InlineAtom {
                    text: " ".to_string(),
                    style: InlineStyle::default(),
                    link: None,
                }],
                TextFlavor::Body(base_text_color),
                base_text_color,
                selection_scope,
                line_copy_prefix,
                window,
                cx,
            ));
            continue;
        }

        let has_links = atoms.iter().any(|atom| atom.link.is_some());
        if !has_links {
            // Render entire non-link lines as a single text element to avoid GPUI flex-wrap
            // layout/paint artifacts that can show up when we emit many inline chunks.
            flow = flow.child(styled_text_block(
                line_id,
                atoms,
                TextFlavor::Body(base_text_color),
                base_text_color,
                selection_scope,
                line_copy_prefix,
                window,
                cx,
            ));
            continue;
        }

        let mut line = div()
            .id(line_id.clone())
            .flex()
            .flex_row()
            .flex_wrap()
            .min_w_0()
            .w_full();

        let chunks = chunk_atoms(atoms);
        let mut next_chunk_copy_prefix = line_copy_prefix;
        for (chunk_ix, chunk) in chunks.into_iter().enumerate() {
            let chunk_id: ElementId = (line_id.clone(), format!("chunk-{chunk_ix}")).into();
            let chunk_copy_prefix =
                if next_chunk_copy_prefix.is_some() && matches!(chunk, InlineChunk::Text(_)) {
                    next_chunk_copy_prefix.take()
                } else {
                    None
                };
            line = line.child(render_inline_chunk(
                chunk_id,
                chunk,
                style,
                base_text_color,
                selection_scope,
                chunk_copy_prefix,
                window,
                cx,
            ));
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
    Link {
        destination: String,
        atoms: Vec<InlineAtom>,
    },
}

fn render_inline_chunk(
    id: ElementId,
    chunk: InlineChunk,
    style: InlineSegmentsStyle,
    base_text_color: Hsla,
    selection_scope: &ElementId,
    copy_prefix: Option<String>,
    window: &mut Window,
    cx: &mut App,
) -> AnyElement {
    match chunk {
        InlineChunk::Text(atoms) => render_inline_atoms(
            id,
            atoms,
            TextFlavor::Body(base_text_color),
            base_text_color,
            selection_scope,
            copy_prefix,
            window,
            cx,
        ),
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
                    style.text.get_or_insert_with(Default::default).color = Some(link_color);
                    style
                })
                .on_click(move |event: &ClickEvent, _window, cx| {
                    if event.standard_click() {
                        if !cx.open_external_url(&url) {
                            cx.write_to_clipboard(ClipboardItem::new_string(url.clone()));
                        }
                    }
                })
                .child(render_inline_atoms(
                    (id, "text").into(),
                    atoms,
                    TextFlavor::Link(link_color),
                    base_text_color,
                    selection_scope,
                    copy_prefix,
                    window,
                    cx,
                ))
                .into_any_element()
        }
    }
}

fn render_inline_atoms(
    id: ElementId,
    atoms: Vec<InlineAtom>,
    flavor: TextFlavor,
    base_text_color: Hsla,
    selection_scope: &ElementId,
    copy_prefix: Option<String>,
    window: &mut Window,
    cx: &mut App,
) -> AnyElement {
    styled_text_div(
        id,
        atoms,
        flavor,
        base_text_color,
        selection_scope,
        copy_prefix,
        window,
        cx,
    )
    .into_any_element()
}

#[derive(Clone, Copy)]
enum TextFlavor {
    Body(Hsla),
    Link(gpui::Hsla),
}

fn atom_markdown_text(
    text: &str,
    style: InlineStyle,
    link: Option<&str>,
    copy_prefix: Option<&str>,
) -> String {
    let styled = markdown_text_for_style(text, style);
    let markdown = if let Some(destination) = link {
        format!("[{styled}]({destination})")
    } else {
        styled
    };
    if let Some(prefix) = copy_prefix {
        format!("{prefix}{markdown}")
    } else {
        markdown
    }
}

fn markdown_text_for_style(text: &str, style: InlineStyle) -> String {
    if style.code {
        return markdown_code_span(text);
    }

    if style.bold && style.italic {
        format!("***{text}***")
    } else if style.bold {
        format!("**{text}**")
    } else if style.italic {
        format!("*{text}*")
    } else {
        text.to_string()
    }
}

fn markdown_code_span(text: &str) -> String {
    let mut longest_run = 0usize;
    let mut current_run = 0usize;
    for ch in text.chars() {
        if ch == '`' {
            current_run += 1;
            longest_run = longest_run.max(current_run);
        } else {
            current_run = 0;
        }
    }

    let fence = "`".repeat(longest_run.saturating_add(1));
    format!("{fence}{text}{fence}")
}

fn styled_text_div(
    id: ElementId,
    atoms: Vec<InlineAtom>,
    flavor: TextFlavor,
    base_text_color: Hsla,
    selection_scope: &ElementId,
    copy_prefix: Option<String>,
    window: &mut Window,
    cx: &mut App,
) -> impl IntoElement {
    let selectable = matches!(flavor, TextFlavor::Body(_));
    let theme = theme_for_window(window, cx);
    let code_bg = match theme.mode {
        crate::styles::ThemeMode::Dark => {
            // In dark mode we invert user message bubbles (light background + dark text). Use a
            // darker chip background there so inline code remains visible.
            if base_text_color == theme.colors.surface {
                theme.colors.border.opacity(0.45)
            } else {
                theme.colors.accent_foreground.opacity(0.12)
            }
        }
        crate::styles::ThemeMode::Light => {
            if base_text_color == theme.colors.surface {
                theme.colors.border.opacity(0.35)
            } else {
                theme.colors.accent.opacity(0.65)
            }
        }
    };
    let mut text = String::new();
    let mut runs = Vec::new();
    let mut copy_spans: Vec<RoundedTextCopySpan> = Vec::new();

    let (default_color, underline) = match flavor {
        TextFlavor::Body(color) => (color, None),
        TextFlavor::Link(color) => (
            color,
            Some(UnderlineStyle {
                color: Some(color),
                thickness: px(1.0),
                wavy: false,
            }),
        ),
    };

    let code_color = if base_text_color == theme.colors.surface {
        default_color.blend(theme.colors.foreground_muted.alpha(0.25))
    } else {
        default_color.blend(theme.colors.foreground_muted.alpha(0.45))
    };

    for atom in atoms {
        if atom.text.is_empty() {
            continue;
        }

        let InlineAtom {
            text: atom_text,
            style: atom_style,
            link,
        } = atom;

        let mut font = if atom_style.code {
            theme.typography.mono.font.clone()
        } else {
            theme.typography.body.font.clone()
        };

        if atom_style.bold {
            font.weight = FontWeight::BOLD;
        }
        if atom_style.italic {
            font.style = FontStyle::Italic;
        }

        let background_color = atom_style.code.then_some(code_bg);
        let color = if atom_style.code {
            code_color
        } else {
            default_color
        };

        runs.push(TextRun {
            len: atom_text.len(),
            font,
            color,
            background_color,
            underline,
            strikethrough: None,
        });
        let start = text.len();
        text.push_str(&atom_text);
        let end = text.len();
        if selectable {
            let copy_prefix = if copy_spans.is_empty() {
                copy_prefix.as_deref()
            } else {
                None
            };
            copy_spans.push(RoundedTextCopySpan {
                range: start..end,
                markdown: atom_markdown_text(&atom_text, atom_style, link.as_deref(), copy_prefix)
                    .into(),
            });
        }
    }

    let mut text_element = RoundedStyledText::new(text)
        .id((id.clone(), "text"))
        .with_runs(runs)
        .background_style(RoundedBackgroundStyle {
            corner_radius: theme.radius.sm,
            trim_horizontal: true,
            padding_x: px(5.0),
            padding_y: px(2.5),
            ..Default::default()
        });
    if selectable {
        text_element = text_element
            .selectable(true)
            .selection_scope(selection_scope.clone())
            .copy_spans(copy_spans);
    }

    div().id(id).min_w_0().flex_shrink().child(text_element)
}

fn styled_text_block(
    id: ElementId,
    atoms: Vec<InlineAtom>,
    flavor: TextFlavor,
    base_text_color: Hsla,
    selection_scope: &ElementId,
    copy_prefix: Option<String>,
    window: &mut Window,
    cx: &mut App,
) -> impl IntoElement {
    let selectable = matches!(flavor, TextFlavor::Body(_));
    // Similar to `styled_text_div`, but forces the line to take full width (which helps keep GPUI's
    // layout stable when inline spans mix styles).
    let theme = theme_for_window(window, cx);
    let code_bg = match theme.mode {
        crate::styles::ThemeMode::Dark => {
            // In dark mode we invert user message bubbles (light background + dark text). Use a
            // darker chip background there so inline code remains visible.
            if base_text_color == theme.colors.surface {
                theme.colors.border.opacity(0.45)
            } else {
                theme.colors.accent_foreground.opacity(0.12)
            }
        }
        crate::styles::ThemeMode::Light => {
            if base_text_color == theme.colors.surface {
                theme.colors.border.opacity(0.35)
            } else {
                theme.colors.accent.opacity(0.65)
            }
        }
    };
    let mut text = String::new();
    let mut runs = Vec::new();
    let mut copy_spans: Vec<RoundedTextCopySpan> = Vec::new();

    let (default_color, underline) = match flavor {
        TextFlavor::Body(color) => (color, None),
        TextFlavor::Link(color) => (
            color,
            Some(UnderlineStyle {
                color: Some(color),
                thickness: px(1.0),
                wavy: false,
            }),
        ),
    };

    let code_color = if base_text_color == theme.colors.surface {
        default_color.blend(theme.colors.foreground_muted.alpha(0.25))
    } else {
        default_color.blend(theme.colors.foreground_muted.alpha(0.45))
    };

    for atom in atoms {
        if atom.text.is_empty() {
            continue;
        }

        let InlineAtom {
            text: atom_text,
            style: atom_style,
            link,
        } = atom;

        let mut font = if atom_style.code {
            theme.typography.mono.font.clone()
        } else {
            theme.typography.body.font.clone()
        };

        if atom_style.bold {
            font.weight = FontWeight::BOLD;
        }
        if atom_style.italic {
            font.style = FontStyle::Italic;
        }

        let background_color = atom_style.code.then_some(code_bg);
        let color = if atom_style.code {
            code_color
        } else {
            default_color
        };

        runs.push(TextRun {
            len: atom_text.len(),
            font,
            color,
            background_color,
            underline,
            strikethrough: None,
        });
        let start = text.len();
        text.push_str(&atom_text);
        let end = text.len();
        if selectable {
            let copy_prefix = if copy_spans.is_empty() {
                copy_prefix.as_deref()
            } else {
                None
            };
            copy_spans.push(RoundedTextCopySpan {
                range: start..end,
                markdown: atom_markdown_text(&atom_text, atom_style, link.as_deref(), copy_prefix)
                    .into(),
            });
        }
    }

    let mut text_element = RoundedStyledText::new(text)
        .id((id.clone(), "text"))
        .with_runs(runs)
        .background_style(RoundedBackgroundStyle {
            corner_radius: theme.radius.sm,
            trim_horizontal: true,
            padding_x: px(5.0),
            padding_y: px(2.5),
            ..Default::default()
        });
    if selectable {
        text_element = text_element
            .selectable(true)
            .selection_scope(selection_scope.clone())
            .copy_spans(copy_spans);
    }

    div().id(id).min_w_0().w_full().child(text_element)
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
        MarkdownInline::Emphasis {
            content: children, ..
        } => {
            let mut style = style;
            style.italic = true;
            for child in children {
                flatten_inline(child, style, link, out);
            }
        }
        MarkdownInline::Strong {
            content: children, ..
        } => {
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
            (Some(InlineChunk::Link { destination, atoms }), Some(dest)) if destination == dest => {
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

        let body_scroll_handle = ScrollHandle::new();
        let body = StyledScrollbar::for_scroll_handle(
            (id.clone(), "body_scrollbar"),
            body_scroll_handle.clone(),
            div()
                .id((id.clone(), "body"))
                .overflow_x_scroll()
                .track_scroll(&body_scroll_handle)
                .scrollbar_width(px(10.0))
                .px(theme.spacing.md)
                .py(theme.spacing.md)
                .font(theme.typography.mono.font.clone())
                .text_size(theme.typography.mono.size)
                .text_color(theme.colors.foreground)
                .whitespace_nowrap()
                .child(self.code),
        )
        .axis(ScrollbarAxis::Horizontal);

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
