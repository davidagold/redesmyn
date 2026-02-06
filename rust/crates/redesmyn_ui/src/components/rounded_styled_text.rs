use std::{cell::RefCell, rc::Rc, sync::Arc};

use gpui::{
    App, AvailableSpace, Bounds, Element, ElementId, GlobalElementId, InspectorElementId, LayoutId,
    Pixels, Point, SharedString, Size, TextAlign, TextRun, WhiteSpace, Window, fill, point, px,
    size,
};

#[derive(Debug, Clone, Copy)]
pub struct RoundedBackgroundStyle {
    pub corner_radius: Pixels,
    pub trim_horizontal: bool,
    pub padding_x: Pixels,
    pub padding_y: Pixels,
}

impl Default for RoundedBackgroundStyle {
    fn default() -> Self {
        Self {
            corner_radius: px(4.0),
            trim_horizontal: false,
            padding_x: px(3.0),
            padding_y: px(2.0),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct BackgroundSpan {
    start: usize,
    end: usize,
    color: gpui::Hsla,
}

/// `StyledText`, but paints any `TextRun.background_color` runs with rounded rects.
///
/// This exists because GPUI currently paints `TextRun.background_color` as square rectangles and
/// because emitting many small inline elements (e.g. to fake rounded "chips") can trigger
/// rendering artifacts inside virtualized lists.
pub struct RoundedStyledText {
    text: SharedString,
    runs: Option<Vec<TextRun>>,
    layout: RoundedTextLayout,
    background_style: RoundedBackgroundStyle,
}

impl RoundedStyledText {
    pub fn new(text: impl Into<SharedString>) -> Self {
        Self {
            text: text.into(),
            runs: None,
            layout: RoundedTextLayout::default(),
            background_style: RoundedBackgroundStyle::default(),
        }
    }

    pub fn with_runs(mut self, runs: Vec<TextRun>) -> Self {
        let mut remaining = self.text.as_ref();
        for run in &runs {
            remaining = remaining.get(run.len..).expect("invalid text run");
        }
        assert!(remaining.is_empty(), "invalid text run");
        self.runs = Some(runs);
        self
    }

    pub fn background_style(mut self, style: RoundedBackgroundStyle) -> Self {
        self.background_style = style;
        self
    }
}

impl gpui::IntoElement for RoundedStyledText {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for RoundedStyledText {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let runs = self.runs.take();
        let layout_id = self.layout.layout(self.text.clone(), runs, window, cx);
        (layout_id, ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _: &mut Self::RequestLayoutState,
        _window: &mut Window,
        _cx: &mut App,
    ) {
        self.layout.prepaint(bounds, self.text.as_ref());
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _: &mut Self::RequestLayoutState,
        _: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        self.layout
            .paint(self.text.as_ref(), self.background_style, window, cx);
    }
}

#[derive(Default, Clone)]
struct RoundedTextLayout(Rc<RefCell<Option<RoundedTextLayoutInner>>>);

struct RoundedTextLayoutInner {
    lines: Vec<gpui::WrappedLine>,
    background_spans: Arc<Vec<BackgroundSpan>>,
    line_height: Pixels,
    wrap_width: Option<Pixels>,
    size: Option<Size<Pixels>>,
    bounds: Option<Bounds<Pixels>>,
}

impl RoundedTextLayout {
    fn layout(
        &self,
        text: SharedString,
        runs: Option<Vec<TextRun>>,
        window: &mut Window,
        _: &mut App,
    ) -> LayoutId {
        let text_style = window.text_style();
        let font_size = text_style.font_size.to_pixels(window.rem_size());
        let line_height = text_style
            .line_height
            .to_pixels(font_size.into(), window.rem_size());

        let runs = Arc::new(runs.unwrap_or_else(|| vec![text_style.to_run(text.len())]));
        let background_spans = Arc::new(compute_background_spans(runs.as_ref()));

        window.request_measured_layout(Default::default(), {
            let element_state = self.clone();
            let text = text.clone();
            let runs = Arc::clone(&runs);
            let background_spans = Arc::clone(&background_spans);

            move |known_dimensions, available_space, window, _cx| {
                let wrap_width = if text_style.white_space == WhiteSpace::Normal {
                    known_dimensions.width.or(match available_space.width {
                        AvailableSpace::Definite(x) => Some(x),
                        _ => None,
                    })
                } else {
                    None
                };

                if let Some(text_layout) = element_state.0.borrow().as_ref()
                    && text_layout.size.is_some()
                    && (wrap_width.is_none() || wrap_width == text_layout.wrap_width)
                {
                    return text_layout.size.unwrap();
                }

                let lines = match window.text_system().shape_text(
                    text.clone(),
                    font_size,
                    runs.as_slice(),
                    wrap_width,
                    text_style.line_clamp,
                ) {
                    Ok(lines) => lines.into_vec(),
                    Err(error) => {
                        redesmyn_logging::tracing::error!(error = ?error, "Failed to shape text.");
                        Vec::new()
                    }
                };

                let mut size: Size<Pixels> = Size::default();
                for line in &lines {
                    let line_size = line.size(line_height);
                    size.height += line_size.height;
                    size.width = size.width.max(line_size.width).ceil();
                }

                element_state
                    .0
                    .borrow_mut()
                    .replace(RoundedTextLayoutInner {
                        lines,
                        background_spans: Arc::clone(&background_spans),
                        line_height,
                        wrap_width,
                        size: Some(size),
                        bounds: None,
                    });

                size
            }
        })
    }

    fn prepaint(&self, bounds: Bounds<Pixels>, text: &str) {
        let mut element_state = self.0.borrow_mut();
        let element_state = element_state
            .as_mut()
            .unwrap_or_else(|| panic!("measurement has not been performed on {text}"));
        element_state.bounds = Some(bounds);
    }

    fn paint(&self, text: &str, style: RoundedBackgroundStyle, window: &mut Window, cx: &mut App) {
        let element_state = self.0.borrow();
        let element_state = element_state
            .as_ref()
            .unwrap_or_else(|| panic!("measurement has not been performed on {text}"));
        let bounds = element_state
            .bounds
            .unwrap_or_else(|| panic!("prepaint has not been performed on {text}"));

        let line_height = element_state.line_height;
        let text_style = window.text_style();
        let font_size = text_style.font_size.to_pixels(window.rem_size());

        let mut line_origin = bounds.origin;
        let mut line_start_ix = 0usize;
        let mut span_ix = 0usize;

        for line in &element_state.lines {
            let line_len = line.len();
            let line_end_ix = line_start_ix + line_len;

            while span_ix < element_state.background_spans.len()
                && element_state.background_spans[span_ix].end <= line_start_ix
            {
                span_ix += 1;
            }

            let line_bounds = Bounds::new(
                line_origin,
                size(
                    line.unwrapped_layout.width,
                    line_height * (line.wrap_boundaries.len() as f32 + 1.0),
                ),
            );

            window.paint_layer(line_bounds, |window| {
                for span in element_state
                    .background_spans
                    .iter()
                    .skip(span_ix)
                    .take_while(|span| span.start < line_end_ix)
                {
                    let local_start = span.start.saturating_sub(line_start_ix).min(line_len);
                    let local_end = span.end.saturating_sub(line_start_ix).min(line_len);
                    if local_start >= local_end {
                        continue;
                    }

                    paint_rounded_background_span(
                        line,
                        local_start,
                        local_end,
                        span.color,
                        line_origin,
                        bounds.size.width,
                        text_style.text_align,
                        font_size,
                        line_height,
                        style,
                        window,
                    );
                }
            });

            if let Err(error) = line.paint(
                line_origin,
                line_height,
                text_style.text_align,
                Some(bounds),
                window,
                cx,
            ) {
                redesmyn_logging::tracing::error!(error = ?error, "Failed to paint text.");
            }

            line_origin.y += line.size(line_height).height;
            line_start_ix = line_end_ix + 1;
        }
    }
}

fn compute_background_spans(runs: &[TextRun]) -> Vec<BackgroundSpan> {
    let mut spans: Vec<BackgroundSpan> = Vec::new();
    let mut offset = 0usize;

    for run in runs {
        let start = offset;
        let end = offset + run.len;
        offset = end;

        let Some(color) = run.background_color else {
            continue;
        };

        if let Some(last) = spans.last_mut()
            && last.end == start
            && last.color == color
        {
            last.end = end;
            continue;
        }

        spans.push(BackgroundSpan { start, end, color });
    }

    spans
}

fn paint_rounded_background_span(
    line: &gpui::WrappedLine,
    span_start: usize,
    span_end: usize,
    color: gpui::Hsla,
    line_origin: Point<Pixels>,
    align_width: Pixels,
    align: TextAlign,
    font_size: Pixels,
    line_height: Pixels,
    style: RoundedBackgroundStyle,
    window: &mut Window,
) {
    let line_len = line.len();
    let span_start = span_start.min(line_len);
    let span_end = span_end.min(line_len);
    if span_start >= span_end {
        return;
    }

    let wrap_end_indices = line
        .wrap_boundaries
        .iter()
        .map(|boundary| line.unwrapped_layout.runs[boundary.run_ix].glyphs[boundary.glyph_ix].index)
        .filter(|&ix| ix > 0 && ix < line_len)
        .chain(std::iter::once(line_len));

    let span_start_x = line.unwrapped_layout.x_for_index(span_start);
    let span_end_x = line.unwrapped_layout.x_for_index(span_end);

    let mut segment_start = 0usize;
    let mut segment_y = px(0.0);

    for segment_end in wrap_end_indices {
        let seg_start = segment_start;
        let seg_end = segment_end;
        segment_start = seg_end;

        let local_start = span_start.max(seg_start);
        let local_end = span_end.min(seg_end);
        if local_start >= local_end {
            segment_y += line_height;
            continue;
        }

        let seg_start_x = line.unwrapped_layout.x_for_index(seg_start);
        let seg_end_x = line.unwrapped_layout.x_for_index(seg_end);
        let segment_origin_x =
            aligned_origin_x(line_origin.x, align_width, seg_end_x - seg_start_x, align);

        let mut x0 = line.unwrapped_layout.x_for_index(local_start) - seg_start_x;
        let mut x1 = line.unwrapped_layout.x_for_index(local_end) - seg_start_x;

        if style.trim_horizontal
            && let Some((trimmed_start, trimmed_end)) =
                trim_span_horizontal(line, local_start, local_end, font_size, window)
        {
            x0 = trimmed_start - seg_start_x;
            x1 = trimmed_end - seg_start_x;
        }

        let segment_width = (seg_end_x - seg_start_x).max(px(0.0));
        let mut min_x0 = px(0.0);
        if local_start == span_start && seg_start < span_start {
            let max_cover = max_whitespace_cover_before(line, span_start, span_start_x, style);
            min_x0 = (span_start_x - max_cover) - seg_start_x;
        }

        let mut max_x1 = segment_width;
        if local_end == span_end && seg_end > span_end {
            let max_cover = max_whitespace_cover_after(line, span_end, span_end_x, style);
            max_x1 = (span_end_x + max_cover) - seg_start_x;
        }

        x0 = (x0 - style.padding_x).max(min_x0).max(px(0.0));
        x1 = (x1 + style.padding_x).min(max_x1).min(segment_width);

        let width = (x1 - x0).max(px(0.0));
        if width <= px(0.0) {
            segment_y += line_height;
            continue;
        }

        let bg_height = (line_height - style.padding_y * 2.0).max(px(0.0));
        let bg_origin = point(
            segment_origin_x + x0,
            line_origin.y + segment_y + style.padding_y,
        );

        window.paint_quad(
            fill(
                Bounds {
                    origin: bg_origin,
                    size: size(width, bg_height),
                },
                color,
            )
            .corner_radii(style.corner_radius),
        );

        segment_y += line_height;
    }
}

fn trim_span_horizontal(
    line: &gpui::WrappedLine,
    start: usize,
    end: usize,
    font_size: Pixels,
    window: &Window,
) -> Option<(Pixels, Pixels)> {
    if start >= end {
        return None;
    }

    let text = line.text.as_ref();
    let span_text = text.get(start..end)?;

    let mut chars = span_text.char_indices();
    let (_, first_char) = chars.next()?;
    let (last_offset, last_char) = span_text.char_indices().last().unwrap_or((0, first_char));
    let last_index = start + last_offset;

    let font_id_start = line.unwrapped_layout.font_id_for_index(start)?;
    let font_id_end = line.unwrapped_layout.font_id_for_index(last_index)?;

    let text_system = window.text_system();
    let first_bounds = text_system
        .typographic_bounds(font_id_start, font_size, first_char)
        .ok()?;
    let last_bounds = text_system
        .typographic_bounds(font_id_end, font_size, last_char)
        .ok()?;

    let untrimmed_start = line.unwrapped_layout.x_for_index(start);
    let untrimmed_end = line.unwrapped_layout.x_for_index(end);

    let trimmed_start = untrimmed_start + first_bounds.origin.x.max(px(0.0));

    let last_origin_x = line.unwrapped_layout.x_for_index(last_index);
    let last_right_x = last_bounds.origin.x + last_bounds.size.width;
    let trimmed_end = (last_origin_x + last_right_x).min(untrimmed_end);

    if trimmed_end > trimmed_start {
        Some((trimmed_start, trimmed_end))
    } else {
        None
    }
}

fn max_whitespace_cover_before(
    line: &gpui::WrappedLine,
    span_start: usize,
    span_start_x: Pixels,
    style: RoundedBackgroundStyle,
) -> Pixels {
    if span_start == 0 {
        return style.padding_x;
    }

    let text = line.text.as_ref();
    let Some((prev_start, prev_ch)) = char_before(text, span_start) else {
        return px(0.0);
    };

    if !prev_ch.is_whitespace() {
        return px(0.0);
    }

    let prev_x = line.unwrapped_layout.x_for_index(prev_start);
    let whitespace_width = (span_start_x - prev_x).max(px(0.0));
    let min_gap = style.padding_x * 0.5;
    (whitespace_width - min_gap)
        .max(px(0.0))
        .min(style.padding_x)
}

fn max_whitespace_cover_after(
    line: &gpui::WrappedLine,
    span_end: usize,
    span_end_x: Pixels,
    style: RoundedBackgroundStyle,
) -> Pixels {
    let text = line.text.as_ref();
    if span_end >= text.len() {
        return style.padding_x;
    }

    let Some(next_ch) = char_at(text, span_end) else {
        return px(0.0);
    };

    if !next_ch.is_whitespace() {
        return px(0.0);
    }

    let next_end = span_end + next_ch.len_utf8();
    let next_end_x = line.unwrapped_layout.x_for_index(next_end);
    let whitespace_width = (next_end_x - span_end_x).max(px(0.0));
    let min_gap = style.padding_x * 0.5;
    (whitespace_width - min_gap)
        .max(px(0.0))
        .min(style.padding_x)
}

fn char_before(text: &str, index: usize) -> Option<(usize, char)> {
    let prefix = text.get(..index)?;
    prefix
        .char_indices()
        .last()
        .map(|(offset, ch)| (offset, ch))
}

fn char_at(text: &str, index: usize) -> Option<char> {
    text.get(index..)?.chars().next()
}

fn aligned_origin_x(
    origin_x: Pixels,
    align_width: Pixels,
    line_width: Pixels,
    align: TextAlign,
) -> Pixels {
    match align {
        TextAlign::Left => origin_x,
        TextAlign::Center => (origin_x * 2.0 + align_width - line_width) / 2.0,
        TextAlign::Right => origin_x + align_width - line_width,
    }
}
