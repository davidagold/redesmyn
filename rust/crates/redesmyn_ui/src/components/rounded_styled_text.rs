use std::{cell::RefCell, cmp::Ordering, collections::HashMap, ops::Range, rc::Rc, sync::Arc};

use gpui::{
    App, AvailableSpace, Bounds, ClipboardItem, CursorStyle, DispatchPhase, Element, ElementId,
    EntityId, GlobalElementId, Hitbox, HitboxBehavior, InspectorElementId, KeyDownEvent, LayoutId,
    MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, Pixels, Point, SharedString, Size,
    TextAlign, TextRun, WhiteSpace, Window, fill, point, px, size,
};

use crate::utils::theme_for_window;

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RoundedTextCopySpan {
    pub range: Range<usize>,
    pub markdown: SharedString,
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
    id: Option<ElementId>,
    selectable: bool,
    selection_scope: Option<ElementId>,
    copy_spans: Option<Arc<[RoundedTextCopySpan]>>,
}

impl RoundedStyledText {
    pub fn new(text: impl Into<SharedString>) -> Self {
        Self {
            text: text.into(),
            runs: None,
            layout: RoundedTextLayout::default(),
            background_style: RoundedBackgroundStyle::default(),
            id: None,
            selectable: false,
            selection_scope: None,
            copy_spans: None,
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

    pub fn id(mut self, id: impl Into<ElementId>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn selectable(mut self, selectable: bool) -> Self {
        self.selectable = selectable;
        self
    }

    pub fn selection_scope(mut self, scope: impl Into<ElementId>) -> Self {
        self.selection_scope = Some(scope.into());
        self
    }

    pub(super) fn copy_spans(mut self, copy_spans: Vec<RoundedTextCopySpan>) -> Self {
        self.copy_spans = Some(copy_spans.into());
        self
    }
}

#[derive(Default)]
struct RoundedTextSelectionGlobal {
    active: Option<ActiveRoundedTextSelection>,
    elements:
        HashMap<RegisteredRoundedTextSelectionElementId, RegisteredRoundedTextSelectionElement>,
    revision: u64,
}

impl gpui::Global for RoundedTextSelectionGlobal {}

#[derive(Clone)]
struct ActiveRoundedTextSelection {
    key: RoundedTextSelectionKey,
    anchor_position: Point<Pixels>,
    head_position: Point<Pixels>,
    selecting: bool,
}

#[derive(Clone, Eq, PartialEq, Hash)]
struct RoundedTextSelectionKey {
    view_id: EntityId,
    scope_id: ElementId,
}

#[derive(Clone, Eq, PartialEq, Hash)]
struct RegisteredRoundedTextSelectionElementId {
    key: RoundedTextSelectionKey,
    element_id: ElementId,
}

#[derive(Clone)]
struct RegisteredRoundedTextSelectionElement {
    key: RoundedTextSelectionKey,
    bounds: Bounds<Pixels>,
    layout: RoundedTextLayout,
    text: SharedString,
    copy_spans: Option<Arc<[RoundedTextCopySpan]>>,
    last_seen_revision: u64,
}

impl RoundedTextSelectionGlobal {
    fn upsert_element(
        &mut self,
        element_id: ElementId,
        key: RoundedTextSelectionKey,
        bounds: Bounds<Pixels>,
        layout: RoundedTextLayout,
        text: SharedString,
        copy_spans: Option<Arc<[RoundedTextCopySpan]>>,
    ) {
        self.revision = self.revision.saturating_add(1);
        self.elements.insert(
            RegisteredRoundedTextSelectionElementId {
                key: key.clone(),
                element_id,
            },
            RegisteredRoundedTextSelectionElement {
                key,
                bounds,
                layout,
                text,
                copy_spans,
                last_seen_revision: self.revision,
            },
        );
        self.prune_stale_elements();
    }

    fn prune_stale_elements(&mut self) {
        const MAX_TRACKED_ELEMENTS: usize = 4096;
        const MAX_STALE_REVISIONS: u64 = 2048;
        if self.elements.len() <= MAX_TRACKED_ELEMENTS {
            return;
        }

        let min_seen = self.revision.saturating_sub(MAX_STALE_REVISIONS);
        self.elements
            .retain(|_, element| element.last_seen_revision >= min_seen);
    }

    fn selected_copy_text_for_key(&self, key: &RoundedTextSelectionKey) -> Option<String> {
        let active = self.active.as_ref()?;
        if &active.key != key {
            return None;
        }

        let mut pieces = self
            .elements
            .values()
            .filter(|element| &element.key == key)
            .filter_map(|element| {
                let selected = element
                    .layout
                    .selection_range_for_positions(active.anchor_position, active.head_position)?;
                let copy_text = copy_text_for_range(
                    element.text.as_ref(),
                    selected,
                    element.copy_spans.as_deref(),
                );
                (!copy_text.is_empty()).then(|| RegisteredSelectionPiece {
                    top: element.bounds.origin.y,
                    left: element.bounds.origin.x,
                    text: copy_text,
                })
            })
            .collect::<Vec<_>>();

        if pieces.is_empty() {
            return None;
        }

        pieces.sort_by(|a, b| {
            a.top
                .partial_cmp(&b.top)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.left.partial_cmp(&b.left).unwrap_or(Ordering::Equal))
        });

        let mut output = String::new();
        let mut previous_top = None;
        for piece in pieces {
            if let Some(prev_top) = previous_top
                && (piece.top < prev_top - px(0.5) || piece.top > prev_top + px(0.5))
            {
                output.push('\n');
            }
            output.push_str(&piece.text);
            previous_top = Some(piece.top);
        }

        Some(output)
    }

    fn selected_copy_text_for_active_view(&self, view_id: EntityId) -> Option<String> {
        let active = self.active.as_ref()?;
        if active.key.view_id != view_id {
            return None;
        }
        self.selected_copy_text_for_key(&active.key)
    }
}

struct RegisteredSelectionPiece {
    top: Pixels,
    left: Pixels,
    text: String,
}

impl gpui::IntoElement for RoundedStyledText {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for RoundedStyledText {
    type RequestLayoutState = ();
    type PrepaintState = Option<Hitbox>;

    fn id(&self) -> Option<ElementId> {
        self.id.clone()
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
        window: &mut Window,
        _cx: &mut App,
    ) -> Self::PrepaintState {
        self.layout.prepaint(bounds, self.text.as_ref());
        if self.selectable {
            Some(window.insert_hitbox(bounds, HitboxBehavior::Normal))
        } else {
            None
        }
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _: &mut Self::RequestLayoutState,
        hitbox: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        if self.selectable
            && let Some(hitbox) = hitbox.as_ref()
            && let Some(scope_id) = self.selection_scope.clone()
        {
            let key = RoundedTextSelectionKey {
                view_id: window.current_view(),
                scope_id,
            };

            if let (Some(element_id), Some(bounds)) = (self.id.clone(), self.layout.bounds()) {
                let global = cx.default_global::<RoundedTextSelectionGlobal>();
                global.upsert_element(
                    element_id,
                    key.clone(),
                    bounds,
                    self.layout.clone(),
                    self.text.clone(),
                    self.copy_spans.clone(),
                );
            }

            if hitbox.is_hovered(window) {
                window.set_cursor_style(CursorStyle::IBeam, hitbox);
            }

            let hitbox_for_down = hitbox.clone();
            let key_for_down = key.clone();
            window.on_mouse_event(move |event: &MouseDownEvent, phase, window, cx| {
                if event.button != MouseButton::Left {
                    return;
                }

                let global = cx.default_global::<RoundedTextSelectionGlobal>();
                if phase == DispatchPhase::Capture {
                    if !hitbox_for_down.is_hovered(window)
                        && global
                            .active
                            .as_ref()
                            .is_some_and(|active| active.key == key_for_down)
                    {
                        global.active = None;
                        window.refresh();
                    }
                    return;
                }

                if phase != DispatchPhase::Bubble || !hitbox_for_down.is_hovered(window) {
                    return;
                }

                global.active = Some(ActiveRoundedTextSelection {
                    key: key_for_down.clone(),
                    anchor_position: event.position,
                    head_position: event.position,
                    selecting: true,
                });
                cx.stop_propagation();
                window.prevent_default();
                window.refresh();
            });

            let key_for_move = key.clone();
            window.on_mouse_event(move |event: &MouseMoveEvent, phase, window, cx| {
                if phase != DispatchPhase::Bubble {
                    return;
                }

                let global = cx.default_global::<RoundedTextSelectionGlobal>();
                let Some(active) = global.active.as_mut() else {
                    return;
                };
                if active.key != key_for_move || !active.selecting {
                    return;
                }

                active.head_position = event.position;
                window.refresh();
            });

            let key_for_up = key.clone();
            window.on_mouse_event(move |event: &MouseUpEvent, phase, window, cx| {
                if phase != DispatchPhase::Bubble || event.button != MouseButton::Left {
                    return;
                }

                let global = cx.default_global::<RoundedTextSelectionGlobal>();
                let Some(active) = global.active.as_mut() else {
                    return;
                };
                if active.key != key_for_up || !active.selecting {
                    return;
                }

                active.head_position = event.position;
                active.selecting = false;
                window.refresh();
            });

            let key_for_copy = key.clone();
            window.on_key_event(move |event: &KeyDownEvent, phase, window, cx| {
                if phase != DispatchPhase::Bubble
                    || !is_copy_keystroke(event)
                    || window.default_prevented()
                {
                    return;
                }

                let global = cx.default_global::<RoundedTextSelectionGlobal>();
                let Some(copy_text) = global.selected_copy_text_for_key(&key_for_copy) else {
                    return;
                };

                cx.write_to_clipboard(ClipboardItem::new_string(copy_text));
                cx.stop_propagation();
                window.prevent_default();
            });

            let selected = {
                let global = cx.default_global::<RoundedTextSelectionGlobal>();
                global.active.as_ref().and_then(|active| {
                    if active.key == key {
                        self.layout.selection_range_for_positions(
                            active.anchor_position,
                            active.head_position,
                        )
                    } else {
                        None
                    }
                })
            };
            self.layout.paint(
                self.text.as_ref(),
                self.background_style,
                selected,
                window,
                cx,
            );
            return;
        }

        self.layout
            .paint(self.text.as_ref(), self.background_style, None, window, cx);
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

    fn bounds(&self) -> Option<Bounds<Pixels>> {
        self.0.borrow().as_ref().and_then(|state| state.bounds)
    }

    fn index_for_position(&self, position: Point<Pixels>) -> Result<usize, usize> {
        let element_state = self.0.borrow();
        let Some(element_state) = element_state.as_ref() else {
            return Err(0);
        };
        let Some(bounds) = element_state.bounds else {
            return Err(0);
        };

        if position.y < bounds.top() {
            return Err(0);
        }

        let line_height = element_state.line_height;
        let mut line_origin = bounds.origin;
        let mut line_start_ix = 0;

        for line in &element_state.lines {
            let line_bottom = line_origin.y + line.size(line_height).height;
            if position.y > line_bottom {
                line_origin.y = line_bottom;
                line_start_ix += line.len() + 1;
                continue;
            }

            let position_within_line = position - line_origin;
            match line.closest_index_for_position(position_within_line, line_height) {
                Ok(ix) | Err(ix) => return Ok(line_start_ix + ix),
            }
        }

        Err(line_start_ix.saturating_sub(1))
    }

    fn index_for_position_clamped(&self, position: Point<Pixels>) -> usize {
        match self.index_for_position(position) {
            Ok(ix) | Err(ix) => ix,
        }
    }

    fn selection_range_for_positions(
        &self,
        anchor_position: Point<Pixels>,
        head_position: Point<Pixels>,
    ) -> Option<Range<usize>> {
        let anchor = self.index_for_position_clamped(anchor_position);
        let head = self.index_for_position_clamped(head_position);
        if anchor == head {
            None
        } else if anchor < head {
            Some(anchor..head)
        } else {
            Some(head..anchor)
        }
    }

    fn paint(
        &self,
        text: &str,
        style: RoundedBackgroundStyle,
        selected_range: Option<Range<usize>>,
        window: &mut Window,
        cx: &mut App,
    ) {
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
        let selection_color = theme_for_window(window, cx).colors.ring.opacity(0.2);

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

            if let Some(selected_range) = selected_range.as_ref() {
                let sel_start = selected_range.start;
                let sel_end = selected_range.end;
                let start = sel_start.clamp(line_start_ix, line_end_ix);
                let end = sel_end.clamp(line_start_ix, line_end_ix);

                if start < end {
                    let local_start = start - line_start_ix;
                    let local_end = end - line_start_ix;
                    let boundary_indices = wrap_boundary_end_indices(line);

                    let mut segment_start = 0usize;
                    let mut segment_y = px(0.0);
                    for segment_end in boundary_indices {
                        let start_ix = local_start.clamp(segment_start, segment_end);
                        let end_ix = local_end.clamp(segment_start, segment_end);
                        if start_ix < end_ix {
                            let segment_start_x = line.unwrapped_layout.x_for_index(segment_start);
                            let x0 = line.unwrapped_layout.x_for_index(start_ix) - segment_start_x;
                            let x1 = line.unwrapped_layout.x_for_index(end_ix) - segment_start_x;

                            let top_left = point(line_origin.x + x0, line_origin.y + segment_y);
                            let bottom_right = point(line_origin.x + x1, top_left.y + line_height);
                            window.paint_quad(fill(
                                Bounds::from_corners(top_left, bottom_right),
                                selection_color,
                            ));
                        }
                        segment_start = segment_end;
                        segment_y += line_height;
                    }
                }
            }

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

fn is_copy_keystroke(event: &KeyDownEvent) -> bool {
    let modifiers = &event.keystroke.modifiers;
    let has_primary_modifier = modifiers.platform || modifiers.control;
    has_primary_modifier && !modifiers.alt && !modifiers.function && event.keystroke.key == "c"
}

pub fn copy_active_rounded_text_selection(
    event: &KeyDownEvent,
    window: &mut Window,
    cx: &mut App,
) -> bool {
    if !is_copy_keystroke(event) || window.default_prevented() {
        return false;
    }

    let view_id = window.current_view();
    let global = cx.default_global::<RoundedTextSelectionGlobal>();
    let Some(copy_text) = global.selected_copy_text_for_active_view(view_id) else {
        return false;
    };

    cx.write_to_clipboard(ClipboardItem::new_string(copy_text));
    window.prevent_default();
    true
}

fn copy_text_for_range(
    text: &str,
    selected_range: Range<usize>,
    copy_spans: Option<&[RoundedTextCopySpan]>,
) -> String {
    let start = selected_range.start.min(text.len());
    let end = selected_range.end.min(text.len());
    if start >= end {
        return String::new();
    }
    let selected_range = start..end;

    let Some(copy_spans) = copy_spans else {
        return text.get(selected_range).unwrap_or("").to_string();
    };

    let mut output = String::new();
    let mut cursor = selected_range.start;
    for span in copy_spans {
        if span.range.end <= selected_range.start || span.range.start >= selected_range.end {
            continue;
        }

        let overlap_start = selected_range.start.max(span.range.start);
        let overlap_end = selected_range.end.min(span.range.end);

        if cursor < overlap_start {
            if let Some(slice) = text.get(cursor..overlap_start) {
                output.push_str(slice);
            }
        }

        if overlap_start == span.range.start && overlap_end == span.range.end {
            output.push_str(span.markdown.as_ref());
        } else if let Some(slice) = text.get(overlap_start..overlap_end) {
            output.push_str(slice);
        }
        cursor = overlap_end;
    }

    if cursor < selected_range.end
        && let Some(slice) = text.get(cursor..selected_range.end)
    {
        output.push_str(slice);
    }

    output
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

fn wrap_boundary_end_indices(line: &gpui::WrappedLine) -> Vec<usize> {
    let mut boundaries = Vec::with_capacity(line.wrap_boundaries.len().saturating_add(1));
    for boundary in &line.wrap_boundaries {
        if let Some(run) = line.unwrapped_layout.runs.get(boundary.run_ix)
            && let Some(glyph) = run.glyphs.get(boundary.glyph_ix)
        {
            boundaries.push(glyph.index);
        }
    }
    boundaries.push(line.len());
    boundaries
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
