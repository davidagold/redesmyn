use std::ops::Range;

use gpui::{
    App, Bounds, ClipboardItem, Context, CursorStyle, Element, ElementId, ElementInputHandler,
    Entity, EntityInputHandler, EventEmitter, FocusHandle, Focusable, GlobalElementId, IntoElement,
    KeyBinding, LayoutId, MouseButton, PaintQuad, Pixels, Point, Render, SharedString, ShapedLine,
    Style, TextRun, UTF16Selection, UnderlineStyle, Window, actions, div, fill, point, px,
    relative, size, prelude::*,
};

use crate::utils::theme_for_window;

actions!(
    redesmyn_ui_text_input,
    [
        Backspace,
        Delete,
        Left,
        Right,
        SelectLeft,
        SelectRight,
        SelectAll,
        Home,
        End,
        Paste,
        Cut,
        Copy,
        Submit,
    ]
);

pub fn bind_text_input_keys(cx: &mut App) {
    cx.bind_keys([
        KeyBinding::new("backspace", Backspace, Some("TextInput")),
        KeyBinding::new("backspace", Backspace, Some("TextArea")),
        KeyBinding::new("delete", Delete, Some("TextInput")),
        KeyBinding::new("delete", Delete, Some("TextArea")),
        KeyBinding::new("left", Left, Some("TextInput")),
        KeyBinding::new("left", Left, Some("TextArea")),
        KeyBinding::new("right", Right, Some("TextInput")),
        KeyBinding::new("right", Right, Some("TextArea")),
        KeyBinding::new("shift-left", SelectLeft, Some("TextInput")),
        KeyBinding::new("shift-left", SelectLeft, Some("TextArea")),
        KeyBinding::new("shift-right", SelectRight, Some("TextInput")),
        KeyBinding::new("shift-right", SelectRight, Some("TextArea")),
        KeyBinding::new("cmd-a", SelectAll, Some("TextInput")),
        KeyBinding::new("cmd-a", SelectAll, Some("TextArea")),
        KeyBinding::new("cmd-v", Paste, Some("TextInput")),
        KeyBinding::new("cmd-v", Paste, Some("TextArea")),
        KeyBinding::new("cmd-c", Copy, Some("TextInput")),
        KeyBinding::new("cmd-c", Copy, Some("TextArea")),
        KeyBinding::new("cmd-x", Cut, Some("TextInput")),
        KeyBinding::new("cmd-x", Cut, Some("TextArea")),
        KeyBinding::new("home", Home, Some("TextInput")),
        KeyBinding::new("home", Home, Some("TextArea")),
        KeyBinding::new("end", End, Some("TextInput")),
        KeyBinding::new("end", End, Some("TextArea")),
        KeyBinding::new("enter", Submit, Some("TextInput")),
        KeyBinding::new("cmd-enter", Submit, Some("TextArea")),
    ]);
}

#[derive(Clone, Debug)]
pub enum TextInputEvent {
    Changed(SharedString),
    Submitted(SharedString),
}

pub struct TextInput {
    focus_handle: FocusHandle,
    content: SharedString,
    placeholder: SharedString,
    selected_range: Range<usize>,
    selection_reversed: bool,
    marked_range: Option<Range<usize>>,
    last_layout: Option<ShapedLine>,
    last_bounds: Option<Bounds<Pixels>>,
    is_selecting: bool,
}

impl EventEmitter<TextInputEvent> for TextInput {}

impl TextInput {
    pub fn new(cx: &mut Context<Self>) -> Self {
        Self {
            focus_handle: cx.focus_handle(),
            content: SharedString::default(),
            placeholder: "Type…".into(),
            selected_range: 0..0,
            selection_reversed: false,
            marked_range: None,
            last_layout: None,
            last_bounds: None,
            is_selecting: false,
        }
    }

    pub fn placeholder(mut self, placeholder: impl Into<SharedString>) -> Self {
        self.placeholder = placeholder.into();
        self
    }

    pub fn text(&self) -> &SharedString {
        &self.content
    }

    pub fn set_text(&mut self, text: impl Into<SharedString>, cx: &mut Context<Self>) {
        self.content = text.into();
        self.selected_range = self.content.len()..self.content.len();
        self.selection_reversed = false;
        self.marked_range = None;
        cx.emit(TextInputEvent::Changed(self.content.clone()));
        cx.notify();
    }

    fn previous_boundary(&self, offset: usize) -> usize {
        if offset == 0 {
            return 0;
        }
        self.content
            .char_indices()
            .take_while(|(ix, _)| *ix < offset)
            .last()
            .map(|(ix, _)| ix)
            .unwrap_or(0)
    }

    fn next_boundary(&self, offset: usize) -> usize {
        if offset >= self.content.len() {
            return self.content.len();
        }
        self.content
            .char_indices()
            .skip_while(|(ix, _)| *ix <= offset)
            .map(|(ix, _)| ix)
            .next()
            .unwrap_or(self.content.len())
    }

    fn move_to(&mut self, offset: usize, cx: &mut Context<Self>) {
        self.selected_range = offset..offset;
        self.selection_reversed = false;
        cx.notify()
    }

    fn select_to(&mut self, offset: usize, cx: &mut Context<Self>) {
        if self.selection_reversed {
            self.selected_range.start = offset
        } else {
            self.selected_range.end = offset
        };
        if self.selected_range.end < self.selected_range.start {
            self.selection_reversed = !self.selection_reversed;
            self.selected_range = self.selected_range.end..self.selected_range.start;
        }
        cx.notify()
    }

    fn cursor_offset(&self) -> usize {
        if self.selection_reversed {
            self.selected_range.start
        } else {
            self.selected_range.end
        }
    }

    fn left(&mut self, _: &Left, _: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.move_to(self.previous_boundary(self.cursor_offset()), cx);
        } else {
            self.move_to(self.selected_range.start, cx)
        }
    }

    fn right(&mut self, _: &Right, _: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.move_to(self.next_boundary(self.selected_range.end), cx);
        } else {
            self.move_to(self.selected_range.end, cx)
        }
    }

    fn select_left(&mut self, _: &SelectLeft, _: &mut Window, cx: &mut Context<Self>) {
        self.select_to(self.previous_boundary(self.cursor_offset()), cx);
    }

    fn select_right(&mut self, _: &SelectRight, _: &mut Window, cx: &mut Context<Self>) {
        self.select_to(self.next_boundary(self.cursor_offset()), cx);
    }

    fn select_all(&mut self, _: &SelectAll, _: &mut Window, cx: &mut Context<Self>) {
        self.selection_reversed = false;
        self.selected_range = 0..self.content.len();
        cx.notify()
    }

    fn home(&mut self, _: &Home, _: &mut Window, cx: &mut Context<Self>) {
        self.move_to(0, cx);
    }

    fn end(&mut self, _: &End, _: &mut Window, cx: &mut Context<Self>) {
        self.move_to(self.content.len(), cx);
    }

    fn backspace(&mut self, _: &Backspace, window: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.select_to(self.previous_boundary(self.cursor_offset()), cx)
        }
        self.replace_text_in_range(None, "", window, cx)
    }

    fn delete(&mut self, _: &Delete, window: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.select_to(self.next_boundary(self.cursor_offset()), cx)
        }
        self.replace_text_in_range(None, "", window, cx)
    }

    fn paste(&mut self, _: &Paste, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(text) = cx.read_from_clipboard().and_then(|item| item.text()) {
            self.replace_text_in_range(None, &text.replace('\n', " "), window, cx);
        }
    }

    fn copy(&mut self, _: &Copy, _: &mut Window, cx: &mut Context<Self>) {
        if !self.selected_range.is_empty() {
            cx.write_to_clipboard(ClipboardItem::new_string(
                self.content[self.selected_range.clone()].to_string(),
            ));
        }
    }

    fn cut(&mut self, _: &Cut, window: &mut Window, cx: &mut Context<Self>) {
        if !self.selected_range.is_empty() {
            cx.write_to_clipboard(ClipboardItem::new_string(
                self.content[self.selected_range.clone()].to_string(),
            ));
            self.replace_text_in_range(None, "", window, cx)
        }
    }

    fn submit(&mut self, _: &Submit, _: &mut Window, cx: &mut Context<Self>) {
        cx.emit(TextInputEvent::Submitted(self.content.clone()));
    }

    fn on_mouse_down(&mut self, event: &gpui::MouseDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        self.is_selecting = true;
        window.focus(&self.focus_handle);

        if event.modifiers.shift {
            self.select_to(self.index_for_mouse_position(event.position), cx);
        } else {
            self.move_to(self.index_for_mouse_position(event.position), cx)
        }
    }

    fn on_mouse_up(&mut self, _: &gpui::MouseUpEvent, _window: &mut Window, _: &mut Context<Self>) {
        self.is_selecting = false;
    }

    fn on_mouse_move(
        &mut self,
        event: &gpui::MouseMoveEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.is_selecting {
            self.select_to(self.index_for_mouse_position(event.position), cx);
        }
    }

    fn index_for_mouse_position(&self, position: Point<Pixels>) -> usize {
        if self.content.is_empty() {
            return 0;
        }

        let (Some(bounds), Some(line)) = (self.last_bounds.as_ref(), self.last_layout.as_ref())
        else {
            return 0;
        };

        if position.y < bounds.top() {
            return 0;
        }

        if position.y > bounds.bottom() {
            return self.content.len();
        }

        if position.x < bounds.left() {
            return 0;
        }

        if position.x > bounds.right() {
            return self.content.len();
        }

        line.closest_index_for_x(position.x - bounds.left())
    }

    fn offset_from_utf16(&self, offset: usize) -> usize {
        let mut utf8_offset = 0;
        let mut utf16_count = 0;

        for ch in self.content.chars() {
            if utf16_count >= offset {
                break;
            }
            utf16_count += ch.len_utf16();
            utf8_offset += ch.len_utf8();
        }

        utf8_offset
    }

    fn offset_to_utf16(&self, offset: usize) -> usize {
        let mut utf16_offset = 0;
        let mut utf8_count = 0;

        for ch in self.content.chars() {
            if utf8_count >= offset {
                break;
            }
            utf8_count += ch.len_utf8();
            utf16_offset += ch.len_utf16();
        }

        utf16_offset
    }

    fn range_to_utf16(&self, range: &Range<usize>) -> Range<usize> {
        self.offset_to_utf16(range.start)..self.offset_to_utf16(range.end)
    }

    fn range_from_utf16(&self, range: &Range<usize>) -> Range<usize> {
        self.offset_from_utf16(range.start)..self.offset_from_utf16(range.end)
    }
}

impl EntityInputHandler for TextInput {
    fn text_for_range(
        &mut self,
        range_utf16: Range<usize>,
        adjusted_range: &mut Option<Range<usize>>,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<String> {
        let range = self.range_from_utf16(&range_utf16);
        adjusted_range.replace(self.range_to_utf16(&range));
        Some(self.content[range].to_string())
    }

    fn selected_text_range(
        &mut self,
        _ignore_disabled_input: bool,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<UTF16Selection> {
        Some(UTF16Selection {
            range: self.range_to_utf16(&self.selected_range),
            reversed: self.selection_reversed,
        })
    }

    fn marked_text_range(&self, _: &mut Window, _: &mut Context<Self>) -> Option<Range<usize>> {
        self.marked_range
            .as_ref()
            .map(|range| self.range_to_utf16(range))
    }

    fn unmark_text(&mut self, _: &mut Window, cx: &mut Context<Self>) {
        self.marked_range = None;
        cx.notify();
    }

    fn replace_text_in_range(
        &mut self,
        range_utf16: Option<Range<usize>>,
        text: &str,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let text = text.replace('\n', " ");

        let replacement_range = range_utf16
            .as_ref()
            .map(|range_utf16| self.range_from_utf16(range_utf16))
            .or(self.marked_range.clone())
            .unwrap_or(self.selected_range.clone());
        let mut content = self.content.to_string();
        content.replace_range(replacement_range.clone(), &text);
        self.content = content.into();

        let cursor = replacement_range.start + text.len();
        self.selected_range = cursor..cursor;
        self.selection_reversed = false;
        self.marked_range = None;

        cx.emit(TextInputEvent::Changed(self.content.clone()));
        cx.notify();
    }

    fn replace_and_mark_text_in_range(
        &mut self,
        range_utf16: Option<Range<usize>>,
        new_text: &str,
        new_selected_range: Option<Range<usize>>,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let new_text = new_text.replace('\n', " ");

        let replacement_range = range_utf16
            .as_ref()
            .map(|range_utf16| self.range_from_utf16(range_utf16))
            .or(self.marked_range.clone())
            .unwrap_or(self.selected_range.clone());

        let mut content = self.content.to_string();
        content.replace_range(replacement_range.clone(), &new_text);
        self.content = content.into();

        self.marked_range = (!new_text.is_empty())
            .then(|| replacement_range.start..replacement_range.start + new_text.len());

        let new_selected_range = new_selected_range
            .as_ref()
            .map(|range_utf16| utf8_range_from_utf16(&new_text, range_utf16))
            .unwrap_or_else(|| new_text.len()..new_text.len());
        self.selected_range = (replacement_range.start + new_selected_range.start)
            ..(replacement_range.start + new_selected_range.end);
        self.selection_reversed = false;

        cx.emit(TextInputEvent::Changed(self.content.clone()));
        cx.notify();
    }

    fn bounds_for_range(
        &mut self,
        range_utf16: Range<usize>,
        element_bounds: Bounds<Pixels>,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<Bounds<Pixels>> {
        let bounds = self.last_bounds.unwrap_or(element_bounds);
        let last_layout = self.last_layout.as_ref()?;

        let range = self.range_from_utf16(&range_utf16);
        Some(Bounds::from_corners(
            point(bounds.left() + last_layout.x_for_index(range.start), bounds.top()),
            point(bounds.left() + last_layout.x_for_index(range.end), bounds.bottom()),
        ))
    }

    fn character_index_for_point(
        &mut self,
        point: Point<Pixels>,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<usize> {
        let last_bounds = self.last_bounds?;
        let last_layout = self.last_layout.as_ref()?;

        let point = last_bounds.localize(&point)?;
        let utf8_index = last_layout.index_for_x(point.x)?;
        Some(self.offset_to_utf16(utf8_index))
    }
}

impl Focusable for TextInput {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

struct TextInputElement {
    input: Entity<TextInput>,
}

struct TextInputPrepaintState {
    line: Option<ShapedLine>,
    cursor: Option<PaintQuad>,
    selection: Option<PaintQuad>,
}

impl IntoElement for TextInputElement {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for TextInputElement {
    type RequestLayoutState = ();
    type PrepaintState = TextInputPrepaintState;

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&gpui::InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let mut style = Style::default();
        style.size.width = relative(1.).into();
        style.size.height = window.line_height().into();
        (window.request_layout(style, [], cx), ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&gpui::InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Self::PrepaintState {
        let theme = theme_for_window(window, cx);
        let input = self.input.read(cx);

        let content = input.content.clone();
        let selected_range = input.selected_range.clone();
        let cursor = input.cursor_offset();
        let style = window.text_style();

        let (display_text, text_color) = if content.is_empty() {
            (input.placeholder.clone(), theme.colors.foreground_muted.opacity(0.8))
        } else {
            (content, theme.colors.foreground)
        };

        let run = TextRun {
            len: display_text.len(),
            font: style.font(),
            color: text_color.into(),
            background_color: None,
            underline: None,
            strikethrough: None,
        };

        let runs = if let Some(marked_range) = input.marked_range.as_ref() {
            vec![
                TextRun {
                    len: marked_range.start,
                    ..run.clone()
                },
                TextRun {
                    len: marked_range.end - marked_range.start,
                    underline: Some(UnderlineStyle {
                        color: Some(run.color),
                        thickness: px(1.0),
                        wavy: false,
                    }),
                    ..run.clone()
                },
                TextRun {
                    len: display_text.len() - marked_range.end,
                    ..run
                },
            ]
            .into_iter()
            .filter(|run| run.len > 0)
            .collect()
        } else {
            vec![run]
        };

        let font_size = style.font_size.to_pixels(window.rem_size());
        let line = window
            .text_system()
            .shape_line(display_text, font_size, &runs, None);

        let cursor_pos = line.x_for_index(cursor);
        let (selection, cursor) = if selected_range.is_empty() {
            (
                None,
                Some(fill(
                    Bounds::new(
                        point(bounds.left() + cursor_pos, bounds.top()),
                        size(px(2.), bounds.bottom() - bounds.top()),
                    ),
                    theme.colors.ring,
                )),
            )
        } else {
            (
                Some(fill(
                    Bounds::from_corners(
                        point(bounds.left() + line.x_for_index(selected_range.start), bounds.top()),
                        point(bounds.left() + line.x_for_index(selected_range.end), bounds.bottom()),
                    ),
                    theme.colors.ring.opacity(0.2),
                )),
                None,
            )
        };

        TextInputPrepaintState {
            line: Some(line),
            cursor,
            selection,
        }
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&gpui::InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        let focus_handle = self.input.read(cx).focus_handle.clone();
        window.handle_input(
            &focus_handle,
            ElementInputHandler::new(bounds, self.input.clone()),
            cx,
        );

        if let Some(selection) = prepaint.selection.take() {
            window.paint_quad(selection)
        }

        let line = prepaint.line.take().unwrap();
        line.paint(bounds.origin, window.line_height(), window, cx)
            .unwrap();

        if focus_handle.is_focused(window)
            && let Some(cursor) = prepaint.cursor.take()
        {
            window.paint_quad(cursor);
        }

        self.input.update(cx, |input, _cx| {
            input.last_layout = Some(line);
            input.last_bounds = Some(bounds);
        });
    }
}

impl Render for TextInput {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = theme_for_window(window, cx);

        div()
            .flex()
            .key_context("TextInput")
            .track_focus(&self.focus_handle(cx))
            .cursor(CursorStyle::IBeam)
            .on_action(cx.listener(Self::backspace))
            .on_action(cx.listener(Self::delete))
            .on_action(cx.listener(Self::left))
            .on_action(cx.listener(Self::right))
            .on_action(cx.listener(Self::select_left))
            .on_action(cx.listener(Self::select_right))
            .on_action(cx.listener(Self::select_all))
            .on_action(cx.listener(Self::home))
            .on_action(cx.listener(Self::end))
            .on_action(cx.listener(Self::paste))
            .on_action(cx.listener(Self::cut))
            .on_action(cx.listener(Self::copy))
            .on_action(cx.listener(Self::submit))
            .on_mouse_down(MouseButton::Left, cx.listener(Self::on_mouse_down))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .on_mouse_up_out(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .on_mouse_move(cx.listener(Self::on_mouse_move))
            .bg(theme.colors.surface_elevated)
            .border_1()
            .border_color(theme.colors.border)
            .overflow_hidden()
            .rounded(theme.radius.md)
            .line_height(window.line_height())
            .text_size(window.text_style().font_size)
            .child(
                div()
                    .h(window.line_height() + theme.spacing.sm + theme.spacing.sm)
                    .w_full()
                    .px(theme.spacing.sm)
                    .py(theme.spacing.sm)
                    .child(TextInputElement { input: cx.entity() }),
            )
    }
}

pub struct TextArea {
    focus_handle: FocusHandle,
    content: SharedString,
    placeholder: SharedString,
    selected_range: Range<usize>,
    selection_reversed: bool,
    marked_range: Option<Range<usize>>,
    last_bounds: Option<Bounds<Pixels>>,
    last_lines: Vec<ShapedLine>,
    is_selecting: bool,
}

impl EventEmitter<TextInputEvent> for TextArea {}

impl TextArea {
    pub fn new(cx: &mut Context<Self>) -> Self {
        Self {
            focus_handle: cx.focus_handle(),
            content: SharedString::default(),
            placeholder: "Type…".into(),
            selected_range: 0..0,
            selection_reversed: false,
            marked_range: None,
            last_bounds: None,
            last_lines: Vec::new(),
            is_selecting: false,
        }
    }

    pub fn placeholder(mut self, placeholder: impl Into<SharedString>) -> Self {
        self.placeholder = placeholder.into();
        self
    }

    pub fn text(&self) -> &SharedString {
        &self.content
    }

    pub fn set_text(&mut self, text: impl Into<SharedString>, cx: &mut Context<Self>) {
        self.content = text.into();
        self.selected_range = self.content.len()..self.content.len();
        self.selection_reversed = false;
        self.marked_range = None;
        cx.emit(TextInputEvent::Changed(self.content.clone()));
        cx.notify();
    }

    fn previous_boundary(&self, offset: usize) -> usize {
        if offset == 0 {
            return 0;
        }
        self.content
            .char_indices()
            .take_while(|(ix, _)| *ix < offset)
            .last()
            .map(|(ix, _)| ix)
            .unwrap_or(0)
    }

    fn next_boundary(&self, offset: usize) -> usize {
        if offset >= self.content.len() {
            return self.content.len();
        }
        self.content
            .char_indices()
            .skip_while(|(ix, _)| *ix <= offset)
            .map(|(ix, _)| ix)
            .next()
            .unwrap_or(self.content.len())
    }

    fn move_to(&mut self, offset: usize, cx: &mut Context<Self>) {
        self.selected_range = offset..offset;
        self.selection_reversed = false;
        cx.notify()
    }

    fn select_to(&mut self, offset: usize, cx: &mut Context<Self>) {
        if self.selection_reversed {
            self.selected_range.start = offset
        } else {
            self.selected_range.end = offset
        };
        if self.selected_range.end < self.selected_range.start {
            self.selection_reversed = !self.selection_reversed;
            self.selected_range = self.selected_range.end..self.selected_range.start;
        }
        cx.notify()
    }

    fn cursor_offset(&self) -> usize {
        if self.selection_reversed {
            self.selected_range.start
        } else {
            self.selected_range.end
        }
    }

    fn offset_from_utf16(&self, offset: usize) -> usize {
        let mut utf8_offset = 0;
        let mut utf16_count = 0;

        for ch in self.content.chars() {
            if utf16_count >= offset {
                break;
            }
            utf16_count += ch.len_utf16();
            utf8_offset += ch.len_utf8();
        }

        utf8_offset
    }

    fn offset_to_utf16(&self, offset: usize) -> usize {
        let mut utf16_offset = 0;
        let mut utf8_count = 0;

        for ch in self.content.chars() {
            if utf8_count >= offset {
                break;
            }
            utf8_count += ch.len_utf8();
            utf16_offset += ch.len_utf16();
        }

        utf16_offset
    }

    fn range_to_utf16(&self, range: &Range<usize>) -> Range<usize> {
        self.offset_to_utf16(range.start)..self.offset_to_utf16(range.end)
    }

    fn range_from_utf16(&self, range: &Range<usize>) -> Range<usize> {
        self.offset_from_utf16(range.start)..self.offset_from_utf16(range.end)
    }

    fn left(&mut self, _: &Left, _: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.move_to(self.previous_boundary(self.cursor_offset()), cx);
        } else {
            self.move_to(self.selected_range.start, cx)
        }
    }

    fn right(&mut self, _: &Right, _: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.move_to(self.next_boundary(self.selected_range.end), cx);
        } else {
            self.move_to(self.selected_range.end, cx)
        }
    }

    fn select_left(&mut self, _: &SelectLeft, _: &mut Window, cx: &mut Context<Self>) {
        self.select_to(self.previous_boundary(self.cursor_offset()), cx);
    }

    fn select_right(&mut self, _: &SelectRight, _: &mut Window, cx: &mut Context<Self>) {
        self.select_to(self.next_boundary(self.cursor_offset()), cx);
    }

    fn select_all(&mut self, _: &SelectAll, _: &mut Window, cx: &mut Context<Self>) {
        self.selection_reversed = false;
        self.selected_range = 0..self.content.len();
        cx.notify()
    }

    fn home(&mut self, _: &Home, _: &mut Window, cx: &mut Context<Self>) {
        self.move_to(0, cx);
    }

    fn end(&mut self, _: &End, _: &mut Window, cx: &mut Context<Self>) {
        self.move_to(self.content.len(), cx);
    }

    fn backspace(&mut self, _: &Backspace, window: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.select_to(self.previous_boundary(self.cursor_offset()), cx)
        }
        self.replace_text_in_range(None, "", window, cx)
    }

    fn delete(&mut self, _: &Delete, window: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.select_to(self.next_boundary(self.cursor_offset()), cx)
        }
        self.replace_text_in_range(None, "", window, cx)
    }

    fn paste(&mut self, _: &Paste, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(text) = cx.read_from_clipboard().and_then(|item| item.text()) {
            let text = text.replace("\r\n", "\n");
            self.replace_text_in_range(None, &text, window, cx);
        }
    }

    fn copy(&mut self, _: &Copy, _: &mut Window, cx: &mut Context<Self>) {
        if !self.selected_range.is_empty() {
            cx.write_to_clipboard(ClipboardItem::new_string(
                self.content[self.selected_range.clone()].to_string(),
            ));
        }
    }

    fn cut(&mut self, _: &Cut, window: &mut Window, cx: &mut Context<Self>) {
        if !self.selected_range.is_empty() {
            cx.write_to_clipboard(ClipboardItem::new_string(
                self.content[self.selected_range.clone()].to_string(),
            ));
            self.replace_text_in_range(None, "", window, cx)
        }
    }

    fn submit(&mut self, _: &Submit, _: &mut Window, cx: &mut Context<Self>) {
        cx.emit(TextInputEvent::Submitted(self.content.clone()));
    }

    fn on_mouse_down(&mut self, event: &gpui::MouseDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        self.is_selecting = true;
        window.focus(&self.focus_handle);
        let offset = self.index_for_mouse_position(event.position, window);
        if event.modifiers.shift {
            self.select_to(offset, cx);
        } else {
            self.move_to(offset, cx)
        }
    }

    fn on_mouse_up(&mut self, _: &gpui::MouseUpEvent, _window: &mut Window, _: &mut Context<Self>) {
        self.is_selecting = false;
    }

    fn on_mouse_move(
        &mut self,
        event: &gpui::MouseMoveEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.is_selecting {
            self.select_to(self.index_for_mouse_position(event.position, window), cx);
        }
    }

    fn index_for_mouse_position(&self, position: Point<Pixels>, window: &Window) -> usize {
        if self.content.is_empty() {
            return 0;
        }
        let Some(bounds) = self.last_bounds else {
            return 0;
        };
        let line_height = window.line_height();
        let y = f32::from(position.y - bounds.top()).max(0.0);
        let line_ix = (y / f32::from(line_height)).floor() as usize;

        let mut offset = 0usize;
        for (ix, line) in self.last_lines.iter().enumerate() {
            let line_len = line.text.len();
            if ix == line_ix {
                let x = position.x - bounds.left();
                return offset + line.closest_index_for_x(x);
            }
            offset += line_len + 1;
        }

        self.content.len()
    }
}

impl EntityInputHandler for TextArea {
    fn text_for_range(
        &mut self,
        range_utf16: Range<usize>,
        adjusted_range: &mut Option<Range<usize>>,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<String> {
        let range = self.range_from_utf16(&range_utf16);
        adjusted_range.replace(self.range_to_utf16(&range));
        Some(self.content[range].to_string())
    }

    fn selected_text_range(
        &mut self,
        _ignore_disabled_input: bool,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<UTF16Selection> {
        Some(UTF16Selection {
            range: self.range_to_utf16(&self.selected_range),
            reversed: self.selection_reversed,
        })
    }

    fn marked_text_range(&self, _: &mut Window, _: &mut Context<Self>) -> Option<Range<usize>> {
        self.marked_range
            .as_ref()
            .map(|range| self.range_to_utf16(range))
    }

    fn unmark_text(&mut self, _: &mut Window, cx: &mut Context<Self>) {
        self.marked_range = None;
        cx.notify();
    }

    fn replace_text_in_range(
        &mut self,
        range_utf16: Option<Range<usize>>,
        text: &str,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let replacement_range = range_utf16
            .as_ref()
            .map(|range_utf16| self.range_from_utf16(range_utf16))
            .or(self.marked_range.clone())
            .unwrap_or(self.selected_range.clone());
        let mut content = self.content.to_string();
        content.replace_range(replacement_range.clone(), text);
        self.content = content.into();

        let cursor = replacement_range.start + text.len();
        self.selected_range = cursor..cursor;
        self.selection_reversed = false;
        self.marked_range = None;

        cx.emit(TextInputEvent::Changed(self.content.clone()));
        cx.notify();
    }

    fn replace_and_mark_text_in_range(
        &mut self,
        range_utf16: Option<Range<usize>>,
        new_text: &str,
        new_selected_range: Option<Range<usize>>,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let replacement_range = range_utf16
            .as_ref()
            .map(|range_utf16| self.range_from_utf16(range_utf16))
            .or(self.marked_range.clone())
            .unwrap_or(self.selected_range.clone());

        let mut content = self.content.to_string();
        content.replace_range(replacement_range.clone(), new_text);
        self.content = content.into();

        self.marked_range = (!new_text.is_empty())
            .then(|| replacement_range.start..replacement_range.start + new_text.len());

        let new_selected_range = new_selected_range
            .as_ref()
            .map(|range_utf16| utf8_range_from_utf16(new_text, range_utf16))
            .unwrap_or_else(|| new_text.len()..new_text.len());
        self.selected_range = (replacement_range.start + new_selected_range.start)
            ..(replacement_range.start + new_selected_range.end);
        self.selection_reversed = false;

        cx.emit(TextInputEvent::Changed(self.content.clone()));
        cx.notify();
    }

    fn bounds_for_range(
        &mut self,
        range_utf16: Range<usize>,
        element_bounds: Bounds<Pixels>,
        window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<Bounds<Pixels>> {
        let bounds = self.last_bounds.unwrap_or(element_bounds);
        let range = self.range_from_utf16(&range_utf16);
        let line_height = window.line_height();

        let mut line_start = 0usize;
        for (line_ix, line) in self.last_lines.iter().enumerate() {
            let line_end = line_start + line.text.len();
            if range.start <= line_end {
                let start_col = range.start.saturating_sub(line_start).min(line.text.len());
                let end_col = range.end.saturating_sub(line_start).min(line.text.len());
                let y0 = bounds.top() + px(f32::from(line_height) * line_ix as f32);
                let y1 = y0 + line_height;
                return Some(Bounds::from_corners(
                    point(bounds.left() + line.x_for_index(start_col), y0),
                    point(bounds.left() + line.x_for_index(end_col), y1),
                ));
            }
            line_start = line_end + 1;
        }

        None
    }

    fn character_index_for_point(
        &mut self,
        point: Point<Pixels>,
        window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<usize> {
        let last_bounds = self.last_bounds?;
        let line_height = window.line_height();
        let y = f32::from(point.y - last_bounds.top()).max(0.0);
        let line_ix = (y / f32::from(line_height)).floor() as usize;

        let mut line_start = 0usize;
        for (ix, line) in self.last_lines.iter().enumerate() {
            if ix == line_ix {
                let local_x = point.x - last_bounds.left();
                let utf8_index = line.index_for_x(local_x)?;
                return Some(self.offset_to_utf16(line_start + utf8_index));
            }
            line_start += line.text.len() + 1;
        }

        Some(self.offset_to_utf16(self.content.len()))
    }
}

impl Focusable for TextArea {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

struct TextAreaElement {
    input: Entity<TextArea>,
}

struct TextAreaPrepaintState {
    lines: Vec<ShapedLine>,
    selection: Vec<PaintQuad>,
    cursor: Option<PaintQuad>,
}

impl IntoElement for TextAreaElement {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for TextAreaElement {
    type RequestLayoutState = ();
    type PrepaintState = TextAreaPrepaintState;

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&gpui::InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let mut style = Style::default();
        style.size.width = relative(1.).into();
        style.size.height = relative(1.).into();
        (window.request_layout(style, [], cx), ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&gpui::InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Self::PrepaintState {
        let theme = theme_for_window(window, cx);
        let input = self.input.read(cx);
        let style = window.text_style();
        let font_size = style.font_size.to_pixels(window.rem_size());
        let line_height = window.line_height();

        let mut lines = Vec::new();
        let mut selection_quads = Vec::new();
        let mut cursor = None;

        if input.content.is_empty() {
            let line = window.text_system().shape_line(
                input.placeholder.clone(),
                font_size,
                &[TextRun {
                    len: input.placeholder.len(),
                    font: style.font(),
                    color: theme.colors.foreground_muted.opacity(0.8).into(),
                    background_color: None,
                    underline: None,
                    strikethrough: None,
                }],
                None,
            );
            lines.push(line);
        } else {
            let mut line_start = 0usize;
            for text in input.content.split('\n') {
                let text: SharedString = text.to_string().into();
                let run = TextRun {
                    len: text.len(),
                    font: style.font(),
                    color: theme.colors.foreground.into(),
                    background_color: None,
                    underline: None,
                    strikethrough: None,
                };

                let runs = if let Some(marked_range) = input.marked_range.as_ref() {
                    let line_end = line_start + text.len();
                    let start = marked_range.start.clamp(line_start, line_end);
                    let end = marked_range.end.clamp(line_start, line_end);

                    if start < end {
                        let local_start = start - line_start;
                        let local_end = end - line_start;
                        vec![
                            TextRun {
                                len: local_start,
                                ..run.clone()
                            },
                            TextRun {
                                len: local_end - local_start,
                                underline: Some(UnderlineStyle {
                                    color: Some(run.color),
                                    thickness: px(1.0),
                                    wavy: false,
                                }),
                                ..run.clone()
                            },
                            TextRun {
                                len: text.len() - local_end,
                                ..run.clone()
                            },
                        ]
                        .into_iter()
                        .filter(|run| run.len > 0)
                        .collect()
                    } else {
                        vec![run.clone()]
                    }
                } else {
                    vec![run.clone()]
                };

                let line = window
                    .text_system()
                    .shape_line(text.clone(), font_size, &runs, None);
                lines.push(line);
                line_start += text.len() + 1;
            }
        }

        // Selection + cursor.
        let cursor_offset = input.cursor_offset();
        if input.selected_range.is_empty() {
            let (cursor_line, cursor_col) = cursor_line_col(&input.content, cursor_offset);
            if let Some(line) = lines.get(cursor_line) {
                let y0 = bounds.top() + px(f32::from(line_height) * cursor_line as f32);
                let y1 = y0 + line_height;
                let x = bounds.left() + line.x_for_index(cursor_col);
                cursor = Some(fill(
                    Bounds::new(point(x, y0), size(px(2.), y1 - y0)),
                    theme.colors.ring,
                ));
            }
        } else {
            // Highlight each intersecting line.
            let mut line_start = 0usize;
            for (line_ix, line) in lines.iter().enumerate() {
                let line_end = line_start + line.text.len();
                let sel_start = input.selected_range.start;
                let sel_end = input.selected_range.end;
                let start = sel_start.clamp(line_start, line_end);
                let end = sel_end.clamp(line_start, line_end);
                if start < end {
                    let start_col = start - line_start;
                    let end_col = end - line_start;
                    let y0 = bounds.top() + px(f32::from(line_height) * line_ix as f32);
                    let y1 = y0 + line_height;
                    selection_quads.push(fill(
                        Bounds::from_corners(
                            point(bounds.left() + line.x_for_index(start_col), y0),
                            point(bounds.left() + line.x_for_index(end_col), y1),
                        ),
                        theme.colors.ring.opacity(0.2),
                    ));
                }
                line_start = line_end + 1;
            }
        }

        TextAreaPrepaintState {
            lines,
            selection: selection_quads,
            cursor,
        }
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&gpui::InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        let focus_handle = self.input.read(cx).focus_handle.clone();
        window.handle_input(
            &focus_handle,
            ElementInputHandler::new(bounds, self.input.clone()),
            cx,
        );

        for quad in std::mem::take(&mut prepaint.selection) {
            window.paint_quad(quad);
        }

        let line_height = window.line_height();
        let lines = std::mem::take(&mut prepaint.lines);
        for (ix, line) in lines.iter().enumerate() {
            let origin = point(bounds.left(), bounds.top() + px(f32::from(line_height) * ix as f32));
            line.paint(origin, line_height, window, cx).unwrap();
        }

        if focus_handle.is_focused(window)
            && let Some(cursor) = prepaint.cursor.take()
        {
            window.paint_quad(cursor);
        }

        self.input.update(cx, |input, _cx| {
            input.last_bounds = Some(bounds);
            input.last_lines = lines;
        });
    }
}

impl Render for TextArea {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = theme_for_window(window, cx);

        div()
            .flex()
            .flex_col()
            .key_context("TextArea")
            .track_focus(&self.focus_handle(cx))
            .cursor(CursorStyle::IBeam)
            .on_action(cx.listener(Self::backspace))
            .on_action(cx.listener(Self::delete))
            .on_action(cx.listener(Self::left))
            .on_action(cx.listener(Self::right))
            .on_action(cx.listener(Self::select_left))
            .on_action(cx.listener(Self::select_right))
            .on_action(cx.listener(Self::select_all))
            .on_action(cx.listener(Self::home))
            .on_action(cx.listener(Self::end))
            .on_action(cx.listener(Self::paste))
            .on_action(cx.listener(Self::cut))
            .on_action(cx.listener(Self::copy))
            .on_action(cx.listener(Self::submit))
            .on_mouse_down(MouseButton::Left, cx.listener(Self::on_mouse_down))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .on_mouse_up_out(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .on_mouse_move(cx.listener(Self::on_mouse_move))
            .bg(theme.colors.surface_elevated)
            .border_1()
            .border_color(theme.colors.border)
            .rounded(theme.radius.md)
            .line_height(window.line_height())
            .text_size(window.text_style().font_size)
            .child(
                div()
                    .min_h(px(96.0))
                    .w_full()
                    .px(theme.spacing.sm)
                    .py(theme.spacing.sm)
                    .child(TextAreaElement { input: cx.entity() }),
            )
    }
}

fn cursor_line_col(text: &str, cursor: usize) -> (usize, usize) {
    let mut line = 0usize;
    let mut line_start = 0usize;
    for (ix, ch) in text.char_indices() {
        if ix >= cursor {
            break;
        }
        if ch == '\n' {
            line += 1;
            line_start = ix + 1;
        }
    }
    (line, cursor.saturating_sub(line_start))
}

fn utf8_offset_from_utf16(text: &str, offset: usize) -> usize {
    let mut utf8_offset = 0;
    let mut utf16_count = 0;

    for ch in text.chars() {
        if utf16_count >= offset {
            break;
        }
        utf16_count += ch.len_utf16();
        utf8_offset += ch.len_utf8();
    }

    utf8_offset
}

fn utf8_range_from_utf16(text: &str, range_utf16: &Range<usize>) -> Range<usize> {
    utf8_offset_from_utf16(text, range_utf16.start)..utf8_offset_from_utf16(text, range_utf16.end)
}
