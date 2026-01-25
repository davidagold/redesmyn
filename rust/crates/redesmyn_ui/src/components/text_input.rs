use std::ops::Range;

use gpui::{
    App, Bounds, ClipboardItem, Context, CursorStyle, Element, ElementId, ElementInputHandler,
    Entity, EntityInputHandler, EventEmitter, FocusHandle, Focusable, GlobalElementId, IntoElement,
    KeyBinding, LayoutId, MouseButton, PaintQuad, Pixels, Point, Render, ScrollHandle, ShapedLine,
    SharedString, Size, Style, TextRun, UTF16Selection, UnderlineStyle, Window, WrappedLine,
    actions, div, fill, point, prelude::*, px, relative, size,
};

use crate::utils::{
    text_editing::{
        line_end_offset, line_start_offset, next_word_boundary, previous_word_boundary,
    },
    theme_for_window,
};

actions!(
    redesmyn_ui_text_input,
    [
        Backspace,
        Delete,
        Left,
        Right,
        MoveWordLeft,
        MoveWordRight,
        SelectLeft,
        SelectRight,
        SelectWordLeft,
        SelectWordRight,
        SelectAll,
        MoveLineStart,
        MoveLineEnd,
        SelectLineStart,
        SelectLineEnd,
        DeleteWordBackward,
        DeleteWordForward,
        DeleteToLineStart,
        DeleteToLineEnd,
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
        KeyBinding::new("enter", Submit, Some("TextInput")),
    ]);

    #[cfg(target_os = "macos")]
    cx.bind_keys([
        KeyBinding::new("cmd-a", SelectAll, Some("TextInput")),
        KeyBinding::new("cmd-a", SelectAll, Some("TextArea")),
        KeyBinding::new("cmd-v", Paste, Some("TextInput")),
        KeyBinding::new("cmd-v", Paste, Some("TextArea")),
        KeyBinding::new("cmd-c", Copy, Some("TextInput")),
        KeyBinding::new("cmd-c", Copy, Some("TextArea")),
        KeyBinding::new("cmd-x", Cut, Some("TextInput")),
        KeyBinding::new("cmd-x", Cut, Some("TextArea")),
        KeyBinding::new("alt-left", MoveWordLeft, Some("TextInput")),
        KeyBinding::new("alt-left", MoveWordLeft, Some("TextArea")),
        KeyBinding::new("alt-right", MoveWordRight, Some("TextInput")),
        KeyBinding::new("alt-right", MoveWordRight, Some("TextArea")),
        KeyBinding::new("shift-alt-left", SelectWordLeft, Some("TextInput")),
        KeyBinding::new("shift-alt-left", SelectWordLeft, Some("TextArea")),
        KeyBinding::new("shift-alt-right", SelectWordRight, Some("TextInput")),
        KeyBinding::new("shift-alt-right", SelectWordRight, Some("TextArea")),
        KeyBinding::new("alt-backspace", DeleteWordBackward, Some("TextInput")),
        KeyBinding::new("alt-backspace", DeleteWordBackward, Some("TextArea")),
        KeyBinding::new("alt-delete", DeleteWordForward, Some("TextInput")),
        KeyBinding::new("alt-delete", DeleteWordForward, Some("TextArea")),
        KeyBinding::new("cmd-left", MoveLineStart, Some("TextInput")),
        KeyBinding::new("cmd-left", MoveLineStart, Some("TextArea")),
        KeyBinding::new("cmd-right", MoveLineEnd, Some("TextInput")),
        KeyBinding::new("cmd-right", MoveLineEnd, Some("TextArea")),
        KeyBinding::new("shift-cmd-left", SelectLineStart, Some("TextInput")),
        KeyBinding::new("shift-cmd-left", SelectLineStart, Some("TextArea")),
        KeyBinding::new("shift-cmd-right", SelectLineEnd, Some("TextInput")),
        KeyBinding::new("shift-cmd-right", SelectLineEnd, Some("TextArea")),
        KeyBinding::new("cmd-backspace", DeleteToLineStart, Some("TextInput")),
        KeyBinding::new("cmd-backspace", DeleteToLineStart, Some("TextArea")),
        KeyBinding::new("cmd-delete", DeleteToLineEnd, Some("TextInput")),
        KeyBinding::new("cmd-delete", DeleteToLineEnd, Some("TextArea")),
        KeyBinding::new("cmd-enter", Submit, Some("TextArea")),
    ]);

    #[cfg(not(target_os = "macos"))]
    cx.bind_keys([
        KeyBinding::new("ctrl-a", SelectAll, Some("TextInput")),
        KeyBinding::new("ctrl-a", SelectAll, Some("TextArea")),
        KeyBinding::new("ctrl-v", Paste, Some("TextInput")),
        KeyBinding::new("ctrl-v", Paste, Some("TextArea")),
        KeyBinding::new("ctrl-c", Copy, Some("TextInput")),
        KeyBinding::new("ctrl-c", Copy, Some("TextArea")),
        KeyBinding::new("ctrl-x", Cut, Some("TextInput")),
        KeyBinding::new("ctrl-x", Cut, Some("TextArea")),
        KeyBinding::new("ctrl-left", MoveWordLeft, Some("TextInput")),
        KeyBinding::new("ctrl-left", MoveWordLeft, Some("TextArea")),
        KeyBinding::new("ctrl-right", MoveWordRight, Some("TextInput")),
        KeyBinding::new("ctrl-right", MoveWordRight, Some("TextArea")),
        KeyBinding::new("shift-ctrl-left", SelectWordLeft, Some("TextInput")),
        KeyBinding::new("shift-ctrl-left", SelectWordLeft, Some("TextArea")),
        KeyBinding::new("shift-ctrl-right", SelectWordRight, Some("TextInput")),
        KeyBinding::new("shift-ctrl-right", SelectWordRight, Some("TextArea")),
        KeyBinding::new("ctrl-backspace", DeleteWordBackward, Some("TextInput")),
        KeyBinding::new("ctrl-backspace", DeleteWordBackward, Some("TextArea")),
        KeyBinding::new("ctrl-delete", DeleteWordForward, Some("TextInput")),
        KeyBinding::new("ctrl-delete", DeleteWordForward, Some("TextArea")),
        KeyBinding::new("home", MoveLineStart, Some("TextInput")),
        KeyBinding::new("home", MoveLineStart, Some("TextArea")),
        KeyBinding::new("end", MoveLineEnd, Some("TextInput")),
        KeyBinding::new("end", MoveLineEnd, Some("TextArea")),
        KeyBinding::new("shift-home", SelectLineStart, Some("TextInput")),
        KeyBinding::new("shift-home", SelectLineStart, Some("TextArea")),
        KeyBinding::new("shift-end", SelectLineEnd, Some("TextInput")),
        KeyBinding::new("shift-end", SelectLineEnd, Some("TextArea")),
        KeyBinding::new("ctrl-enter", Submit, Some("TextArea")),
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

    fn move_word_left(&mut self, _: &MoveWordLeft, _: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.move_to(
                previous_word_boundary(&self.content, self.cursor_offset()),
                cx,
            );
        } else {
            self.move_to(self.selected_range.start, cx)
        }
    }

    fn move_word_right(&mut self, _: &MoveWordRight, _: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.move_to(next_word_boundary(&self.content, self.cursor_offset()), cx);
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

    fn select_word_left(&mut self, _: &SelectWordLeft, _: &mut Window, cx: &mut Context<Self>) {
        self.select_to(
            previous_word_boundary(&self.content, self.cursor_offset()),
            cx,
        );
    }

    fn select_word_right(&mut self, _: &SelectWordRight, _: &mut Window, cx: &mut Context<Self>) {
        self.select_to(next_word_boundary(&self.content, self.cursor_offset()), cx);
    }

    fn select_all(&mut self, _: &SelectAll, _: &mut Window, cx: &mut Context<Self>) {
        self.selection_reversed = false;
        self.selected_range = 0..self.content.len();
        cx.notify()
    }

    fn move_line_start(&mut self, _: &MoveLineStart, _: &mut Window, cx: &mut Context<Self>) {
        self.move_to(0, cx);
    }

    fn move_line_end(&mut self, _: &MoveLineEnd, _: &mut Window, cx: &mut Context<Self>) {
        self.move_to(self.content.len(), cx);
    }

    fn select_line_start(&mut self, _: &SelectLineStart, _: &mut Window, cx: &mut Context<Self>) {
        self.select_to(0, cx);
    }

    fn select_line_end(&mut self, _: &SelectLineEnd, _: &mut Window, cx: &mut Context<Self>) {
        self.select_to(self.content.len(), cx);
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

    fn delete_word_backward(
        &mut self,
        _: &DeleteWordBackward,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.selected_range.is_empty() {
            self.select_to(
                previous_word_boundary(&self.content, self.cursor_offset()),
                cx,
            )
        }
        self.replace_text_in_range(None, "", window, cx)
    }

    fn delete_word_forward(
        &mut self,
        _: &DeleteWordForward,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.selected_range.is_empty() {
            self.select_to(next_word_boundary(&self.content, self.cursor_offset()), cx)
        }
        self.replace_text_in_range(None, "", window, cx)
    }

    fn delete_to_line_start(
        &mut self,
        _: &DeleteToLineStart,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.selected_range.is_empty() {
            self.select_to(0, cx);
        }
        self.replace_text_in_range(None, "", window, cx)
    }

    fn delete_to_line_end(
        &mut self,
        _: &DeleteToLineEnd,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.selected_range.is_empty() {
            self.select_to(self.content.len(), cx);
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

    fn on_mouse_down(
        &mut self,
        event: &gpui::MouseDownEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
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
            point(
                bounds.left() + last_layout.x_for_index(range.start),
                bounds.top(),
            ),
            point(
                bounds.left() + last_layout.x_for_index(range.end),
                bounds.bottom(),
            ),
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
            (
                input.placeholder.clone(),
                theme.colors.foreground_muted.opacity(0.8),
            )
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
                        point(
                            bounds.left() + line.x_for_index(selected_range.start),
                            bounds.top(),
                        ),
                        point(
                            bounds.left() + line.x_for_index(selected_range.end),
                            bounds.bottom(),
                        ),
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
            .on_action(cx.listener(Self::move_word_left))
            .on_action(cx.listener(Self::move_word_right))
            .on_action(cx.listener(Self::select_left))
            .on_action(cx.listener(Self::select_right))
            .on_action(cx.listener(Self::select_word_left))
            .on_action(cx.listener(Self::select_word_right))
            .on_action(cx.listener(Self::select_all))
            .on_action(cx.listener(Self::move_line_start))
            .on_action(cx.listener(Self::move_line_end))
            .on_action(cx.listener(Self::select_line_start))
            .on_action(cx.listener(Self::select_line_end))
            .on_action(cx.listener(Self::delete_word_backward))
            .on_action(cx.listener(Self::delete_word_forward))
            .on_action(cx.listener(Self::delete_to_line_start))
            .on_action(cx.listener(Self::delete_to_line_end))
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
    scroll_handle: ScrollHandle,
    content: SharedString,
    placeholder: SharedString,
    selected_range: Range<usize>,
    selection_reversed: bool,
    marked_range: Option<Range<usize>>,
    layout_cache: Option<TextAreaLayoutCache>,
    last_bounds: Option<Bounds<Pixels>>,
    last_wrap_width: Option<Pixels>,
    is_selecting: bool,
    pending_scroll_to_cursor: bool,
}

impl EventEmitter<TextInputEvent> for TextArea {}

const TEXT_AREA_MIN_ROWS: usize = 4;
const TEXT_AREA_MAX_ROWS: usize = 10;

#[derive(Clone, Debug)]
struct TextAreaLayoutCache {
    text: SharedString,
    is_placeholder: bool,
    marked_range: Option<Range<usize>>,
    font: gpui::Font,
    font_size: Pixels,
    line_height: Pixels,
    wrap_width: Option<Pixels>,
    text_color: gpui::Hsla,
    lines: Vec<WrappedLine>,
    size: Size<Pixels>,
}

impl TextAreaLayoutCache {
    fn matches(
        &self,
        text: &SharedString,
        is_placeholder: bool,
        marked_range: &Option<Range<usize>>,
        font: &gpui::Font,
        font_size: Pixels,
        line_height: Pixels,
        wrap_width: Option<Pixels>,
        text_color: gpui::Hsla,
    ) -> bool {
        self.text == *text
            && self.is_placeholder == is_placeholder
            && self.marked_range == *marked_range
            && self.font == *font
            && self.font_size == font_size
            && self.line_height == line_height
            && self.wrap_width == wrap_width
            && self.text_color == text_color
    }
}

impl TextArea {
    pub fn new(cx: &mut Context<Self>) -> Self {
        Self {
            focus_handle: cx.focus_handle(),
            scroll_handle: ScrollHandle::new(),
            content: SharedString::default(),
            placeholder: "Type…".into(),
            selected_range: 0..0,
            selection_reversed: false,
            marked_range: None,
            layout_cache: None,
            last_bounds: None,
            last_wrap_width: None,
            is_selecting: false,
            pending_scroll_to_cursor: false,
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
        self.layout_cache = None;
        self.pending_scroll_to_cursor = true;
        cx.emit(TextInputEvent::Changed(self.content.clone()));
        cx.notify();
    }

    fn ensure_layout_for_width(
        &mut self,
        wrap_width: Option<Pixels>,
        window: &mut Window,
        cx: &App,
    ) {
        let theme = theme_for_window(window, cx);
        let style = window.text_style();
        let font = style.font();
        let font_size = style.font_size.to_pixels(window.rem_size());
        let line_height = window.line_height();

        let (text, is_placeholder, text_color, marked_range) = if self.content.is_empty() {
            (
                self.placeholder.clone(),
                true,
                theme.colors.foreground_muted.opacity(0.8),
                None,
            )
        } else {
            (
                self.content.clone(),
                false,
                theme.colors.foreground,
                self.marked_range.clone(),
            )
        };

        let marked_range = marked_range
            .filter(|range| range.start < range.end)
            .map(|range| range.start.min(text.len())..range.end.min(text.len()));

        if let Some(cache) = self.layout_cache.as_ref()
            && cache.matches(
                &text,
                is_placeholder,
                &marked_range,
                &font,
                font_size,
                line_height,
                wrap_width,
                text_color,
            )
        {
            return;
        }

        let base_run = TextRun {
            len: text.len(),
            font: font.clone(),
            color: text_color.into(),
            background_color: None,
            underline: None,
            strikethrough: None,
        };

        let runs = if let Some(marked_range) = marked_range.as_ref() {
            let before_len = marked_range.start;
            let marked_len = marked_range.end.saturating_sub(marked_range.start);
            let after_len = text.len().saturating_sub(marked_range.end);

            vec![
                TextRun {
                    len: before_len,
                    ..base_run.clone()
                },
                TextRun {
                    len: marked_len,
                    underline: Some(UnderlineStyle {
                        color: Some(base_run.color),
                        thickness: px(1.0),
                        wavy: false,
                    }),
                    ..base_run.clone()
                },
                TextRun {
                    len: after_len,
                    ..base_run.clone()
                },
            ]
            .into_iter()
            .filter(|run| run.len > 0)
            .collect()
        } else {
            vec![base_run]
        };

        let lines =
            match window
                .text_system()
                .shape_text(text.clone(), font_size, &runs, wrap_width, None)
            {
                Ok(lines) => lines.into_vec(),
                Err(error) => {
                    redesmyn_logging::tracing::error!(
                        error = ?error,
                        "Failed to shape TextArea text."
                    );
                    Vec::new()
                }
            };

        let mut size = Size::<Pixels>::default();
        for line in &lines {
            let line_size = line.size(line_height);
            size.height += line_size.height;
            size.width = size.width.max(line_size.width).ceil();
        }

        size.height = size
            .height
            .max(px(f32::from(line_height) * TEXT_AREA_MIN_ROWS as f32));
        if let Some(wrap_width) = wrap_width {
            size.width = wrap_width;
        }

        self.layout_cache = Some(TextAreaLayoutCache {
            text,
            is_placeholder,
            marked_range,
            font,
            font_size,
            line_height,
            wrap_width,
            text_color,
            lines,
            size,
        });
    }

    fn caret_bounds(&self, index: usize, bounds: Bounds<Pixels>) -> Option<Bounds<Pixels>> {
        let layout = self.layout_cache.as_ref()?;
        let line_height = layout.line_height;

        let mut line_origin = bounds.origin;
        let mut line_start_ix = 0usize;
        for line in &layout.lines {
            let line_end_ix = line_start_ix + line.len();
            if index <= line_end_ix {
                let ix_within_line = index.saturating_sub(line_start_ix);
                let pos_within_line = line.position_for_index(ix_within_line, line_height)?;
                return Some(Bounds::new(
                    line_origin + pos_within_line,
                    size(px(2.), line_height),
                ));
            }

            line_origin.y += line.size(line_height).height;
            line_start_ix = line_end_ix + 1;
        }

        None
    }

    fn scroll_caret_into_view(&mut self, window: &mut Window, cx: &App) {
        let Some(bounds) = self.last_bounds else {
            self.pending_scroll_to_cursor = true;
            return;
        };

        self.ensure_layout_for_width(Some(bounds.size.width), window, cx);

        let viewport = self.scroll_handle.bounds();
        if viewport.size.height == px(0.0) {
            self.pending_scroll_to_cursor = true;
            return;
        }

        let Some(caret) = self.caret_bounds(self.cursor_offset(), bounds) else {
            self.pending_scroll_to_cursor = true;
            return;
        };

        let margin = theme_for_window(window, cx).spacing.xs;
        let offset = self.scroll_handle.offset();
        let max_offset_y = self.scroll_handle.max_offset().height;

        let new_offset_y =
            scroll_offset_y_to_reveal(offset.y, max_offset_y, viewport, caret, margin);
        if new_offset_y != offset.y {
            self.scroll_handle
                .set_offset(gpui::point(offset.x, new_offset_y));
        }

        self.pending_scroll_to_cursor = false;
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
        self.pending_scroll_to_cursor = true;
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
        self.pending_scroll_to_cursor = true;
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

    fn left(&mut self, _: &Left, window: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.move_to(self.previous_boundary(self.cursor_offset()), cx);
        } else {
            self.move_to(self.selected_range.start, cx)
        }
        self.scroll_caret_into_view(window, cx);
    }

    fn right(&mut self, _: &Right, window: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.move_to(self.next_boundary(self.selected_range.end), cx);
        } else {
            self.move_to(self.selected_range.end, cx)
        }
        self.scroll_caret_into_view(window, cx);
    }

    fn move_word_left(&mut self, _: &MoveWordLeft, window: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.move_to(
                previous_word_boundary(&self.content, self.cursor_offset()),
                cx,
            );
        } else {
            self.move_to(self.selected_range.start, cx)
        }
        self.scroll_caret_into_view(window, cx);
    }

    fn move_word_right(&mut self, _: &MoveWordRight, window: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.move_to(next_word_boundary(&self.content, self.cursor_offset()), cx);
        } else {
            self.move_to(self.selected_range.end, cx)
        }
        self.scroll_caret_into_view(window, cx);
    }

    fn select_left(&mut self, _: &SelectLeft, window: &mut Window, cx: &mut Context<Self>) {
        self.select_to(self.previous_boundary(self.cursor_offset()), cx);
        self.scroll_caret_into_view(window, cx);
    }

    fn select_right(&mut self, _: &SelectRight, window: &mut Window, cx: &mut Context<Self>) {
        self.select_to(self.next_boundary(self.cursor_offset()), cx);
        self.scroll_caret_into_view(window, cx);
    }

    fn select_word_left(
        &mut self,
        _: &SelectWordLeft,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.select_to(
            previous_word_boundary(&self.content, self.cursor_offset()),
            cx,
        );
        self.scroll_caret_into_view(window, cx);
    }

    fn select_word_right(
        &mut self,
        _: &SelectWordRight,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.select_to(next_word_boundary(&self.content, self.cursor_offset()), cx);
        self.scroll_caret_into_view(window, cx);
    }

    fn select_all(&mut self, _: &SelectAll, window: &mut Window, cx: &mut Context<Self>) {
        self.selection_reversed = false;
        self.selected_range = 0..self.content.len();
        cx.notify();
        self.scroll_caret_into_view(window, cx);
    }

    fn move_line_start(&mut self, _: &MoveLineStart, window: &mut Window, cx: &mut Context<Self>) {
        let start = line_start_offset(&self.content, self.cursor_offset());
        self.move_to(start, cx);
        self.scroll_caret_into_view(window, cx);
    }

    fn move_line_end(&mut self, _: &MoveLineEnd, window: &mut Window, cx: &mut Context<Self>) {
        let end = line_end_offset(&self.content, self.cursor_offset());
        self.move_to(end, cx);
        self.scroll_caret_into_view(window, cx);
    }

    fn select_line_start(
        &mut self,
        _: &SelectLineStart,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.select_to(line_start_offset(&self.content, self.cursor_offset()), cx);
        self.scroll_caret_into_view(window, cx);
    }

    fn select_line_end(&mut self, _: &SelectLineEnd, window: &mut Window, cx: &mut Context<Self>) {
        self.select_to(line_end_offset(&self.content, self.cursor_offset()), cx);
        self.scroll_caret_into_view(window, cx);
    }

    fn backspace(&mut self, _: &Backspace, window: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.select_to(self.previous_boundary(self.cursor_offset()), cx)
        }
        self.replace_text_in_range(None, "", window, cx);
    }

    fn delete(&mut self, _: &Delete, window: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.select_to(self.next_boundary(self.cursor_offset()), cx)
        }
        self.replace_text_in_range(None, "", window, cx);
    }

    fn delete_word_backward(
        &mut self,
        _: &DeleteWordBackward,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.selected_range.is_empty() {
            self.select_to(
                previous_word_boundary(&self.content, self.cursor_offset()),
                cx,
            )
        }
        self.replace_text_in_range(None, "", window, cx)
    }

    fn delete_word_forward(
        &mut self,
        _: &DeleteWordForward,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.selected_range.is_empty() {
            self.select_to(next_word_boundary(&self.content, self.cursor_offset()), cx)
        }
        self.replace_text_in_range(None, "", window, cx)
    }

    fn delete_to_line_start(
        &mut self,
        _: &DeleteToLineStart,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.selected_range.is_empty() {
            self.select_to(line_start_offset(&self.content, self.cursor_offset()), cx);
        }
        self.replace_text_in_range(None, "", window, cx)
    }

    fn delete_to_line_end(
        &mut self,
        _: &DeleteToLineEnd,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.selected_range.is_empty() {
            self.select_to(line_end_offset(&self.content, self.cursor_offset()), cx);
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
            self.replace_text_in_range(None, "", window, cx);
        }
    }

    fn submit(&mut self, _: &Submit, _: &mut Window, cx: &mut Context<Self>) {
        cx.emit(TextInputEvent::Submitted(self.content.clone()));
    }

    fn on_mouse_down(
        &mut self,
        event: &gpui::MouseDownEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.is_selecting = true;
        window.focus(&self.focus_handle);
        let offset = self.index_for_mouse_position(event.position, window);
        if event.modifiers.shift {
            self.select_to(offset, cx);
        } else {
            self.move_to(offset, cx)
        }
        self.scroll_caret_into_view(window, cx);
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
            self.scroll_caret_into_view(window, cx);
        }
    }

    fn index_for_mouse_position(&self, position: Point<Pixels>, _window: &Window) -> usize {
        if self.content.is_empty() {
            return 0;
        }
        let Some(bounds) = self.last_bounds else {
            return 0;
        };
        let Some(layout) = self.layout_cache.as_ref() else {
            return 0;
        };
        let Some(local) = bounds.localize(&position) else {
            return 0;
        };

        let line_height = layout.line_height;
        let mut line_origin_y = px(0.0);
        let mut line_start_ix = 0usize;

        for line in &layout.lines {
            let line_size = line.size(line_height);
            if local.y > line_origin_y + line_size.height {
                line_origin_y += line_size.height;
                line_start_ix += line.len() + 1;
                continue;
            }

            let within_line = point(local.x, local.y - line_origin_y);
            let ix = match line.closest_index_for_position(within_line, line_height) {
                Ok(ix) | Err(ix) => ix,
            };
            return (line_start_ix + ix).min(self.content.len());
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
        window: &mut Window,
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
        self.layout_cache = None;
        self.pending_scroll_to_cursor = true;
        self.scroll_caret_into_view(window, cx);

        cx.emit(TextInputEvent::Changed(self.content.clone()));
        cx.notify();
    }

    fn replace_and_mark_text_in_range(
        &mut self,
        range_utf16: Option<Range<usize>>,
        new_text: &str,
        new_selected_range: Option<Range<usize>>,
        window: &mut Window,
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
        self.layout_cache = None;
        self.pending_scroll_to_cursor = true;
        self.scroll_caret_into_view(window, cx);

        cx.emit(TextInputEvent::Changed(self.content.clone()));
        cx.notify();
    }

    fn bounds_for_range(
        &mut self,
        range_utf16: Range<usize>,
        element_bounds: Bounds<Pixels>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<Bounds<Pixels>> {
        let bounds = self.last_bounds.unwrap_or(element_bounds);
        self.ensure_layout_for_width(Some(bounds.size.width), window, cx);
        let layout = self.layout_cache.as_ref()?;

        let range = self.range_from_utf16(&range_utf16);
        let start = range.start.min(self.content.len());
        let end = range.end.min(self.content.len());
        let line_height = layout.line_height;

        let position_for_index = |index: usize| -> Option<Point<Pixels>> {
            let mut line_origin = bounds.origin;
            let mut line_start_ix = 0usize;
            for line in &layout.lines {
                let line_end_ix = line_start_ix + line.len();
                if index < line_start_ix {
                    break;
                } else if index > line_end_ix {
                    line_origin.y += line.size(line_height).height;
                    line_start_ix = line_end_ix + 1;
                    continue;
                } else {
                    let ix_within_line = index - line_start_ix;
                    return Some(
                        line_origin + line.position_for_index(ix_within_line, line_height)?,
                    );
                }
            }
            None
        };

        let start_pos = position_for_index(start)?;
        if start == end {
            return Some(Bounds::new(start_pos, size(px(2.), line_height)));
        }

        let end_pos = position_for_index(end)?;
        Some(Bounds::from_corners(
            point(start_pos.x.min(end_pos.x), start_pos.y.min(end_pos.y)),
            point(
                start_pos.x.max(end_pos.x),
                start_pos.y.max(end_pos.y) + line_height,
            ),
        ))
    }

    fn character_index_for_point(
        &mut self,
        window_point: Point<Pixels>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<usize> {
        let bounds = self.last_bounds?;
        self.ensure_layout_for_width(Some(bounds.size.width), window, cx);
        let layout = self.layout_cache.as_ref()?;
        let local = bounds.localize(&window_point)?;

        let line_height = layout.line_height;
        let mut line_origin_y = px(0.0);
        let mut line_start_ix = 0usize;

        for line in &layout.lines {
            let line_size = line.size(line_height);
            if local.y > line_origin_y + line_size.height {
                line_origin_y += line_size.height;
                line_start_ix += line.len() + 1;
                continue;
            }

            let within_line = gpui::point(local.x, local.y - line_origin_y);
            let ix = match line.closest_index_for_position(within_line, line_height) {
                Ok(ix) | Err(ix) => ix,
            };
            return Some(self.offset_to_utf16(line_start_ix + ix));
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

impl IntoElement for TextAreaElement {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for TextAreaElement {
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
        _inspector_id: Option<&gpui::InspectorElementId>,
        window: &mut Window,
        _cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let input = self.input.clone();
        let mut style = Style::default();
        style.size.width = relative(1.).into();

        let layout_id = window.request_measured_layout(
            style,
            move |known_dimensions, available_space, window, cx| {
                let wrap_width = known_dimensions.width.or(match available_space.width {
                    gpui::AvailableSpace::Definite(width) => Some(width),
                    _ => None,
                });

                let mut measured_size = Size::default();
                input.update(cx, |input, cx| {
                    input.ensure_layout_for_width(wrap_width, window, cx);
                    input.last_wrap_width = wrap_width;

                    measured_size = input
                        .layout_cache
                        .as_ref()
                        .map(|layout| layout.size)
                        .unwrap_or_default();

                    if input.pending_scroll_to_cursor {
                        input.scroll_caret_into_view(window, cx);
                    }
                });

                measured_size
            },
        );

        (layout_id, ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&gpui::InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        _window: &mut Window,
        _cx: &mut App,
    ) -> Self::PrepaintState {
        self.input
            .update(_cx, |input, _cx| input.last_bounds = Some(bounds));
        ()
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&gpui::InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        _prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        let input_handle = self.input.clone();
        input_handle.update(cx, |input, cx| {
            let focus_handle = input.focus_handle.clone();
            window.handle_input(
                &focus_handle,
                ElementInputHandler::new(bounds, input_handle.clone()),
                cx,
            );

            let theme = theme_for_window(window, cx);

            let Some(layout) = input.layout_cache.as_ref() else {
                input.last_bounds = Some(bounds);
                input.last_wrap_width = Some(bounds.size.width);
                return;
            };

            let line_height = layout.line_height;

            // Paint selection.
            let sel_start = input.selected_range.start;
            let sel_end = input.selected_range.end;
            if sel_start < sel_end {
                let selection_color = theme.colors.ring.opacity(0.2);
                let mut line_origin = bounds.origin;
                let mut line_start_ix = 0usize;

                for line in &layout.lines {
                    let line_end_ix = line_start_ix + line.len();
                    let start = sel_start.clamp(line_start_ix, line_end_ix);
                    let end = sel_end.clamp(line_start_ix, line_end_ix);

                    if start < end {
                        let local_start = start - line_start_ix;
                        let local_end = end - line_start_ix;

                        let boundary_indices = line
                            .wrap_boundaries
                            .iter()
                            .map(|boundary| {
                                line.unwrapped_layout.runs[boundary.run_ix].glyphs
                                    [boundary.glyph_ix]
                                    .index
                            })
                            .chain(std::iter::once(line.len()));

                        let mut segment_start = 0usize;
                        let mut segment_y = px(0.0);
                        for segment_end in boundary_indices {
                            let start_ix = local_start.clamp(segment_start, segment_end);
                            let end_ix = local_end.clamp(segment_start, segment_end);
                            if start_ix < end_ix {
                                let segment_start_x =
                                    line.unwrapped_layout.x_for_index(segment_start);
                                let x0 =
                                    line.unwrapped_layout.x_for_index(start_ix) - segment_start_x;
                                let x1 =
                                    line.unwrapped_layout.x_for_index(end_ix) - segment_start_x;

                                let top_left =
                                    gpui::point(line_origin.x + x0, line_origin.y + segment_y);
                                let bottom_right =
                                    gpui::point(line_origin.x + x1, top_left.y + line_height);
                                window.paint_quad(fill(
                                    Bounds::from_corners(top_left, bottom_right),
                                    selection_color,
                                ));
                            }
                            segment_start = segment_end;
                            segment_y += line_height;
                        }
                    }

                    line_origin.y += line.size(line_height).height;
                    line_start_ix = line_end_ix + 1;
                }
            }

            // Paint text.
            let text_align = window.text_style().text_align;
            let mut line_origin = bounds.origin;
            for line in &layout.lines {
                line.paint(
                    line_origin,
                    line_height,
                    text_align,
                    Some(bounds),
                    window,
                    cx,
                )
                .unwrap();
                line_origin.y += line.size(line_height).height;
            }

            // Paint cursor.
            if focus_handle.is_focused(window) && input.selected_range.is_empty() {
                let cursor_offset = input.cursor_offset();

                let mut cursor_bounds = None;
                let mut line_origin = bounds.origin;
                let mut line_start_ix = 0usize;
                for line in &layout.lines {
                    let line_end_ix = line_start_ix + line.len();
                    if cursor_offset <= line_end_ix {
                        let ix_within_line = cursor_offset.saturating_sub(line_start_ix);
                        if let Some(pos) = line.position_for_index(ix_within_line, line_height) {
                            cursor_bounds =
                                Some(Bounds::new(line_origin + pos, size(px(2.), line_height)));
                        }
                        break;
                    }
                    line_origin.y += line.size(line_height).height;
                    line_start_ix = line_end_ix + 1;
                }

                if let Some(cursor_bounds) = cursor_bounds {
                    window.paint_quad(fill(cursor_bounds, theme.colors.ring));
                }
            }

            input.last_bounds = Some(bounds);
            input.last_wrap_width = Some(bounds.size.width);
        });
    }
}

impl Render for TextArea {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = theme_for_window(window, cx);
        let line_height = window.line_height();
        let min_h = px(f32::from(line_height) * TEXT_AREA_MIN_ROWS as f32)
            + theme.spacing.sm
            + theme.spacing.sm;
        let max_h = px(f32::from(line_height) * TEXT_AREA_MAX_ROWS as f32)
            + theme.spacing.sm
            + theme.spacing.sm;

        div()
            .flex()
            .flex_col()
            .key_context("TextArea")
            .track_focus(&self.focus_handle(cx))
            .cursor(CursorStyle::IBeam)
            .block_mouse_except_scroll()
            .on_action(cx.listener(Self::backspace))
            .on_action(cx.listener(Self::delete))
            .on_action(cx.listener(Self::left))
            .on_action(cx.listener(Self::right))
            .on_action(cx.listener(Self::move_word_left))
            .on_action(cx.listener(Self::move_word_right))
            .on_action(cx.listener(Self::select_left))
            .on_action(cx.listener(Self::select_right))
            .on_action(cx.listener(Self::select_word_left))
            .on_action(cx.listener(Self::select_word_right))
            .on_action(cx.listener(Self::select_all))
            .on_action(cx.listener(Self::move_line_start))
            .on_action(cx.listener(Self::move_line_end))
            .on_action(cx.listener(Self::select_line_start))
            .on_action(cx.listener(Self::select_line_end))
            .on_action(cx.listener(Self::delete_word_backward))
            .on_action(cx.listener(Self::delete_word_forward))
            .on_action(cx.listener(Self::delete_to_line_start))
            .on_action(cx.listener(Self::delete_to_line_end))
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
            .overflow_hidden()
            .line_height(line_height)
            .text_size(window.text_style().font_size)
            .text_left()
            .child(
                div()
                    .id(("text_area_scroll", cx.entity_id()))
                    .min_h(min_h)
                    .max_h(max_h)
                    .w_full()
                    .overflow_y_scroll()
                    .track_scroll(&self.scroll_handle)
                    .px(theme.spacing.sm)
                    .py(theme.spacing.sm)
                    .text_left()
                    .child(TextAreaElement { input: cx.entity() }),
            )
    }
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

fn scroll_offset_y_to_reveal(
    current_offset_y: Pixels,
    max_offset_y: Pixels,
    viewport: Bounds<Pixels>,
    target: Bounds<Pixels>,
    margin: Pixels,
) -> Pixels {
    let mut offset_y = current_offset_y;

    let desired_top = viewport.top() + margin;
    let desired_bottom = viewport.bottom() - margin;

    if target.top() < desired_top {
        offset_y += desired_top - target.top();
    } else if target.bottom() > desired_bottom {
        offset_y -= target.bottom() - desired_bottom;
    }

    let min_offset_y = -max_offset_y;
    if offset_y < min_offset_y {
        offset_y = min_offset_y;
    }
    if offset_y > px(0.0) {
        offset_y = px(0.0);
    }

    offset_y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scroll_offset_y_to_reveal_scrolls_up() {
        let viewport = Bounds::new(point(px(0.0), px(0.0)), size(px(100.0), px(100.0)));
        let target = Bounds::new(point(px(0.0), px(-10.0)), size(px(2.0), px(10.0)));

        let new_offset =
            scroll_offset_y_to_reveal(px(-100.0), px(500.0), viewport, target, px(4.0));
        assert_eq!(new_offset, px(-86.0));
    }

    #[test]
    fn scroll_offset_y_to_reveal_scrolls_down_and_clamps() {
        let viewport = Bounds::new(point(px(0.0), px(0.0)), size(px(100.0), px(100.0)));
        let target = Bounds::new(point(px(0.0), px(150.0)), size(px(2.0), px(20.0)));

        let new_offset = scroll_offset_y_to_reveal(px(0.0), px(50.0), viewport, target, px(4.0));
        assert_eq!(new_offset, px(-50.0));
    }

    #[test]
    fn scroll_offset_y_to_reveal_noop_when_visible() {
        let viewport = Bounds::new(point(px(0.0), px(0.0)), size(px(100.0), px(100.0)));
        let target = Bounds::new(point(px(0.0), px(10.0)), size(px(2.0), px(10.0)));

        let new_offset = scroll_offset_y_to_reveal(px(-12.0), px(500.0), viewport, target, px(4.0));
        assert_eq!(new_offset, px(-12.0));
    }
}
