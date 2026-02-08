use gpui::{
    AbsoluteLength, AnyElement, App, ElementId, Hsla, ParentElement, RenderOnce, ScrollHandle,
    Window, div, prelude::*,
};

use crate::components::{ScrollbarStyle, StyledScrollbar};

#[derive(IntoElement)]
pub struct ScrollArea {
    id: ElementId,
    scroll_handle: ScrollHandle,
    scrollbar_width: Option<AbsoluteLength>,
    styled_scrollbar: Option<ScrollbarStyle>,
    background_color: Option<Hsla>,
    children: Vec<AnyElement>,
}

impl ScrollArea {
    pub fn new(id: impl Into<ElementId>, scroll_handle: ScrollHandle) -> Self {
        Self {
            id: id.into(),
            scroll_handle,
            scrollbar_width: None,
            styled_scrollbar: Some(ScrollbarStyle::default()),
            background_color: None,
            children: Vec::new(),
        }
    }

    pub fn scrollbar_width(mut self, width: impl Into<AbsoluteLength>) -> Self {
        self.scrollbar_width = Some(width.into());
        self
    }

    pub fn styled_scrollbar(mut self, enabled: bool) -> Self {
        self.styled_scrollbar = enabled.then_some(ScrollbarStyle::default());
        self
    }

    pub fn scrollbar_style(mut self, style: ScrollbarStyle) -> Self {
        self.styled_scrollbar = Some(style);
        self
    }

    pub fn bg(mut self, color: Hsla) -> Self {
        self.background_color = Some(color);
        self
    }
}

impl ParentElement for ScrollArea {
    fn extend(&mut self, elements: impl IntoIterator<Item = AnyElement>) {
        self.children.extend(elements);
    }
}

impl RenderOnce for ScrollArea {
    fn render(self, _window: &mut Window, _cx: &mut App) -> impl IntoElement {
        let id = self.id;
        let scroll_handle = self.scroll_handle;
        let scrollbar_id = (id.clone(), "scrollbar");

        let mut root = div()
            .id(id)
            .size_full()
            .overflow_y_scroll()
            .track_scroll(&scroll_handle)
            .block_mouse_except_scroll();

        if let Some(color) = self.background_color {
            root = root.bg(color);
        }

        if let Some(scrollbar_width) = self.scrollbar_width {
            root = root.scrollbar_width(scrollbar_width);
        }

        root.extend(self.children);

        if let Some(style) = self.styled_scrollbar {
            StyledScrollbar::for_scroll_handle(scrollbar_id, scroll_handle, root)
                .style(style)
                .into_any_element()
        } else {
            root.into_any_element()
        }
    }
}
