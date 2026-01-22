use gpui::{
    AbsoluteLength, AnyElement, App, ElementId, ParentElement, RenderOnce, ScrollHandle, Window,
    div, prelude::*,
};

#[derive(IntoElement)]
pub struct ScrollArea {
    id: ElementId,
    scroll_handle: ScrollHandle,
    scrollbar_width: Option<AbsoluteLength>,
    children: Vec<AnyElement>,
}

impl ScrollArea {
    pub fn new(id: impl Into<ElementId>, scroll_handle: ScrollHandle) -> Self {
        Self {
            id: id.into(),
            scroll_handle,
            scrollbar_width: None,
            children: Vec::new(),
        }
    }

    pub fn scrollbar_width(mut self, width: impl Into<AbsoluteLength>) -> Self {
        self.scrollbar_width = Some(width.into());
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
        let mut root = div()
            .id(self.id)
            .size_full()
            .overflow_y_scroll()
            .track_scroll(&self.scroll_handle)
            .block_mouse_except_scroll();

        if let Some(scrollbar_width) = self.scrollbar_width {
            root = root.scrollbar_width(scrollbar_width);
        }

        root.extend(self.children);
        root
    }
}
