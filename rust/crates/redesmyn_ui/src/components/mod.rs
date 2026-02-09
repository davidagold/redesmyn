//! Reusable UI components.

mod badge;
mod button;
mod callout;
mod cascading_menu;
mod expandable;
mod markdown;
mod overlay_surface;
mod progress;
mod rounded_styled_text;
mod scroll_area;
mod scroll_fade;
mod select;
mod split_pane;
mod styled_scrollbar;
mod text_input;
mod tooltip;

pub use badge::{Badge, BadgeKind, BadgeSize, BadgeStyle};
pub use button::{ButtonKind, IconButton, TextButton};
pub use callout::{Callout, CalloutKind};
pub use cascading_menu::{
    CascadingMenu, CascadingMenuId, CascadingMenuMetrics, CascadingMenuRowStyle,
    CascadingMenuSecondarySide, CascadingMenuState, CascadingMenuSurfaceStyle,
    cascading_menu_checkbox_indicator, cascading_menu_move_left_to_primary,
    cascading_menu_radio_indicator, cascading_menu_row, cascading_menu_row_value,
    cascading_menu_surface, cascading_select_menu_item, clear_open_cascading_menu_for,
    set_open_cascading_menu, set_open_cascading_menu_for,
};
pub use expandable::Expandable;
pub use markdown::{MarkdownInlineSingleLineContent, MarkdownInlineSingleLineView, MarkdownView};
pub use overlay_surface::{OverlaySurfaceKind, overlay_surface};
pub use progress::{ProgressPill, ProgressPillKind};
pub use rounded_styled_text::{
    RoundedBackgroundStyle, RoundedStyledText, copy_active_rounded_text_selection,
};
pub use scroll_area::ScrollArea;
pub use scroll_fade::ScrollFade;
pub use select::{Select, SelectOption};
pub use split_pane::{
    SplitPane, SplitPaneAxis, SplitPaneEvent, SplitPaneResizeMode, SplitPaneState,
};
pub use styled_scrollbar::{ScrollbarAxis, ScrollbarStyle, ScrollbarTarget, StyledScrollbar};
pub use text_input::{
    TextArea, TextInput, TextInputEvent, TextInputPastedImage, bind_text_input_keys,
};
pub use tooltip::Tooltip;
