use gpui::SharedString;

/// Shared UI state for a user-triggered action.
///
/// Conventions:
/// - Set `in_flight = true` immediately on trigger.
/// - Disable the triggering control while `in_flight` unless concurrent actions are safe.
/// - Render `error` in-place with actionable text; preserve user input.
#[derive(Debug, Default, Clone)]
pub struct UserActionState {
    pub in_flight: bool,
    pub error: Option<SharedString>,
}

impl UserActionState {
    pub fn start(&mut self) {
        self.in_flight = true;
        self.error = None;
    }

    pub fn succeed(&mut self) {
        self.in_flight = false;
    }

    pub fn fail(&mut self, error: impl Into<SharedString>) {
        self.in_flight = false;
        self.error = Some(error.into());
    }

    pub fn clear_error(&mut self) {
        self.error = None;
    }
}
