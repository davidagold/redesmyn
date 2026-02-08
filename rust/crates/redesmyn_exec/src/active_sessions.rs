use std::collections::{HashMap, HashSet};

use redesmyn_ids::{SessionId, TaskId};
use redesmyn_protocol::session::InterfaceMode;

#[derive(Debug, Default)]
pub(crate) struct ActiveSessionsByTask {
    map: HashMap<(TaskId, InterfaceMode), HashSet<SessionId>>,
}

impl ActiveSessionsByTask {
    pub(crate) fn clear(&mut self) {
        self.map.clear();
    }

    pub(crate) fn any_session(
        &self,
        task_id: TaskId,
        interface_mode: InterfaceMode,
    ) -> Option<SessionId> {
        self.map
            .get(&(task_id, interface_mode))
            .and_then(|sessions| sessions.iter().next().copied())
    }

    pub(crate) fn sessions(
        &self,
        task_id: TaskId,
        interface_mode: InterfaceMode,
    ) -> Vec<SessionId> {
        self.map
            .get(&(task_id, interface_mode))
            .map(|sessions| sessions.iter().copied().collect())
            .unwrap_or_default()
    }

    pub(crate) fn insert(
        &mut self,
        task_id: TaskId,
        interface_mode: InterfaceMode,
        session_id: SessionId,
    ) {
        self.map
            .entry((task_id, interface_mode))
            .or_default()
            .insert(session_id);
    }

    pub(crate) fn remove(
        &mut self,
        task_id: TaskId,
        interface_mode: InterfaceMode,
        session_id: SessionId,
    ) {
        let key = (task_id, interface_mode);
        let Some(active) = self.map.get_mut(&key) else {
            return;
        };

        active.remove(&session_id);
        if active.is_empty() {
            self.map.remove(&key);
        }
    }
}
