use std::collections::BTreeSet;

use redesmyn_protocol::client::{MergeReadiness, TaskState};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskFilterCategory {
    TaskState,
    MergeReadiness,
    AgentStatus,
}

impl TaskFilterCategory {
    pub const ALL: [Self; 3] = [Self::TaskState, Self::MergeReadiness, Self::AgentStatus];

    pub fn title(self) -> &'static str {
        match self {
            Self::TaskState => "Task state",
            Self::MergeReadiness => "Merge readiness",
            Self::AgentStatus => "Agent status",
        }
    }

    pub fn plural_title(self) -> &'static str {
        match self {
            Self::TaskState => "task states",
            Self::MergeReadiness => "merge readiness states",
            Self::AgentStatus => "agent statuses",
        }
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum TaskAgentStatus {
    Unknown,
    Running,
    Blocked,
    Stopped,
    Error,
}

impl TaskAgentStatus {
    pub const ALL: [Self; 5] = [
        Self::Unknown,
        Self::Running,
        Self::Blocked,
        Self::Stopped,
        Self::Error,
    ];

    pub fn title(self) -> &'static str {
        match self {
            Self::Unknown => "Unknown",
            Self::Running => "Running",
            Self::Blocked => "Blocked",
            Self::Stopped => "Stopped",
            Self::Error => "Error",
        }
    }
}

pub const TASK_STATE_OPTIONS: [TaskState; 5] = [
    TaskState::Unknown,
    TaskState::Todo,
    TaskState::InProgress,
    TaskState::Blocked,
    TaskState::Done,
];

pub fn task_state_title(value: TaskState) -> &'static str {
    match value {
        TaskState::Unknown => "Unknown",
        TaskState::Todo => "Todo",
        TaskState::InProgress => "In progress",
        TaskState::Blocked => "Blocked",
        TaskState::Done => "Done",
    }
}

pub const MERGE_READINESS_OPTIONS: [MergeReadiness; 3] = [
    MergeReadiness::Unknown,
    MergeReadiness::Ready,
    MergeReadiness::Blocked,
];

pub fn merge_readiness_title(value: MergeReadiness) -> &'static str {
    match value {
        MergeReadiness::Unknown => "Unknown",
        MergeReadiness::Ready => "Ready",
        MergeReadiness::Blocked => "Blocked",
    }
}

pub trait TaskFilterTarget {
    fn task_state(&self) -> TaskState;
    fn merge_readiness(&self) -> MergeReadiness;
    fn agent_status(&self) -> TaskAgentStatus;
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TaskFilters {
    pub task_states: BTreeSet<TaskState>,
    pub merge_readiness: BTreeSet<MergeReadiness>,
    pub agent_statuses: BTreeSet<TaskAgentStatus>,
}

impl TaskFilters {
    pub fn is_active(&self) -> bool {
        !self.task_states.is_empty()
            || !self.merge_readiness.is_empty()
            || !self.agent_statuses.is_empty()
    }

    pub fn selected_count(&self, category: TaskFilterCategory) -> usize {
        match category {
            TaskFilterCategory::TaskState => self.task_states.len(),
            TaskFilterCategory::MergeReadiness => self.merge_readiness.len(),
            TaskFilterCategory::AgentStatus => self.agent_statuses.len(),
        }
    }

    pub fn matches(&self, target: &impl TaskFilterTarget) -> bool {
        if !self.task_states.is_empty() && !self.task_states.contains(&target.task_state()) {
            return false;
        }

        if !self.merge_readiness.is_empty()
            && !self.merge_readiness.contains(&target.merge_readiness())
        {
            return false;
        }

        if !self.agent_statuses.is_empty() && !self.agent_statuses.contains(&target.agent_status())
        {
            return false;
        }

        true
    }

    pub fn chip_label(&self, category: TaskFilterCategory) -> Option<String> {
        let count = self.selected_count(category);
        if count == 0 {
            return None;
        }

        Some(format!(
            "{category} is any of {count} {plural}",
            category = category.title(),
            plural = category.plural_title()
        ))
    }
}
