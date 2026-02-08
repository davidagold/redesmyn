//! Command payloads/events for local doc sync operations.

pub const LOCAL_SYNC_FROM_DOCS_KIND: &str = "sync.local.from_docs";

pub const LOCAL_SYNC_EVENT_STARTED: &str = "sync.local.started";
pub const LOCAL_SYNC_EVENT_APPLIED: &str = "sync.local.applied";
pub const LOCAL_SYNC_EVENT_FAILED: &str = "sync.local.failed";

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LocalSyncFromDocsCommand {
    pub repo_root: String,
    pub epic_slug: String,
    #[serde(default)]
    pub create_branches: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LocalSyncStats {
    pub epics_created: u64,
    pub epics_updated: u64,
    pub tasks_created: u64,
    pub tasks_updated: u64,
    pub parent_links_updated: u64,
}
