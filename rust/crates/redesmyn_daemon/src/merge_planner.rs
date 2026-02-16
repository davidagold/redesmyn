use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;

use redesmyn_git::{GitBackend, GitRevision, GitRunOptions};
use redesmyn_ids::{EpicId, HostInstanceId, TaskId};
use redesmyn_logging::tracing;
use redesmyn_protocol::{ErrorDetail, RepoScope};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MergeRestackCommandKind {
    Merge,
    Restack,
}

impl MergeRestackCommandKind {
    #[must_use]
    pub const fn default_requires_repo_primary(self) -> bool {
        match self {
            Self::Merge | Self::Restack => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PlanPolicyInput {
    pub command_kind: MergeRestackCommandKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires_repo_primary: Option<bool>,
}

impl PlanPolicyInput {
    #[must_use]
    pub fn resolved_requires_repo_primary(&self) -> bool {
        self.requires_repo_primary
            .unwrap_or(self.command_kind.default_requires_repo_primary())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PlanPolicyMetadata {
    pub requires_repo_primary: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanScope {
    /// Only the root→target spine.
    Spine,
    /// Spine plus all descendants of every spine node.
    Descendants,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PlannerTaskNode {
    pub task_id: TaskId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_task_id: Option<TaskId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch_name: Option<String>,
    #[serde(default)]
    pub merge_ready: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PlannerTaskGraph {
    pub epic_id: EpicId,
    pub root_branch: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tasks: Vec<PlannerTaskNode>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WorktreeHealth {
    pub branch_name: String,
    pub worktree_path: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checked_out_branch: Option<String>,
    #[serde(default)]
    pub dirty: bool,
    #[serde(default)]
    pub git_operation_in_progress: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum RepoInstanceStatus {
    #[default]
    Ready,
    Busy {
        lock_path: PathBuf,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        owner_host_instance_id: Option<HostInstanceId>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum RepoPrimaryStatus {
    #[default]
    Unknown,
    Primary,
    Other {
        host_instance_id: HostInstanceId,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PlannerContext {
    pub repo_scope: RepoScope,
    pub repo_root: PathBuf,
    pub target_task_id: TaskId,
    pub scope: PlanScope,
    pub graph: PlannerTaskGraph,
    #[serde(default)]
    pub repo_instance_status: RepoInstanceStatus,
    #[serde(default)]
    pub repo_primary_status: RepoPrimaryStatus,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub worktrees: Vec<WorktreeHealth>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MergePlanRequest {
    pub context: PlannerContext,
    pub policy: PlanPolicyInput,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RestackPlanRequest {
    pub context: PlannerContext,
    pub policy: PlanPolicyInput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanStepKind {
    Rebase,
    MergeFf,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PlanStep {
    pub step_index: usize,
    pub kind: PlanStepKind,
    pub task_id: TaskId,
    pub task_branch: String,
    pub target_worktree_path: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upstream_ref: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_ref: Option<String>,
    /// Stable textual reason explaining step placement in the deterministic order.
    pub order_reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MergePlan {
    pub repo_scope: RepoScope,
    pub epic_id: EpicId,
    pub root_branch: String,
    pub scope: PlanScope,
    pub target_task_id: TaskId,
    pub policy: PlanPolicyMetadata,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub spine_task_ids: Vec<TaskId>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub affected_task_ids: Vec<TaskId>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub steps: Vec<PlanStep>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RestackPlan {
    pub repo_scope: RepoScope,
    pub epic_id: EpicId,
    pub root_branch: String,
    pub scope: PlanScope,
    pub target_task_id: TaskId,
    pub policy: PlanPolicyMetadata,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub spine_task_ids: Vec<TaskId>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub affected_task_ids: Vec<TaskId>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub steps: Vec<PlanStep>,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum PlanBlockerCode {
    BranchMissing,
    MergeNotReady,
    WorktreeMissing,
    WorktreeBranchMismatch,
    DirtyWorktree,
    GitOperationInProgress,
    RepoInstanceBusy,
    NotPrimaryExecutor,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PlanBlocker {
    pub code: PlanBlockerCode,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<TaskId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worktree_path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<ErrorDetail>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PlanBlockedError {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blockers: Vec<PlanBlocker>,
}

#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    #[error("invalid planner input: {0}")]
    InvalidInput(String),
    #[error("planner precondition query failed: {0}")]
    Git(#[from] redesmyn_git::GitError),
    #[error("planning blocked by {0} blocker(s)")]
    Blocked(usize, PlanBlockedError),
}

impl PlanError {
    fn blocked(mut blockers: Vec<PlanBlocker>) -> Self {
        blockers.sort_by(|a, b| {
            (
                a.code,
                a.task_id.map(|id| id.to_string()),
                a.branch_name.clone(),
                a.worktree_path
                    .as_ref()
                    .map(|path| path.display().to_string()),
            )
                .cmp(&(
                    b.code,
                    b.task_id.map(|id| id.to_string()),
                    b.branch_name.clone(),
                    b.worktree_path
                        .as_ref()
                        .map(|path| path.display().to_string()),
                ))
        });
        let count = blockers.len();
        Self::Blocked(count, PlanBlockedError { blockers })
    }
}

#[derive(Clone)]
pub struct MergeRestackPlanner {
    git_backend: Arc<dyn GitBackend>,
}

impl MergeRestackPlanner {
    #[must_use]
    pub fn new(git_backend: Arc<dyn GitBackend>) -> Self {
        Self { git_backend }
    }

    pub async fn build_merge_plan(
        &self,
        request: MergePlanRequest,
    ) -> Result<MergePlan, PlanError> {
        if request.policy.command_kind != MergeRestackCommandKind::Merge {
            return Err(PlanError::InvalidInput(
                "merge planner requires command_kind=merge".to_string(),
            ));
        }
        self.plan_merge(request).await
    }

    pub async fn build_restack_plan(
        &self,
        request: RestackPlanRequest,
    ) -> Result<RestackPlan, PlanError> {
        if request.policy.command_kind != MergeRestackCommandKind::Restack {
            return Err(PlanError::InvalidInput(
                "restack planner requires command_kind=restack".to_string(),
            ));
        }
        self.plan_restack(request).await
    }

    async fn plan_merge(&self, request: MergePlanRequest) -> Result<MergePlan, PlanError> {
        let span = redesmyn_logging::redesmyn_info_span!("daemon.merge_planner.merge_plan");
        redesmyn_logging::span::record_workspace_id(&span, request.context.repo_scope.workspace_id);
        redesmyn_logging::span::record_repo_id(&span, request.context.repo_scope.repo_id);
        redesmyn_logging::span::record_epic_id(&span, request.context.graph.epic_id);
        redesmyn_logging::span::record_task_id(&span, request.context.target_task_id);
        let _enter = span.enter();

        let topology = TaskTopology::from_graph(&request.context.graph)?;
        let scope_task_ids =
            topology.scope_task_ids(request.context.target_task_id, request.context.scope)?;
        let ordered_task_ids = topology.ordered_task_ids(&scope_task_ids)?;
        let spine_task_ids = topology.spine_task_ids(request.context.target_task_id)?;
        let worktrees_by_branch = build_worktree_index(&request.context.worktrees)?;
        let requires_repo_primary = request.policy.resolved_requires_repo_primary();

        let blockers = self
            .collect_blockers(
                &request.context,
                &topology,
                &ordered_task_ids,
                &spine_task_ids,
                &worktrees_by_branch,
                requires_repo_primary,
                true,
                true,
            )
            .await?;
        if !blockers.is_empty() {
            tracing::warn!(blocker_count = blockers.len(), "merge planning blocked");
            return Err(PlanError::blocked(blockers));
        }

        let mut steps = Vec::new();
        for task_id in &ordered_task_ids {
            let node = topology.task(*task_id)?;
            let task_branch = node.branch_name.clone().ok_or_else(|| {
                PlanError::InvalidInput("task branch unexpectedly missing".to_string())
            })?;
            let worktree = worktrees_by_branch.get(&task_branch).ok_or_else(|| {
                PlanError::InvalidInput(
                    "worktree unexpectedly missing after validation".to_string(),
                )
            })?;
            let upstream_ref =
                topology.upstream_ref(*task_id, &request.context.graph.root_branch)?;
            steps.push(PlanStep {
                step_index: steps.len(),
                kind: PlanStepKind::Rebase,
                task_id: *task_id,
                task_branch,
                target_worktree_path: worktree.worktree_path.clone(),
                upstream_ref: Some(upstream_ref),
                base_ref: None,
                order_reason: "topological_depth_then_branch".to_string(),
            });
        }

        let base_worktree = worktrees_by_branch
            .get(&request.context.graph.root_branch)
            .ok_or_else(|| {
                PlanError::InvalidInput(
                    "base worktree unexpectedly missing after validation".to_string(),
                )
            })?
            .worktree_path
            .clone();

        for task_id in &spine_task_ids {
            let node = topology.task(*task_id)?;
            let task_branch = node.branch_name.clone().ok_or_else(|| {
                PlanError::InvalidInput("task branch unexpectedly missing".to_string())
            })?;
            steps.push(PlanStep {
                step_index: steps.len(),
                kind: PlanStepKind::MergeFf,
                task_id: *task_id,
                task_branch,
                target_worktree_path: base_worktree.clone(),
                upstream_ref: None,
                base_ref: Some(request.context.graph.root_branch.clone()),
                order_reason: "spine_root_to_leaf".to_string(),
            });
        }

        tracing::info!(
            scope = ?request.context.scope,
            affected_task_count = ordered_task_ids.len(),
            step_count = steps.len(),
            requires_repo_primary,
            "merge plan built"
        );

        Ok(MergePlan {
            repo_scope: request.context.repo_scope,
            epic_id: request.context.graph.epic_id,
            root_branch: request.context.graph.root_branch.clone(),
            scope: request.context.scope,
            target_task_id: request.context.target_task_id,
            policy: PlanPolicyMetadata {
                requires_repo_primary,
            },
            spine_task_ids,
            affected_task_ids: ordered_task_ids,
            steps,
        })
    }

    async fn plan_restack(&self, request: RestackPlanRequest) -> Result<RestackPlan, PlanError> {
        let span = redesmyn_logging::redesmyn_info_span!("daemon.merge_planner.restack_plan");
        redesmyn_logging::span::record_workspace_id(&span, request.context.repo_scope.workspace_id);
        redesmyn_logging::span::record_repo_id(&span, request.context.repo_scope.repo_id);
        redesmyn_logging::span::record_epic_id(&span, request.context.graph.epic_id);
        redesmyn_logging::span::record_task_id(&span, request.context.target_task_id);
        let _enter = span.enter();

        let topology = TaskTopology::from_graph(&request.context.graph)?;
        let scope_task_ids =
            topology.scope_task_ids(request.context.target_task_id, request.context.scope)?;
        let ordered_task_ids = topology.ordered_task_ids(&scope_task_ids)?;
        let spine_task_ids = topology.spine_task_ids(request.context.target_task_id)?;
        let worktrees_by_branch = build_worktree_index(&request.context.worktrees)?;
        let requires_repo_primary = request.policy.resolved_requires_repo_primary();

        let blockers = self
            .collect_blockers(
                &request.context,
                &topology,
                &ordered_task_ids,
                &spine_task_ids,
                &worktrees_by_branch,
                requires_repo_primary,
                false,
                false,
            )
            .await?;
        if !blockers.is_empty() {
            tracing::warn!(blocker_count = blockers.len(), "restack planning blocked");
            return Err(PlanError::blocked(blockers));
        }

        let mut steps = Vec::new();
        for task_id in &ordered_task_ids {
            let node = topology.task(*task_id)?;
            let task_branch = node.branch_name.clone().ok_or_else(|| {
                PlanError::InvalidInput("task branch unexpectedly missing".to_string())
            })?;
            let worktree = worktrees_by_branch.get(&task_branch).ok_or_else(|| {
                PlanError::InvalidInput(
                    "worktree unexpectedly missing after validation".to_string(),
                )
            })?;
            let upstream_ref =
                topology.upstream_ref(*task_id, &request.context.graph.root_branch)?;
            steps.push(PlanStep {
                step_index: steps.len(),
                kind: PlanStepKind::Rebase,
                task_id: *task_id,
                task_branch,
                target_worktree_path: worktree.worktree_path.clone(),
                upstream_ref: Some(upstream_ref),
                base_ref: None,
                order_reason: "topological_depth_then_branch".to_string(),
            });
        }

        tracing::info!(
            scope = ?request.context.scope,
            affected_task_count = ordered_task_ids.len(),
            step_count = steps.len(),
            requires_repo_primary,
            "restack plan built"
        );

        Ok(RestackPlan {
            repo_scope: request.context.repo_scope,
            epic_id: request.context.graph.epic_id,
            root_branch: request.context.graph.root_branch.clone(),
            scope: request.context.scope,
            target_task_id: request.context.target_task_id,
            policy: PlanPolicyMetadata {
                requires_repo_primary,
            },
            spine_task_ids,
            affected_task_ids: ordered_task_ids,
            steps,
        })
    }

    async fn collect_blockers(
        &self,
        context: &PlannerContext,
        topology: &TaskTopology,
        ordered_task_ids: &[TaskId],
        spine_task_ids: &[TaskId],
        worktrees_by_branch: &BTreeMap<String, WorktreeHealth>,
        requires_repo_primary: bool,
        require_base_worktree: bool,
        enforce_merge_ready: bool,
    ) -> Result<Vec<PlanBlocker>, PlanError> {
        let mut blockers = Vec::new();

        if let RepoInstanceStatus::Busy {
            lock_path,
            owner_host_instance_id,
        } = &context.repo_instance_status
        {
            let mut detail = ErrorDetail::new();
            detail.insert("lock_path".to_string(), lock_path.display().to_string());
            if let Some(owner) = owner_host_instance_id {
                detail.insert("owner_host_instance_id".to_string(), owner.to_string());
            }
            blockers.push(PlanBlocker {
                code: PlanBlockerCode::RepoInstanceBusy,
                message: "Repo instance is busy (attach lock held by another daemon).".to_string(),
                task_id: None,
                branch_name: None,
                worktree_path: Some(lock_path.clone()),
                detail: Some(detail),
            });
        }

        if requires_repo_primary {
            match &context.repo_primary_status {
                RepoPrimaryStatus::Primary => {}
                RepoPrimaryStatus::Other { host_instance_id } => {
                    blockers.push(PlanBlocker {
                        code: PlanBlockerCode::NotPrimaryExecutor,
                        message: "Not primary executor for repo scope.".to_string(),
                        task_id: None,
                        branch_name: None,
                        worktree_path: None,
                        detail: Some(ErrorDetail::from([(
                            "primary_host_instance_id".to_string(),
                            host_instance_id.to_string(),
                        )])),
                    });
                }
                RepoPrimaryStatus::Unknown => {
                    blockers.push(PlanBlocker {
                        code: PlanBlockerCode::NotPrimaryExecutor,
                        message: "Primary executor is unknown for repo scope.".to_string(),
                        task_id: None,
                        branch_name: None,
                        worktree_path: None,
                        detail: None,
                    });
                }
            }
        }

        if enforce_merge_ready {
            for task_id in spine_task_ids {
                let task = topology.task(*task_id)?;
                if task.merge_ready {
                    continue;
                }
                blockers.push(PlanBlocker {
                    code: PlanBlockerCode::MergeNotReady,
                    message: "Task on merge spine is not merge-ready.".to_string(),
                    task_id: Some(*task_id),
                    branch_name: task.branch_name.clone(),
                    worktree_path: None,
                    detail: None,
                });
            }
        }

        let mut refs_to_check = BTreeSet::new();
        refs_to_check.insert(context.graph.root_branch.clone());
        for task_id in ordered_task_ids {
            let task = topology.task(*task_id)?;
            if let Some(branch) = &task.branch_name {
                refs_to_check.insert(branch.clone());
            }
        }
        for task_id in spine_task_ids {
            let task = topology.task(*task_id)?;
            if let Some(branch) = &task.branch_name {
                refs_to_check.insert(branch.clone());
            }
        }

        let revs = refs_to_check
            .iter()
            .map(|ref_name| GitRevision::new(ref_name.clone()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| {
                PlanError::InvalidInput(format!("invalid git ref in plan input: {err}"))
            })?;

        let resolved = self
            .git_backend
            .resolve_commits(&context.repo_root, &revs, GitRunOptions::default())
            .await?;
        let mut resolved_by_ref = BTreeMap::new();
        for (rev, oid) in refs_to_check.iter().zip(resolved.into_iter()) {
            resolved_by_ref.insert(rev.clone(), oid.is_some());
        }
        let root_branch_exists = resolved_by_ref
            .get(&context.graph.root_branch)
            .copied()
            .unwrap_or(false);
        if !root_branch_exists {
            blockers.push(PlanBlocker {
                code: PlanBlockerCode::BranchMissing,
                message: "Base branch does not exist in the repo.".to_string(),
                task_id: None,
                branch_name: Some(context.graph.root_branch.clone()),
                worktree_path: None,
                detail: None,
            });
        }

        for task_id in ordered_task_ids {
            let task = topology.task(*task_id)?;
            let Some(branch_name) = &task.branch_name else {
                blockers.push(PlanBlocker {
                    code: PlanBlockerCode::BranchMissing,
                    message: "Task branch is not configured.".to_string(),
                    task_id: Some(*task_id),
                    branch_name: None,
                    worktree_path: None,
                    detail: None,
                });
                continue;
            };

            if !resolved_by_ref.get(branch_name).copied().unwrap_or(false) {
                blockers.push(PlanBlocker {
                    code: PlanBlockerCode::BranchMissing,
                    message: "Task branch does not exist in the repo.".to_string(),
                    task_id: Some(*task_id),
                    branch_name: Some(branch_name.clone()),
                    worktree_path: None,
                    detail: None,
                });
                continue;
            }

            let Some(health) = worktrees_by_branch.get(branch_name) else {
                blockers.push(PlanBlocker {
                    code: PlanBlockerCode::WorktreeMissing,
                    message: "Task worktree is missing.".to_string(),
                    task_id: Some(*task_id),
                    branch_name: Some(branch_name.clone()),
                    worktree_path: None,
                    detail: None,
                });
                continue;
            };

            if let Some(checked_out_branch) = &health.checked_out_branch
                && checked_out_branch != branch_name
            {
                blockers.push(PlanBlocker {
                    code: PlanBlockerCode::WorktreeBranchMismatch,
                    message: "Worktree is not checked out on the expected branch.".to_string(),
                    task_id: Some(*task_id),
                    branch_name: Some(branch_name.clone()),
                    worktree_path: Some(health.worktree_path.clone()),
                    detail: Some(ErrorDetail::from([
                        ("expected_branch".to_string(), branch_name.clone()),
                        ("actual_branch".to_string(), checked_out_branch.clone()),
                    ])),
                });
            }

            if health.dirty {
                blockers.push(PlanBlocker {
                    code: PlanBlockerCode::DirtyWorktree,
                    message: "Worktree has uncommitted changes.".to_string(),
                    task_id: Some(*task_id),
                    branch_name: Some(branch_name.clone()),
                    worktree_path: Some(health.worktree_path.clone()),
                    detail: None,
                });
            }

            if health.git_operation_in_progress {
                blockers.push(PlanBlocker {
                    code: PlanBlockerCode::GitOperationInProgress,
                    message: "Worktree has an in-progress git operation.".to_string(),
                    task_id: Some(*task_id),
                    branch_name: Some(branch_name.clone()),
                    worktree_path: Some(health.worktree_path.clone()),
                    detail: None,
                });
            }
        }

        if require_base_worktree {
            let Some(base_worktree) = worktrees_by_branch.get(&context.graph.root_branch) else {
                blockers.push(PlanBlocker {
                    code: PlanBlockerCode::WorktreeMissing,
                    message: "Base branch worktree is missing.".to_string(),
                    task_id: None,
                    branch_name: Some(context.graph.root_branch.clone()),
                    worktree_path: None,
                    detail: None,
                });
                return Ok(blockers);
            };

            if base_worktree.dirty {
                blockers.push(PlanBlocker {
                    code: PlanBlockerCode::DirtyWorktree,
                    message: "Base branch worktree has uncommitted changes.".to_string(),
                    task_id: None,
                    branch_name: Some(context.graph.root_branch.clone()),
                    worktree_path: Some(base_worktree.worktree_path.clone()),
                    detail: None,
                });
            }
            if base_worktree.git_operation_in_progress {
                blockers.push(PlanBlocker {
                    code: PlanBlockerCode::GitOperationInProgress,
                    message: "Base branch worktree has an in-progress git operation.".to_string(),
                    task_id: None,
                    branch_name: Some(context.graph.root_branch.clone()),
                    worktree_path: Some(base_worktree.worktree_path.clone()),
                    detail: None,
                });
            }
        }

        Ok(blockers)
    }
}

fn build_worktree_index(
    worktrees: &[WorktreeHealth],
) -> Result<BTreeMap<String, WorktreeHealth>, PlanError> {
    let mut out = BTreeMap::new();
    for worktree in worktrees {
        if out
            .insert(worktree.branch_name.clone(), worktree.clone())
            .is_some()
        {
            return Err(PlanError::InvalidInput(format!(
                "duplicate worktree health entry for branch {}",
                worktree.branch_name
            )));
        }
    }
    Ok(out)
}

struct TaskTopology {
    tasks: BTreeMap<TaskId, PlannerTaskNode>,
    children: BTreeMap<Option<TaskId>, Vec<TaskId>>,
}

impl TaskTopology {
    fn from_graph(graph: &PlannerTaskGraph) -> Result<Self, PlanError> {
        let mut tasks = BTreeMap::new();
        for task in &graph.tasks {
            if tasks.insert(task.task_id, task.clone()).is_some() {
                return Err(PlanError::InvalidInput(format!(
                    "duplicate task id in graph: {}",
                    task.task_id
                )));
            }
        }

        let mut children: BTreeMap<Option<TaskId>, Vec<TaskId>> = BTreeMap::new();
        for task in tasks.values() {
            if let Some(parent_id) = task.parent_task_id
                && !tasks.contains_key(&parent_id)
            {
                return Err(PlanError::InvalidInput(format!(
                    "task {} references missing parent {}",
                    task.task_id, parent_id
                )));
            }
            children
                .entry(task.parent_task_id)
                .or_default()
                .push(task.task_id);
        }
        for child_list in children.values_mut() {
            child_list.sort();
        }
        Ok(Self { tasks, children })
    }

    fn task(&self, task_id: TaskId) -> Result<&PlannerTaskNode, PlanError> {
        self.tasks.get(&task_id).ok_or_else(|| {
            PlanError::InvalidInput(format!(
                "unknown target task id for planner request: {}",
                task_id
            ))
        })
    }

    fn spine_task_ids(&self, target_task_id: TaskId) -> Result<Vec<TaskId>, PlanError> {
        let mut seen = BTreeSet::new();
        let mut spine_rev = Vec::new();
        let mut cursor = Some(target_task_id);
        while let Some(task_id) = cursor {
            if !seen.insert(task_id) {
                return Err(PlanError::InvalidInput(format!(
                    "cycle detected while resolving spine at task {}",
                    task_id
                )));
            }
            let node = self.task(task_id)?;
            spine_rev.push(task_id);
            cursor = node.parent_task_id;
        }
        spine_rev.reverse();
        Ok(spine_rev)
    }

    fn scope_task_ids(
        &self,
        target_task_id: TaskId,
        scope: PlanScope,
    ) -> Result<BTreeSet<TaskId>, PlanError> {
        let spine = self.spine_task_ids(target_task_id)?;
        let mut out: BTreeSet<TaskId> = spine.iter().copied().collect();
        if scope == PlanScope::Spine {
            return Ok(out);
        }

        let mut queue: VecDeque<TaskId> = spine.into_iter().collect();
        while let Some(parent) = queue.pop_front() {
            let Some(children) = self.children.get(&Some(parent)) else {
                continue;
            };
            for child in children {
                if out.insert(*child) {
                    queue.push_back(*child);
                }
            }
        }
        Ok(out)
    }

    fn ordered_task_ids(&self, task_ids: &BTreeSet<TaskId>) -> Result<Vec<TaskId>, PlanError> {
        let mut with_keys = Vec::new();
        for task_id in task_ids {
            let depth = self.depth(*task_id)?;
            let branch_name = self
                .tasks
                .get(task_id)
                .and_then(|task| task.branch_name.as_deref())
                .unwrap_or("");
            with_keys.push((depth, branch_name.to_string(), *task_id));
        }
        with_keys.sort();
        Ok(with_keys.into_iter().map(|(_, _, id)| id).collect())
    }

    fn depth(&self, task_id: TaskId) -> Result<usize, PlanError> {
        let mut depth = 0_usize;
        let mut seen = BTreeSet::new();
        seen.insert(task_id);
        let mut cursor = self
            .tasks
            .get(&task_id)
            .and_then(|task| task.parent_task_id);
        while let Some(parent_id) = cursor {
            if !seen.insert(parent_id) {
                return Err(PlanError::InvalidInput(format!(
                    "cycle detected while computing depth for task {}",
                    task_id
                )));
            }
            depth = depth.saturating_add(1);
            cursor = self
                .tasks
                .get(&parent_id)
                .and_then(|task| task.parent_task_id);
        }
        Ok(depth)
    }

    fn upstream_ref(&self, task_id: TaskId, root_branch: &str) -> Result<String, PlanError> {
        let task = self.task(task_id)?;
        let Some(parent_id) = task.parent_task_id else {
            return Ok(root_branch.to_string());
        };
        let parent = self.task(parent_id)?;
        Ok(parent
            .branch_name
            .clone()
            .unwrap_or_else(|| root_branch.to_string()))
    }
}
