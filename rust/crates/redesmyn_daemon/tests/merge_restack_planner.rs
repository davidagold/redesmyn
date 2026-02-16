use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use redesmyn_daemon::{
    MergePlanRequest, MergeRestackCommandKind, MergeRestackPlanner, PlanBlockerCode, PlanError,
    PlanPolicyInput, PlanScope, PlannerContext, PlannerTaskGraph, PlannerTaskNode,
    RepoInstanceStatus, RepoPrimaryStatus, RestackPlanRequest, WorktreeHealth,
};
use redesmyn_git::GitCliBackend;
use redesmyn_ids::{EpicId, HostInstanceId, RepoId, TaskId, WorkspaceId};
use redesmyn_protocol::RepoScope;
use tempfile::TempDir;

fn git(repo_root: &Path, args: &[&str]) {
    let output = Command::new("git")
        .arg("--no-pager")
        .arg("-c")
        .arg("color.ui=false")
        .args(args)
        .current_dir(repo_root)
        .output()
        .expect("git subprocess should run");

    if !output.status.success() {
        panic!(
            "git {:?} failed: status={:?} stdout={} stderr={}",
            args,
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}

#[derive(Clone)]
struct Fixture {
    repo_root: PathBuf,
    _tmp: Arc<TempDir>,
    repo_scope: RepoScope,
    epic_id: EpicId,
    task_a: TaskId,
    task_b: TaskId,
    task_c: TaskId,
    task_d: TaskId,
    branch_a: String,
    branch_b: String,
    branch_c: String,
    branch_d: String,
    worktrees_by_branch: BTreeMap<String, PathBuf>,
}

impl Fixture {
    fn graph(&self) -> PlannerTaskGraph {
        PlannerTaskGraph {
            epic_id: self.epic_id,
            root_branch: "main".to_string(),
            tasks: vec![
                PlannerTaskNode {
                    task_id: self.task_a,
                    parent_task_id: None,
                    branch_name: Some(self.branch_a.clone()),
                    merge_ready: true,
                },
                PlannerTaskNode {
                    task_id: self.task_b,
                    parent_task_id: Some(self.task_a),
                    branch_name: Some(self.branch_b.clone()),
                    merge_ready: true,
                },
                PlannerTaskNode {
                    task_id: self.task_c,
                    parent_task_id: Some(self.task_b),
                    branch_name: Some(self.branch_c.clone()),
                    merge_ready: true,
                },
                PlannerTaskNode {
                    task_id: self.task_d,
                    parent_task_id: Some(self.task_b),
                    branch_name: Some(self.branch_d.clone()),
                    merge_ready: true,
                },
            ],
        }
    }

    fn worktree_health(&self) -> Vec<WorktreeHealth> {
        self.worktrees_by_branch
            .iter()
            .map(|(branch, path)| WorktreeHealth {
                branch_name: branch.clone(),
                worktree_path: path.clone(),
                checked_out_branch: Some(branch.clone()),
                dirty: false,
                git_operation_in_progress: false,
            })
            .collect()
    }

    fn context(&self, target_task_id: TaskId, scope: PlanScope) -> PlannerContext {
        PlannerContext {
            repo_scope: self.repo_scope,
            repo_root: self.repo_root.clone(),
            target_task_id,
            scope,
            graph: self.graph(),
            repo_instance_status: RepoInstanceStatus::Ready,
            repo_primary_status: RepoPrimaryStatus::Primary,
            worktrees: self.worktree_health(),
        }
    }
}

fn setup_fixture() -> Fixture {
    let tmp = Arc::new(TempDir::new().expect("tempdir"));
    let repo_root = tmp.path().join("repo");
    std::fs::create_dir_all(&repo_root).expect("repo dir");

    git(&repo_root, &["init"]);
    git(&repo_root, &["config", "user.email", "test@example.com"]);
    git(&repo_root, &["config", "user.name", "Redesmyn Test"]);
    git(&repo_root, &["checkout", "-b", "main"]);

    std::fs::write(repo_root.join("base.txt"), "base\n").expect("write base");
    git(&repo_root, &["add", "base.txt"]);
    git(&repo_root, &["commit", "-m", "base"]);

    let branch_a = "rn/test/task-a".to_string();
    let branch_b = "rn/test/task-b".to_string();
    let branch_c = "rn/test/task-c".to_string();
    let branch_d = "rn/test/task-d".to_string();

    git(&repo_root, &["checkout", "-b", &branch_a]);
    std::fs::write(repo_root.join("a.txt"), "a\n").expect("write a");
    git(&repo_root, &["add", "a.txt"]);
    git(&repo_root, &["commit", "-m", "a"]);

    git(&repo_root, &["checkout", "-b", &branch_b]);
    std::fs::write(repo_root.join("b.txt"), "b\n").expect("write b");
    git(&repo_root, &["add", "b.txt"]);
    git(&repo_root, &["commit", "-m", "b"]);

    git(&repo_root, &["checkout", "-b", &branch_c]);
    std::fs::write(repo_root.join("c.txt"), "c\n").expect("write c");
    git(&repo_root, &["add", "c.txt"]);
    git(&repo_root, &["commit", "-m", "c"]);

    git(&repo_root, &["checkout", &branch_b]);
    git(&repo_root, &["checkout", "-b", &branch_d]);
    std::fs::write(repo_root.join("d.txt"), "d\n").expect("write d");
    git(&repo_root, &["add", "d.txt"]);
    git(&repo_root, &["commit", "-m", "d"]);

    git(&repo_root, &["checkout", "main"]);

    let mut worktrees_by_branch = BTreeMap::new();
    worktrees_by_branch.insert("main".to_string(), repo_root.clone());
    for branch in [&branch_a, &branch_b, &branch_c, &branch_d] {
        let wt_path = repo_root.join("worktrees").join(branch.replace('/', "_"));
        if let Some(parent) = wt_path.parent() {
            std::fs::create_dir_all(parent).expect("worktree parent");
        }
        let wt_path_str = wt_path.display().to_string();
        git(&repo_root, &["worktree", "add", &wt_path_str, branch]);
        worktrees_by_branch.insert(branch.clone(), wt_path);
    }

    Fixture {
        repo_root,
        _tmp: tmp,
        repo_scope: RepoScope {
            workspace_id: WorkspaceId::new(),
            repo_id: RepoId::new(),
        },
        epic_id: EpicId::new(),
        task_a: TaskId::new(),
        task_b: TaskId::new(),
        task_c: TaskId::new(),
        task_d: TaskId::new(),
        branch_a,
        branch_b,
        branch_c,
        branch_d,
        worktrees_by_branch,
    }
}

fn extract_blockers(err: PlanError) -> Vec<redesmyn_daemon::PlanBlocker> {
    match err {
        PlanError::Blocked(_, blocked) => blocked.blockers,
        other => panic!("expected blocked planner error, got {other}"),
    }
}

#[tokio::test]
async fn merge_plan_is_deterministic_for_descendants_scope() {
    let fixture = setup_fixture();
    let planner = MergeRestackPlanner::new(Arc::new(GitCliBackend::new()));

    let request = MergePlanRequest {
        context: fixture.context(fixture.task_b, PlanScope::Descendants),
        policy: PlanPolicyInput {
            command_kind: MergeRestackCommandKind::Merge,
            requires_repo_primary: None,
        },
    };

    let first = planner
        .build_merge_plan(request.clone())
        .await
        .expect("first merge plan");
    let second = planner
        .build_merge_plan(request)
        .await
        .expect("second merge plan");

    assert_eq!(first, second);
    assert!(!first.policy.requires_repo_primary);

    let kinds_and_branches: Vec<_> = first
        .steps
        .iter()
        .map(|step| (step.kind, step.task_branch.clone()))
        .collect();
    assert_eq!(
        kinds_and_branches,
        vec![
            (
                redesmyn_daemon::PlanStepKind::Rebase,
                fixture.branch_a.clone()
            ),
            (
                redesmyn_daemon::PlanStepKind::Rebase,
                fixture.branch_b.clone()
            ),
            (
                redesmyn_daemon::PlanStepKind::Rebase,
                fixture.branch_c.clone()
            ),
            (
                redesmyn_daemon::PlanStepKind::Rebase,
                fixture.branch_d.clone()
            ),
            (
                redesmyn_daemon::PlanStepKind::MergeFf,
                fixture.branch_a.clone()
            ),
            (
                redesmyn_daemon::PlanStepKind::MergeFf,
                fixture.branch_b.clone()
            ),
        ]
    );
    assert_eq!(
        first
            .steps
            .iter()
            .map(|step| step.step_index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4, 5]
    );
}

#[tokio::test]
async fn merge_plan_reports_missing_worktree_dirty_and_in_progress_blockers() {
    let fixture = setup_fixture();
    let planner = MergeRestackPlanner::new(Arc::new(GitCliBackend::new()));

    let mut context = fixture.context(fixture.task_b, PlanScope::Descendants);
    context
        .worktrees
        .retain(|w| w.branch_name != fixture.branch_c.as_str());
    if let Some(branch_b_worktree) = context
        .worktrees
        .iter_mut()
        .find(|w| w.branch_name == fixture.branch_b)
    {
        branch_b_worktree.dirty = true;
        branch_b_worktree.git_operation_in_progress = true;
    }

    let err = planner
        .build_merge_plan(MergePlanRequest {
            context,
            policy: PlanPolicyInput {
                command_kind: MergeRestackCommandKind::Merge,
                requires_repo_primary: None,
            },
        })
        .await
        .expect_err("expected structured blockers");
    let blockers = extract_blockers(err);

    assert!(blockers.iter().any(|b| {
        b.code == PlanBlockerCode::WorktreeMissing
            && b.branch_name.as_deref() == Some(fixture.branch_c.as_str())
    }));
    assert!(blockers.iter().any(|b| {
        b.code == PlanBlockerCode::DirtyWorktree
            && b.branch_name.as_deref() == Some(fixture.branch_b.as_str())
    }));
    assert!(blockers.iter().any(|b| {
        b.code == PlanBlockerCode::GitOperationInProgress
            && b.branch_name.as_deref() == Some(fixture.branch_b.as_str())
    }));
}

#[tokio::test]
async fn merge_plan_reports_merge_not_ready_spine_blocker() {
    let fixture = setup_fixture();
    let planner = MergeRestackPlanner::new(Arc::new(GitCliBackend::new()));

    let mut context = fixture.context(fixture.task_b, PlanScope::Spine);
    for task in &mut context.graph.tasks {
        if task.task_id == fixture.task_b {
            task.merge_ready = false;
        }
    }

    let err = planner
        .build_merge_plan(MergePlanRequest {
            context,
            policy: PlanPolicyInput {
                command_kind: MergeRestackCommandKind::Merge,
                requires_repo_primary: None,
            },
        })
        .await
        .expect_err("expected merge readiness blocker");
    let blockers = extract_blockers(err);
    assert!(blockers.iter().any(|b| {
        b.code == PlanBlockerCode::MergeNotReady && b.task_id == Some(fixture.task_b)
    }));
}

#[tokio::test]
async fn merge_plan_reports_repo_instance_busy() {
    let fixture = setup_fixture();
    let planner = MergeRestackPlanner::new(Arc::new(GitCliBackend::new()));

    let mut context = fixture.context(fixture.task_b, PlanScope::Spine);
    context.repo_instance_status = RepoInstanceStatus::Busy {
        lock_path: fixture.repo_root.join(".redesmyn/repo.lock"),
        owner_host_instance_id: Some(HostInstanceId::new()),
    };

    let err = planner
        .build_merge_plan(MergePlanRequest {
            context,
            policy: PlanPolicyInput {
                command_kind: MergeRestackCommandKind::Merge,
                requires_repo_primary: None,
            },
        })
        .await
        .expect_err("expected repo busy blocker");
    let blockers = extract_blockers(err);
    assert!(
        blockers
            .iter()
            .any(|b| b.code == PlanBlockerCode::RepoInstanceBusy)
    );
}

#[tokio::test]
async fn not_primary_blocker_applies_only_when_requires_repo_primary_is_true() {
    let fixture = setup_fixture();
    let planner = MergeRestackPlanner::new(Arc::new(GitCliBackend::new()));

    let mut context = fixture.context(fixture.task_b, PlanScope::Spine);
    context.repo_primary_status = RepoPrimaryStatus::Other {
        host_instance_id: HostInstanceId::new(),
    };

    let plan_without_requirement = planner
        .build_merge_plan(MergePlanRequest {
            context: context.clone(),
            policy: PlanPolicyInput {
                command_kind: MergeRestackCommandKind::Merge,
                requires_repo_primary: None,
            },
        })
        .await
        .expect("should not block when primary is not required");
    assert!(!plan_without_requirement.policy.requires_repo_primary);

    let err = planner
        .build_merge_plan(MergePlanRequest {
            context,
            policy: PlanPolicyInput {
                command_kind: MergeRestackCommandKind::Merge,
                requires_repo_primary: Some(true),
            },
        })
        .await
        .expect_err("expected not primary blocker");
    let blockers = extract_blockers(err);
    assert!(
        blockers
            .iter()
            .any(|b| b.code == PlanBlockerCode::NotPrimaryExecutor)
    );
}

#[tokio::test]
async fn restack_plan_spine_scope_is_deterministic() {
    let fixture = setup_fixture();
    let planner = MergeRestackPlanner::new(Arc::new(GitCliBackend::new()));

    let plan = planner
        .build_restack_plan(RestackPlanRequest {
            context: fixture.context(fixture.task_b, PlanScope::Spine),
            policy: PlanPolicyInput {
                command_kind: MergeRestackCommandKind::Restack,
                requires_repo_primary: None,
            },
        })
        .await
        .expect("restack plan");

    assert_eq!(plan.affected_task_ids.len(), 2);
    assert_eq!(
        plan.steps
            .iter()
            .map(|step| (step.kind, step.task_branch.as_str()))
            .collect::<Vec<_>>(),
        vec![
            (
                redesmyn_daemon::PlanStepKind::Rebase,
                fixture.branch_a.as_str()
            ),
            (
                redesmyn_daemon::PlanStepKind::Rebase,
                fixture.branch_b.as_str()
            ),
        ]
    );
}
