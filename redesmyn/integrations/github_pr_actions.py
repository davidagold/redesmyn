from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from redesmyn.db.models import Epic, Task
from redesmyn.repo import GitCommandError, branch_exists, git_is_ancestor, git_push
from redesmyn.integrations.github_pr import (
    GitHubPullRequestInfo,
    GitHubPullRequestRef,
    create_pull_request,
    detect_pull_request_for_branch,
)
from redesmyn.integrations.github_repo import GithubRepoRef, detect_github_repo_ref
from redesmyn.integrations.linear import LinearClient, refresh_access_token
from redesmyn.integrations.linear_credentials import (
    LinearCredentials,
    default_linear_credential_store,
    is_expiring_soon,
)
from redesmyn.integrations.linear_sync import fetch_issue_url
from redesmyn.settings import load_settings


class GitHubPullRequestActionError(RuntimeError):
    pass


def _github_repo_ref_from_epic(epic: Epic) -> GithubRepoRef | None:
    if (
        epic.github_repo_host is None
        or epic.github_repo_owner is None
        or epic.github_repo_name is None
    ):
        return None
    return GithubRepoRef(
        host=epic.github_repo_host,
        owner=epic.github_repo_owner,
        repo=epic.github_repo_name,
    )


def _github_repo_ref_from_task(task: Task) -> GithubRepoRef | None:
    if (
        task.github_repo_host is None
        or task.github_repo_owner is None
        or task.github_repo_name is None
    ):
        return None
    return GithubRepoRef(
        host=task.github_repo_host,
        owner=task.github_repo_owner,
        repo=task.github_repo_name,
    )


def effective_github_repo_ref_for_task(
    repo_root: Path,
    *,
    epic: Epic,
    task: Task,
) -> GithubRepoRef:
    configured = _github_repo_ref_from_task(task) or _github_repo_ref_from_epic(epic)
    if configured is not None:
        return configured

    detected = detect_github_repo_ref(repo_root)
    if detected is None:
        raise GitHubPullRequestActionError(
            "Could not infer GitHub owner/repo from local git remotes. "
            "Ensure your remote points at github.com (e.g. "
            "`git@github.com:owner/repo.git`)."
        )
    return detected


def effective_upstream_branch_for_task(
    repo_root: Path,
    *,
    task: Task,
    base_branch: str,
    tasks_by_id: dict[int, Task],
) -> str:
    if task.parent_task_id is None:
        return base_branch

    seen: set[int] = set()
    current_parent_id = task.parent_task_id
    while current_parent_id is not None and current_parent_id not in seen:
        seen.add(current_parent_id)
        parent = tasks_by_id.get(current_parent_id)
        if parent is None or parent.branch_name is None:
            return base_branch
        if not branch_exists(repo_root, parent.branch_name):
            return base_branch
        try:
            merged = git_is_ancestor(repo_root, parent.branch_name, base_branch)
        except Exception:
            return base_branch
        if merged is True:
            current_parent_id = parent.parent_task_id
            continue
        return parent.branch_name

    return base_branch


def _is_non_fast_forward_push_error(message: str) -> bool:
    text = message.lower()
    return (
        "non-fast-forward" in text
        or ("fetch first" in text and "rejected" in text)
        or ("would be overwritten by merge" in text and "rejected" in text)
    )


def push_branch_for_github_pr(
    repo_root: Path,
    *,
    branch_name: str,
    allow_force_with_lease: bool,
) -> None:
    try:
        git_push(
            repo_root,
            remote="origin",
            branch_name=branch_name,
            set_upstream=True,
        )
    except GitCommandError as e:
        if _is_non_fast_forward_push_error(str(e)) and not allow_force_with_lease:
            raise GitHubPullRequestActionError(
                f"git push rejected for {branch_name!r} (non-fast-forward). "
                "Push manually with `git push --force-with-lease`, or enable "
                "`rn github config set auto_force_push true`."
            ) from e
        if allow_force_with_lease and _is_non_fast_forward_push_error(str(e)):
            git_push(
                repo_root,
                remote="origin",
                branch_name=branch_name,
                set_upstream=True,
                force_with_lease=True,
            )
            return
        raise GitHubPullRequestActionError(str(e)) from e


async def _maybe_linear_issue_url(
    repo_root: Path, *, linear_issue_id: str | None
) -> str | None:
    if linear_issue_id is None:
        return None

    store = default_linear_credential_store()
    creds = store.get()
    if creds is None:
        return None

    if is_expiring_soon(creds, skew=timedelta(minutes=5)):
        if not creds.refresh_token:
            return None
        try:
            settings = load_settings(repo_root=repo_root)
            token = await refresh_access_token(
                settings, refresh_token=creds.refresh_token
            )
        except Exception:
            return None
        store.set(
            LinearCredentials(
                access_token=token.access_token,
                refresh_token=token.refresh_token or creds.refresh_token,
                token_type=token.token_type,
                scope=token.scope or creds.scope,
                expires_at=token.expires_at,
                connected_at=creds.connected_at,
            )
        )
        creds = store.get() or creds

    try:
        client = LinearClient(access_token=creds.access_token)
        return await fetch_issue_url(client, issue_id=linear_issue_id)
    except Exception:
        return None


def default_github_pr_body(*, linear_issue_url: str | None) -> str | None:
    lines: list[str] = []
    if linear_issue_url is not None:
        lines.append(f"Linear: {linear_issue_url}")
    if not lines:
        return None
    return "\n".join(lines) + "\n"


async def ensure_task_pull_request(
    repo_root: Path,
    *,
    epic: Epic,
    task: Task,
    tasks_by_id: dict[int, Task],
    access_token: str,
    allow_force_with_lease: bool,
) -> GitHubPullRequestInfo:
    if task.branch_name is None:
        raise GitHubPullRequestActionError(
            "Task has no branch backing (branch_name is null). "
            "Run `rn sync --from local --create-branches`."
        )

    repo_ref = effective_github_repo_ref_for_task(repo_root, epic=epic, task=task)
    owner, repo = repo_ref.owner, repo_ref.repo

    base_branch = effective_upstream_branch_for_task(
        repo_root,
        task=task,
        base_branch=epic.root_branch,
        tasks_by_id=tasks_by_id,
    )
    if base_branch == task.branch_name:
        base_branch = epic.root_branch
    if base_branch != epic.root_branch and not branch_exists(repo_root, base_branch):
        base_branch = epic.root_branch

    push_branch_for_github_pr(
        repo_root,
        branch_name=task.branch_name,
        allow_force_with_lease=allow_force_with_lease,
    )

    # Best-effort: ensure stacked bases exist on the remote, so GitHub can accept the PR base.
    if base_branch != epic.root_branch:
        push_branch_for_github_pr(
            repo_root,
            branch_name=base_branch,
            allow_force_with_lease=allow_force_with_lease,
        )

    if task.github_pr_id:
        try:
            ref = GitHubPullRequestRef.parse(task.github_pr_id)
        except ValueError:
            pass
        else:
            return GitHubPullRequestInfo(ref=ref, url=ref.url)

    existing = await detect_pull_request_for_branch(
        owner=owner,
        repo=repo,
        head_branch=task.branch_name,
        access_token=access_token,
    )
    if existing is not None:
        return existing

    linear_url = await _maybe_linear_issue_url(
        repo_root, linear_issue_id=task.linear_issue_id
    )
    created = await create_pull_request(
        owner=owner,
        repo=repo,
        title=task.title,
        body=default_github_pr_body(linear_issue_url=linear_url),
        head_branch=task.branch_name,
        base_branch=base_branch,
        access_token=access_token,
    )
    return created
