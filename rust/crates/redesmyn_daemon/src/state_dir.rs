use std::path::{Path, PathBuf};

const STATE_DIR_NAME: &str = ".redesmyn";
const FALLBACK_STATE_DIR_NAME: &str = ".redesmyn_state";

/// Resolve a writable daemon state directory from the configured repo registry path.
///
/// In normal repo layouts `repo_registry_dir` is `<state_dir>/repos`, so we use its parent.
/// In linked worktree layouts, `<worktree>/.redesmyn` can be a file; in that case we fall back
/// to the nearest ancestor directory named `.redesmyn`.
pub(crate) fn daemon_state_dir_from_repo_registry_dir(repo_registry_dir: &Path) -> PathBuf {
    let direct = repo_registry_dir.parent().unwrap_or(repo_registry_dir);
    if !direct.exists() || direct.is_dir() {
        return direct.to_path_buf();
    }

    for ancestor in repo_registry_dir.ancestors() {
        if ancestor
            .file_name()
            .is_some_and(|name| name == STATE_DIR_NAME)
            && ancestor.is_dir()
        {
            return ancestor.to_path_buf();
        }
    }

    direct
        .parent()
        .map(|parent| parent.join(FALLBACK_STATE_DIR_NAME))
        .unwrap_or_else(|| PathBuf::from(FALLBACK_STATE_DIR_NAME))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::daemon_state_dir_from_repo_registry_dir;

    #[test]
    fn returns_direct_parent_when_parent_is_missing_or_directory() {
        let tmp = tempdir().expect("tempdir");
        let repo_registry_dir = tmp.path().join(".redesmyn").join("repos");

        let resolved = daemon_state_dir_from_repo_registry_dir(&repo_registry_dir);

        assert_eq!(resolved, tmp.path().join(".redesmyn"));
    }

    #[test]
    fn falls_back_to_ancestor_redesmyn_dir_for_worktree_layout() {
        let tmp = tempdir().expect("tempdir");
        let workspace_state_dir = tmp.path().join(".redesmyn");
        fs::create_dir_all(workspace_state_dir.join("worktrees/rn/task")).expect("state dir");

        let worktree_root = workspace_state_dir.join("worktrees/rn/task");
        fs::write(worktree_root.join(".redesmyn"), "harness config").expect("worktree .redesmyn");

        let repo_registry_dir = worktree_root.join(".redesmyn").join("repos");
        let resolved = daemon_state_dir_from_repo_registry_dir(&repo_registry_dir);

        assert_eq!(resolved, workspace_state_dir);
    }

    #[test]
    fn falls_back_to_sibling_state_dir_when_no_ancestor_redesmyn_dir_exists() {
        let tmp = tempdir().expect("tempdir");
        fs::write(tmp.path().join(".redesmyn"), "local config").expect("local .redesmyn");

        let repo_registry_dir = tmp.path().join(".redesmyn").join("repos");
        let resolved = daemon_state_dir_from_repo_registry_dir(&repo_registry_dir);

        assert_eq!(resolved, tmp.path().join(".redesmyn_state"));
    }
}
