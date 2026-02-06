use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use redesmyn_ids::{EpicId, TaskId};
use redesmyn_protocol::RepoScope;
use redesmyn_protocol::sync_commands::{
    LOCAL_SYNC_EVENT_APPLIED, LOCAL_SYNC_EVENT_STARTED, LocalSyncFromDocsCommand, LocalSyncStats,
};
use redesmyn_storage::events::EventScope;
use serde::Serialize;
use serde_yaml::Value as YamlValue;

use crate::event_log::EventLog;

#[derive(Debug, thiserror::Error)]
pub(crate) enum LocalSyncError {
    #[error("invalid sync payload: {0}")]
    InvalidPayload(String),
    #[error("epic doc not found: {0}")]
    EpicDocMissing(String),
    #[error("task doc missing title: {0}")]
    TaskDocMissingTitle(String),
    #[error("task doc missing local task ref (expected rn.id or T-<n> directory): {0}")]
    TaskDocMissingRef(String),
    #[error("unknown parent ref {parent_ref:?} in {path}")]
    UnknownParentRef { parent_ref: String, path: String },
    #[error("epic metadata slug mismatch: expected {expected:?}, found {actual:?}")]
    EpicSlugMismatch { expected: String, actual: String },
    #[error("duplicate local task ref {local_ref:?} in docs")]
    DuplicateTaskRef { local_ref: String },
    #[error("repository scope not found in DB")]
    RepoScopeNotFound,
    #[error("invalid YAML frontmatter in {path}: {message}")]
    InvalidFrontmatter { path: String, message: String },
    #[error("unterminated YAML frontmatter in {path}")]
    UnterminatedFrontmatter { path: String },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Storage(#[from] redesmyn_storage::StorageError),
    #[error(transparent)]
    Sqlx(#[from] sqlx::Error),
}

#[derive(Debug, Clone)]
struct ParsedEpicDoc {
    slug: String,
    title: String,
}

#[derive(Debug, Clone)]
struct ParsedTaskDoc {
    path: PathBuf,
    local_ref: String,
    title: String,
    parent_ref: Option<String>,
    branch_name: Option<String>,
}

#[derive(Debug)]
struct TaskRowState {
    id: TaskId,
    parent_task_id: Option<TaskId>,
}

#[derive(Debug, Serialize)]
struct LocalSyncEventPayload<'a> {
    epic_slug: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stats: Option<&'a LocalSyncStats>,
}

pub(crate) async fn run_local_sync_from_docs(
    pool: &sqlx::SqlitePool,
    event_log: &EventLog,
    scope: RepoScope,
    command: &LocalSyncFromDocsCommand,
) -> Result<LocalSyncStats, LocalSyncError> {
    let repo_root = PathBuf::from(command.repo_root.trim());
    if command.epic_slug.trim().is_empty() {
        return Err(LocalSyncError::InvalidPayload(
            "epic_slug must not be empty".to_string(),
        ));
    }
    if command.repo_root.trim().is_empty() {
        return Err(LocalSyncError::InvalidPayload(
            "repo_root must not be empty".to_string(),
        ));
    }

    let epic_doc = load_epic_doc(&repo_root, &command.epic_slug)?;
    let task_docs = load_task_docs(&repo_root, &epic_doc.slug)?;

    let started_payload = LocalSyncEventPayload {
        epic_slug: &epic_doc.slug,
        message: Some("sync started"),
        stats: None,
    };
    event_log
        .append_event(
            EventScope::Repo {
                workspace_id: scope.workspace_id,
                repo_id: scope.repo_id,
            },
            LOCAL_SYNC_EVENT_STARTED,
            serde_json::to_vec(&started_payload).unwrap_or_default(),
        )
        .await?;

    let stats =
        apply_docs_to_db(pool, scope, &epic_doc, &task_docs, command.create_branches).await?;

    let applied_payload = LocalSyncEventPayload {
        epic_slug: &epic_doc.slug,
        message: Some("sync applied"),
        stats: Some(&stats),
    };
    event_log
        .append_event(
            EventScope::Repo {
                workspace_id: scope.workspace_id,
                repo_id: scope.repo_id,
            },
            LOCAL_SYNC_EVENT_APPLIED,
            serde_json::to_vec(&applied_payload).unwrap_or_default(),
        )
        .await?;

    Ok(stats)
}

async fn apply_docs_to_db(
    pool: &sqlx::SqlitePool,
    scope: RepoScope,
    epic_doc: &ParsedEpicDoc,
    task_docs: &[ParsedTaskDoc],
    create_branches: bool,
) -> Result<LocalSyncStats, LocalSyncError> {
    let task_docs = task_docs.to_vec();
    let epic_slug = epic_doc.slug.clone();
    let epic_title = epic_doc.title.clone();
    let mut tx = pool.begin().await?;

    let repo_exists: Option<u8> = sqlx::query_scalar(
        r#"
        SELECT 1
        FROM repositories
        WHERE id = ?1 AND workspace_id = ?2
        LIMIT 1
        "#,
    )
    .bind(scope.repo_id)
    .bind(scope.workspace_id)
    .fetch_optional(&mut *tx)
    .await?;
    if repo_exists.is_none() {
        return Err(LocalSyncError::RepoScopeNotFound);
    }

    let now_ms = now_unix_ms();

    let mut stats = LocalSyncStats {
        epics_created: 0,
        epics_updated: 0,
        tasks_created: 0,
        tasks_updated: 0,
        parent_links_updated: 0,
    };

    let epic_row: Option<(EpicId, String)> = sqlx::query_as(
        r#"
        SELECT id, title
        FROM epics
        WHERE repo_id = ?1 AND slug = ?2
        LIMIT 1
        "#,
    )
    .bind(scope.repo_id)
    .bind(&epic_slug)
    .fetch_optional(&mut *tx)
    .await?;

    let epic_id = if let Some((epic_id, existing_title)) = epic_row {
        if existing_title != epic_title {
            sqlx::query(
                r#"
                UPDATE epics
                SET title = ?1, updated_at_ms = ?2
                WHERE id = ?3
                "#,
            )
            .bind(&epic_title)
            .bind(now_ms)
            .bind(epic_id)
            .execute(&mut *tx)
            .await?;
            stats.epics_updated += 1;
        }
        epic_id
    } else {
        let epic_id = EpicId::new();
        sqlx::query(
            r#"
            INSERT INTO epics (id, repo_id, created_at_ms, updated_at_ms, slug, title)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            "#,
        )
        .bind(epic_id)
        .bind(scope.repo_id)
        .bind(now_ms)
        .bind(now_ms)
        .bind(&epic_slug)
        .bind(&epic_title)
        .execute(&mut *tx)
        .await?;
        stats.epics_created += 1;
        epic_id
    };

    let mut task_rows: HashMap<String, TaskRowState> = HashMap::with_capacity(task_docs.len());
    let mut created_task_ids: HashSet<TaskId> = HashSet::new();
    let mut updated_task_ids: HashSet<TaskId> = HashSet::new();

    for doc in &task_docs {
        let existing: Option<(TaskId, String, Option<TaskId>, Option<String>)> = sqlx::query_as(
            r#"
            SELECT id, title, parent_task_id, branch_name
            FROM tasks
            WHERE epic_id = ?1 AND local_ref = ?2
            LIMIT 1
            "#,
        )
        .bind(epic_id)
        .bind(&doc.local_ref)
        .fetch_optional(&mut *tx)
        .await?;

        let desired_branch = if create_branches {
            doc.branch_name.clone()
        } else {
            None
        };

        if let Some((task_id, existing_title, parent_task_id, existing_branch)) = existing {
            let mut changed = false;

            if existing_title != doc.title {
                changed = true;
            }

            let next_branch = if create_branches {
                desired_branch
            } else {
                existing_branch.clone()
            };
            if create_branches && existing_branch != next_branch {
                changed = true;
            }

            if changed {
                sqlx::query(
                    r#"
                    UPDATE tasks
                    SET title = ?1,
                        branch_name = ?2,
                        updated_at_ms = ?3
                    WHERE id = ?4
                    "#,
                )
                .bind(&doc.title)
                .bind(next_branch.as_deref())
                .bind(now_ms)
                .bind(task_id)
                .execute(&mut *tx)
                .await?;
                updated_task_ids.insert(task_id);
            }

            task_rows.insert(
                doc.local_ref.clone(),
                TaskRowState {
                    id: task_id,
                    parent_task_id,
                },
            );
        } else {
            let task_id = TaskId::new();
            let branch_name = if create_branches {
                desired_branch
            } else {
                None
            };
            sqlx::query(
                r#"
                INSERT INTO tasks (
                    id,
                    epic_id,
                    parent_task_id,
                    created_at_ms,
                    updated_at_ms,
                    local_ref,
                    title,
                    branch_name,
                    merge_readiness,
                    state
                )
                VALUES (?1, ?2, NULL, ?3, ?4, ?5, ?6, ?7, 'unknown', 'todo')
                "#,
            )
            .bind(task_id)
            .bind(epic_id)
            .bind(now_ms)
            .bind(now_ms)
            .bind(&doc.local_ref)
            .bind(&doc.title)
            .bind(branch_name.as_deref())
            .execute(&mut *tx)
            .await?;

            stats.tasks_created += 1;
            created_task_ids.insert(task_id);
            task_rows.insert(
                doc.local_ref.clone(),
                TaskRowState {
                    id: task_id,
                    parent_task_id: None,
                },
            );
        }
    }

    for doc in &task_docs {
        let desired_parent = match doc.parent_ref.as_deref() {
            Some(parent_ref) => {
                let Some(parent_row) = task_rows.get(parent_ref) else {
                    return Err(LocalSyncError::UnknownParentRef {
                        parent_ref: parent_ref.to_string(),
                        path: doc.path.display().to_string(),
                    });
                };
                Some(parent_row.id)
            }
            None => None,
        };

        let Some(task_row) = task_rows.get_mut(&doc.local_ref) else {
            continue;
        };

        if task_row.parent_task_id != desired_parent {
            sqlx::query(
                r#"
                UPDATE tasks
                SET parent_task_id = ?1,
                    updated_at_ms = ?2
                WHERE id = ?3
                "#,
            )
            .bind(desired_parent)
            .bind(now_ms)
            .bind(task_row.id)
            .execute(&mut *tx)
            .await?;
            stats.parent_links_updated += 1;
            task_row.parent_task_id = desired_parent;
            if !created_task_ids.contains(&task_row.id) {
                updated_task_ids.insert(task_row.id);
            }
        }
    }

    stats.tasks_updated = updated_task_ids.len().min(u64::MAX as usize) as u64;

    tx.commit().await?;
    Ok(stats)
}

fn load_epic_doc(repo_root: &Path, epic_slug: &str) -> Result<ParsedEpicDoc, LocalSyncError> {
    let epic_readme = repo_root.join("epics").join(epic_slug).join("README.md");
    if !epic_readme.exists() {
        return Err(LocalSyncError::EpicDocMissing(
            epic_readme.display().to_string(),
        ));
    }

    let text = std::fs::read_to_string(&epic_readme)?;
    let (frontmatter, body) = split_frontmatter(&text, &epic_readme)?;

    let rn_slug = yaml_get_string(frontmatter.as_ref(), &["rn", "slug"]);
    if let Some(actual) = rn_slug
        && actual != epic_slug
    {
        return Err(LocalSyncError::EpicSlugMismatch {
            expected: epic_slug.to_string(),
            actual,
        });
    }

    let title = yaml_get_string(frontmatter.as_ref(), &["rn", "name"])
        .or_else(|| extract_first_h1(body))
        .unwrap_or_else(|| epic_slug.to_string());

    Ok(ParsedEpicDoc {
        slug: epic_slug.to_string(),
        title,
    })
}

fn load_task_docs(repo_root: &Path, epic_slug: &str) -> Result<Vec<ParsedTaskDoc>, LocalSyncError> {
    let tasks_dir = repo_root.join("epics").join(epic_slug).join("tasks");
    if !tasks_dir.exists() {
        return Ok(Vec::new());
    }

    let mut docs = Vec::new();
    let mut seen_refs: HashSet<String> = HashSet::new();

    let mut task_dirs: Vec<PathBuf> = std::fs::read_dir(&tasks_dir)?
        .filter_map(|entry| entry.ok().map(|v| v.path()))
        .filter(|path| path.is_dir())
        .collect();
    task_dirs.sort();

    for task_dir in task_dirs {
        let readme = task_dir.join("README.md");
        if !readme.exists() {
            continue;
        }

        let text = std::fs::read_to_string(&readme)?;
        let (frontmatter, body) = split_frontmatter(&text, &readme)?;
        let title = extract_first_h1(body)
            .ok_or_else(|| LocalSyncError::TaskDocMissingTitle(readme.display().to_string()))?;

        let local_ref = yaml_get_string(frontmatter.as_ref(), &["rn", "id"]).or_else(|| {
            task_dir
                .file_name()
                .and_then(|name| name.to_str())
                .and_then(parse_task_ref_from_dir_name)
        });
        let local_ref = local_ref
            .ok_or_else(|| LocalSyncError::TaskDocMissingRef(readme.display().to_string()))?;

        if !seen_refs.insert(local_ref.clone()) {
            return Err(LocalSyncError::DuplicateTaskRef { local_ref });
        }

        let parent_ref = yaml_get_string(frontmatter.as_ref(), &["rn", "parent"]);
        let branch_name = yaml_get_string(frontmatter.as_ref(), &["rn", "node", "branch"]);

        docs.push(ParsedTaskDoc {
            path: readme,
            local_ref,
            title,
            parent_ref,
            branch_name,
        });
    }

    Ok(docs)
}

fn split_frontmatter<'a>(
    text: &'a str,
    path: &Path,
) -> Result<(Option<YamlValue>, &'a str), LocalSyncError> {
    let (prefix_len, tail) = if let Some(tail) = text.strip_prefix("---\n") {
        (4_usize, tail)
    } else if let Some(tail) = text.strip_prefix("---\r\n") {
        (5_usize, tail)
    } else {
        return Ok((None, text));
    };

    let mut offset = prefix_len;
    for segment in tail.split_inclusive('\n') {
        let line = segment.trim_end_matches(['\n', '\r']);
        if line == "---" {
            let yaml_start = prefix_len;
            let yaml_end = offset;
            let body_start = offset + segment.len();
            let yaml_raw = &text[yaml_start..yaml_end];
            let yaml = serde_yaml::from_str::<YamlValue>(yaml_raw).map_err(|err| {
                LocalSyncError::InvalidFrontmatter {
                    path: path.display().to_string(),
                    message: err.to_string(),
                }
            })?;
            return Ok((Some(yaml), &text[body_start..]));
        }
        offset += segment.len();
    }

    Err(LocalSyncError::UnterminatedFrontmatter {
        path: path.display().to_string(),
    })
}

fn yaml_get_string(root: Option<&YamlValue>, path: &[&str]) -> Option<String> {
    let mut current = root?;
    for key in path {
        let YamlValue::Mapping(map) = current else {
            return None;
        };
        current = map.get(&YamlValue::String((*key).to_string()))?;
    }
    match current {
        YamlValue::String(value) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        }
        _ => None,
    }
}

fn extract_first_h1(body: &str) -> Option<String> {
    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("# ") {
            let title = rest.trim();
            if !title.is_empty() {
                return Some(title.to_string());
            }
        }
    }
    None
}

fn parse_task_ref_from_dir_name(name: &str) -> Option<String> {
    let suffix = name.strip_prefix("T-")?;
    if suffix.is_empty() || !suffix.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    Some(name.to_string())
}

fn now_unix_ms() -> i64 {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX)
}
