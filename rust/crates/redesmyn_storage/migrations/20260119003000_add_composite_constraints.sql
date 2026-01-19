-- Strengthen DB-level invariants with composite foreign keys (no triggers).
--
-- Goals:
-- - Prevent impossible repo scope pairings (workspace A + repo B).
-- - Ensure tasks.parent_task_id never crosses epics.
-- - Ensure session_events scope chains stay coherent (repo → epic → task).
--
-- SQLite constraints can’t be altered in-place, so we rebuild affected tables.

-- Support composite foreign keys.
CREATE UNIQUE INDEX idx_repositories_workspace_id_id ON repositories (workspace_id, id);
CREATE UNIQUE INDEX idx_epics_repo_id_id ON epics (repo_id, id);

-- Backfill: older rows allowed task-scoped session events without epic_id.
UPDATE session_events
SET epic_id = (SELECT epic_id FROM tasks WHERE tasks.id = session_events.task_id)
WHERE scope_kind = 'task' AND epic_id IS NULL;

-- Rebuild tasks to enforce same-epic parenting via composite FK.
CREATE TABLE tasks_new (
    id BLOB(16) PRIMARY KEY NOT NULL,
    epic_id BLOB(16) NOT NULL,
    parent_task_id BLOB(16),
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    local_ref TEXT,
    title TEXT NOT NULL,
    branch_name TEXT,
    merge_readiness TEXT NOT NULL DEFAULT 'unknown',
    UNIQUE (epic_id, id),
    FOREIGN KEY (epic_id) REFERENCES epics (id) ON DELETE CASCADE,
    FOREIGN KEY (epic_id, parent_task_id) REFERENCES tasks_new (epic_id, id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,
    CHECK (merge_readiness IN ('unknown', 'ready', 'blocked'))
);

INSERT INTO tasks_new (
    id,
    epic_id,
    parent_task_id,
    created_at_ms,
    updated_at_ms,
    local_ref,
    title,
    branch_name,
    merge_readiness
)
SELECT
    id,
    epic_id,
    parent_task_id,
    created_at_ms,
    updated_at_ms,
    local_ref,
    title,
    branch_name,
    merge_readiness
FROM tasks;

DROP TABLE tasks;
ALTER TABLE tasks_new RENAME TO tasks;

CREATE INDEX idx_tasks_epic_id ON tasks (epic_id);
CREATE INDEX idx_tasks_parent_task_id ON tasks (parent_task_id);
CREATE UNIQUE INDEX idx_tasks_epic_local_ref ON tasks (epic_id, local_ref) WHERE local_ref IS NOT NULL;

-- Rebuild commands / command_updates with composite repo scope FK.
ALTER TABLE command_updates RENAME TO command_updates_old;
ALTER TABLE commands RENAME TO commands_old;

DROP INDEX idx_command_updates_command_created_at;
DROP INDEX idx_commands_scope_created_at;
DROP INDEX idx_commands_state_created_at;
DROP INDEX idx_commands_target_task_created_at;

CREATE TABLE commands (
    id BLOB(16) PRIMARY KEY NOT NULL,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    scope_kind TEXT NOT NULL,
    scope_workspace_id BLOB(16),
    scope_repo_id BLOB(16),
    target_task_id BLOB(16),
    kind TEXT NOT NULL,
    state TEXT NOT NULL,
    payload BLOB NOT NULL,
    FOREIGN KEY (scope_workspace_id, scope_repo_id) REFERENCES repositories (workspace_id, id) ON DELETE CASCADE,
    FOREIGN KEY (target_task_id) REFERENCES tasks (id) ON DELETE SET NULL,
    CHECK (scope_kind IN ('none', 'repo')),
    CHECK (
        (scope_kind = 'none' AND scope_workspace_id IS NULL AND scope_repo_id IS NULL)
        OR (scope_kind = 'repo' AND scope_workspace_id IS NOT NULL AND scope_repo_id IS NOT NULL)
    ),
    CHECK (state IN ('accepted', 'running', 'blocked', 'resumable', 'succeeded', 'failed', 'canceled'))
);

CREATE INDEX idx_commands_scope_created_at ON commands (
    scope_kind,
    scope_workspace_id,
    scope_repo_id,
    created_at_ms
);
CREATE INDEX idx_commands_state_created_at ON commands (state, created_at_ms);
CREATE INDEX idx_commands_target_task_created_at ON commands (target_task_id, created_at_ms);

INSERT INTO commands (
    id,
    created_at_ms,
    updated_at_ms,
    scope_kind,
    scope_workspace_id,
    scope_repo_id,
    target_task_id,
    kind,
    state,
    payload
)
SELECT
    id,
    created_at_ms,
    updated_at_ms,
    scope_kind,
    scope_workspace_id,
    scope_repo_id,
    target_task_id,
    kind,
    state,
    payload
FROM commands_old;

CREATE TABLE command_updates (
    id BLOB(16) PRIMARY KEY NOT NULL,
    command_id BLOB(16) NOT NULL,
    created_at_ms INTEGER NOT NULL,
    state TEXT NOT NULL,
    message TEXT,
    progress_current INTEGER,
    progress_total INTEGER,
    detail BLOB,
    FOREIGN KEY (command_id) REFERENCES commands (id) ON DELETE CASCADE,
    CHECK (state IN ('accepted', 'running', 'blocked', 'resumable', 'succeeded', 'failed', 'canceled'))
);

CREATE INDEX idx_command_updates_command_created_at ON command_updates (command_id, created_at_ms);

INSERT INTO command_updates (
    id,
    command_id,
    created_at_ms,
    state,
    message,
    progress_current,
    progress_total,
    detail
)
SELECT
    id,
    command_id,
    created_at_ms,
    state,
    message,
    progress_current,
    progress_total,
    detail
FROM command_updates_old;

DROP TABLE command_updates_old;
DROP TABLE commands_old;

-- Rebuild events with composite repo scope FK.
ALTER TABLE events RENAME TO events_old;
DROP INDEX idx_events_scope_created_at;

CREATE TABLE events (
    id BLOB(16) PRIMARY KEY NOT NULL,
    created_at_ms INTEGER NOT NULL,
    scope_kind TEXT NOT NULL,
    scope_workspace_id BLOB(16),
    scope_repo_id BLOB(16),
    kind TEXT NOT NULL,
    payload BLOB NOT NULL,
    FOREIGN KEY (scope_workspace_id, scope_repo_id) REFERENCES repositories (workspace_id, id) ON DELETE CASCADE,
    CHECK (scope_kind IN ('none', 'repo')),
    CHECK (
        (scope_kind = 'none' AND scope_workspace_id IS NULL AND scope_repo_id IS NULL)
        OR (scope_kind = 'repo' AND scope_workspace_id IS NOT NULL AND scope_repo_id IS NOT NULL)
    )
);

INSERT INTO events (id, created_at_ms, scope_kind, scope_workspace_id, scope_repo_id, kind, payload)
SELECT id, created_at_ms, scope_kind, scope_workspace_id, scope_repo_id, kind, payload
FROM events_old;

DROP TABLE events_old;

CREATE INDEX idx_events_scope_created_at ON events (
    scope_kind,
    scope_workspace_id,
    scope_repo_id,
    created_at_ms
);

-- Rebuild artifacts / session_events with composite scope FK + scope chain constraints.
ALTER TABLE session_events RENAME TO session_events_old;
ALTER TABLE artifacts RENAME TO artifacts_old;

DROP INDEX idx_session_events_session_created_at;
DROP INDEX idx_session_events_task_created_at;
DROP INDEX idx_artifacts_scope_created_at;

CREATE TABLE artifacts (
    id BLOB(16) PRIMARY KEY NOT NULL,
    created_at_ms INTEGER NOT NULL,
    scope_kind TEXT NOT NULL,
    scope_workspace_id BLOB(16),
    scope_repo_id BLOB(16),
    kind TEXT NOT NULL,
    content_hash TEXT,
    byte_len INTEGER,
    mime TEXT,
    storage_hint TEXT,
    FOREIGN KEY (scope_workspace_id, scope_repo_id) REFERENCES repositories (workspace_id, id) ON DELETE CASCADE,
    CHECK (scope_kind IN ('none', 'repo')),
    CHECK (
        (scope_kind = 'none' AND scope_workspace_id IS NULL AND scope_repo_id IS NULL)
        OR (scope_kind = 'repo' AND scope_workspace_id IS NOT NULL AND scope_repo_id IS NOT NULL)
    )
);

CREATE INDEX idx_artifacts_scope_created_at ON artifacts (
    scope_kind,
    scope_workspace_id,
    scope_repo_id,
    created_at_ms
);

INSERT INTO artifacts (
    id,
    created_at_ms,
    scope_kind,
    scope_workspace_id,
    scope_repo_id,
    kind,
    content_hash,
    byte_len,
    mime,
    storage_hint
)
SELECT
    id,
    created_at_ms,
    scope_kind,
    scope_workspace_id,
    scope_repo_id,
    kind,
    content_hash,
    byte_len,
    mime,
    storage_hint
FROM artifacts_old;

CREATE TABLE session_events (
    id BLOB(16) PRIMARY KEY NOT NULL,
    session_id BLOB(16) NOT NULL,
    created_at_ms INTEGER NOT NULL,
    scope_kind TEXT NOT NULL,
    scope_workspace_id BLOB(16),
    scope_repo_id BLOB(16),
    epic_id BLOB(16),
    task_id BLOB(16),
    kind TEXT NOT NULL,
    turn_id TEXT,
    message_preview TEXT,
    artifact_id BLOB(16),
    payload BLOB NOT NULL DEFAULT X'',
    -- Repo scope pairing must exist and be consistent.
    FOREIGN KEY (scope_workspace_id, scope_repo_id) REFERENCES repositories (workspace_id, id) ON DELETE CASCADE,
    -- Epic/task scope chains must be consistent with repo scope.
    FOREIGN KEY (scope_repo_id, epic_id) REFERENCES epics (repo_id, id) ON DELETE CASCADE,
    FOREIGN KEY (epic_id, task_id) REFERENCES tasks (epic_id, id) ON DELETE CASCADE,
    FOREIGN KEY (artifact_id) REFERENCES artifacts (id) ON DELETE SET NULL,
    CHECK (scope_kind IN ('none', 'repo', 'epic', 'task')),
    CHECK (
        (scope_kind = 'none' AND scope_workspace_id IS NULL AND scope_repo_id IS NULL AND epic_id IS NULL AND task_id IS NULL)
        OR (scope_kind = 'repo' AND scope_workspace_id IS NOT NULL AND scope_repo_id IS NOT NULL AND epic_id IS NULL AND task_id IS NULL)
        OR (scope_kind = 'epic' AND scope_workspace_id IS NOT NULL AND scope_repo_id IS NOT NULL AND epic_id IS NOT NULL AND task_id IS NULL)
        OR (scope_kind = 'task' AND scope_workspace_id IS NOT NULL AND scope_repo_id IS NOT NULL AND epic_id IS NOT NULL AND task_id IS NOT NULL)
    )
);

CREATE INDEX idx_session_events_session_created_at ON session_events (session_id, created_at_ms);
CREATE INDEX idx_session_events_task_created_at ON session_events (task_id, created_at_ms);

INSERT INTO session_events (
    id,
    session_id,
    created_at_ms,
    scope_kind,
    scope_workspace_id,
    scope_repo_id,
    epic_id,
    task_id,
    kind,
    turn_id,
    message_preview,
    artifact_id,
    payload
)
SELECT
    id,
    session_id,
    created_at_ms,
    scope_kind,
    scope_workspace_id,
    scope_repo_id,
    epic_id,
    task_id,
    kind,
    turn_id,
    message_preview,
    artifact_id,
    payload
FROM session_events_old;

DROP TABLE session_events_old;
DROP TABLE artifacts_old;
