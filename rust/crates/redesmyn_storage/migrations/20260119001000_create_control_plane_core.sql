-- Control plane core schema (v0).
--
-- Conventions:
-- - ULIDs are stored as `BLOB(16)` (raw bytes).
-- - `*_at_ms` fields are UNIX epoch milliseconds.

CREATE TABLE workspaces (
    id BLOB(16) PRIMARY KEY NOT NULL,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    name TEXT NOT NULL
);

CREATE TABLE repositories (
    id BLOB(16) PRIMARY KEY NOT NULL,
    workspace_id BLOB(16) NOT NULL,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    slug TEXT NOT NULL,
    title TEXT NOT NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces (id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX idx_repositories_workspace_slug ON repositories (workspace_id, slug);

CREATE TABLE epics (
    id BLOB(16) PRIMARY KEY NOT NULL,
    repo_id BLOB(16) NOT NULL,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    slug TEXT NOT NULL,
    title TEXT NOT NULL,
    FOREIGN KEY (repo_id) REFERENCES repositories (id) ON DELETE CASCADE
);

CREATE INDEX idx_epics_repo_id ON epics (repo_id);
CREATE UNIQUE INDEX idx_epics_repo_slug ON epics (repo_id, slug);

CREATE TABLE tasks (
    id BLOB(16) PRIMARY KEY NOT NULL,
    epic_id BLOB(16) NOT NULL,
    parent_task_id BLOB(16),
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    local_ref TEXT,
    title TEXT NOT NULL,
    branch_name TEXT,
    merge_readiness TEXT NOT NULL DEFAULT 'unknown',
    FOREIGN KEY (epic_id) REFERENCES epics (id) ON DELETE CASCADE,
    FOREIGN KEY (parent_task_id) REFERENCES tasks (id) ON DELETE SET NULL,
    CHECK (merge_readiness IN ('unknown', 'ready', 'blocked'))
);

CREATE INDEX idx_tasks_epic_id ON tasks (epic_id);
CREATE INDEX idx_tasks_parent_task_id ON tasks (parent_task_id);
CREATE UNIQUE INDEX idx_tasks_epic_local_ref ON tasks (epic_id, local_ref) WHERE local_ref IS NOT NULL;

CREATE TABLE task_relations (
    id BLOB(16) PRIMARY KEY NOT NULL,
    created_at_ms INTEGER NOT NULL,
    from_task_id BLOB(16) NOT NULL,
    to_task_id BLOB(16) NOT NULL,
    kind TEXT NOT NULL,
    FOREIGN KEY (from_task_id) REFERENCES tasks (id) ON DELETE CASCADE,
    FOREIGN KEY (to_task_id) REFERENCES tasks (id) ON DELETE CASCADE,
    CHECK (kind IN ('after'))
);

CREATE UNIQUE INDEX idx_task_relations_unique ON task_relations (from_task_id, to_task_id, kind);
CREATE INDEX idx_task_relations_from_task_id ON task_relations (from_task_id);
CREATE INDEX idx_task_relations_to_task_id ON task_relations (to_task_id);

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
    FOREIGN KEY (epic_id) REFERENCES epics (id) ON DELETE CASCADE,
    FOREIGN KEY (task_id) REFERENCES tasks (id) ON DELETE CASCADE,
    FOREIGN KEY (artifact_id) REFERENCES artifacts (id) ON DELETE SET NULL,
    CHECK (scope_kind IN ('none', 'repo', 'epic', 'task')),
    CHECK (
        (scope_kind = 'none' AND scope_workspace_id IS NULL AND scope_repo_id IS NULL AND epic_id IS NULL AND task_id IS NULL)
        OR (scope_kind = 'repo' AND scope_workspace_id IS NOT NULL AND scope_repo_id IS NOT NULL AND epic_id IS NULL AND task_id IS NULL)
        OR (scope_kind = 'epic' AND scope_workspace_id IS NOT NULL AND scope_repo_id IS NOT NULL AND epic_id IS NOT NULL AND task_id IS NULL)
        OR (scope_kind = 'task' AND scope_workspace_id IS NOT NULL AND scope_repo_id IS NOT NULL AND task_id IS NOT NULL)
    )
);

CREATE INDEX idx_session_events_session_created_at ON session_events (session_id, created_at_ms);
CREATE INDEX idx_session_events_task_created_at ON session_events (task_id, created_at_ms);

CREATE TABLE hosts (
    id BLOB(16) PRIMARY KEY NOT NULL,
    created_at_ms INTEGER NOT NULL,
    hostname TEXT
);

CREATE TABLE daemon_presence (
    host_instance_id BLOB(16) PRIMARY KEY NOT NULL,
    host_id BLOB(16) NOT NULL,
    connected_at_ms INTEGER NOT NULL,
    last_heartbeat_at_ms INTEGER NOT NULL,
    disconnected_at_ms INTEGER,
    FOREIGN KEY (host_id) REFERENCES hosts (id) ON DELETE CASCADE
);

CREATE INDEX idx_daemon_presence_host_id ON daemon_presence (host_id);
CREATE INDEX idx_daemon_presence_last_heartbeat ON daemon_presence (last_heartbeat_at_ms);

-- Expand the append-only event log to include scope columns and indexing.
ALTER TABLE events RENAME TO events_old;

CREATE TABLE events (
    id BLOB(16) PRIMARY KEY NOT NULL,
    created_at_ms INTEGER NOT NULL,
    scope_kind TEXT NOT NULL,
    scope_workspace_id BLOB(16),
    scope_repo_id BLOB(16),
    kind TEXT NOT NULL,
    payload BLOB NOT NULL,
    CHECK (scope_kind IN ('none', 'repo')),
    CHECK (
        (scope_kind = 'none' AND scope_workspace_id IS NULL AND scope_repo_id IS NULL)
        OR (scope_kind = 'repo' AND scope_workspace_id IS NOT NULL AND scope_repo_id IS NOT NULL)
    )
);

INSERT INTO events (id, created_at_ms, scope_kind, scope_workspace_id, scope_repo_id, kind, payload)
SELECT id, created_at_ms, 'none', NULL, NULL, kind, payload
FROM events_old;

DROP TABLE events_old;

CREATE INDEX idx_events_scope_created_at ON events (
    scope_kind,
    scope_workspace_id,
    scope_repo_id,
    created_at_ms
);
