-- Add durable session metadata + pinning.
--
-- T-40 introduces:
-- - `agent_sessions`: durable session/conversation rows (task-scoped and chat-scoped).
-- - `session_pins`: 0/1 pinned chat session per epic.
--
-- We also strengthen session event integrity by ensuring `session_events.session_id` references
-- an existing `agent_sessions` row.

CREATE TABLE agent_sessions (
    session_id BLOB(16) PRIMARY KEY NOT NULL,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    scope_workspace_id BLOB(16) NOT NULL,
    scope_repo_id BLOB(16) NOT NULL,
    scope_kind TEXT NOT NULL,
    task_id BLOB(16),
    agent_kind TEXT NOT NULL,
    interface_mode TEXT NOT NULL,
    status TEXT NOT NULL,
    external_session_ref TEXT NOT NULL DEFAULT '{"type":"none"}',
    title TEXT,
    started_at_ms INTEGER,
    ended_at_ms INTEGER,
    closed_at_ms INTEGER,
    FOREIGN KEY (scope_workspace_id, scope_repo_id) REFERENCES repositories (workspace_id, id) ON DELETE CASCADE,
    FOREIGN KEY (task_id) REFERENCES tasks (id) ON DELETE CASCADE,
    CHECK (scope_kind IN ('task', 'chat')),
    CHECK (
        (scope_kind = 'chat' AND task_id IS NULL)
        OR (scope_kind = 'task' AND task_id IS NOT NULL)
    ),
    CHECK (agent_kind IN ('codex', 'claude_code', 'shell')),
    CHECK (interface_mode IN ('shell_tmux', 'structured_exec', 'app_server')),
    CHECK (status IN ('running', 'blocked', 'stopped', 'error'))
);

CREATE INDEX idx_agent_sessions_task_created_at ON agent_sessions (task_id, created_at_ms);
CREATE INDEX idx_agent_sessions_scope_created_at ON agent_sessions (
    scope_workspace_id,
    scope_repo_id,
    created_at_ms
);

-- At most one active task-scoped conversation per (task_id, interface_mode).
CREATE UNIQUE INDEX uq_agent_sessions_active_task_interface_mode
ON agent_sessions (task_id, interface_mode)
WHERE scope_kind = 'task' AND ended_at_ms IS NULL;

CREATE TABLE session_pins (
    epic_id BLOB(16) PRIMARY KEY NOT NULL,
    session_id BLOB(16) NOT NULL,
    FOREIGN KEY (epic_id) REFERENCES epics (id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES agent_sessions (session_id) ON DELETE CASCADE
);

CREATE INDEX idx_session_pins_session_id ON session_pins (session_id);

-- Best-effort backfill for existing DBs that wrote `session_events` before `agent_sessions` existed.
INSERT OR IGNORE INTO agent_sessions (
    session_id,
    created_at_ms,
    updated_at_ms,
    scope_workspace_id,
    scope_repo_id,
    scope_kind,
    task_id,
    agent_kind,
    interface_mode,
    status,
    external_session_ref
)
SELECT
    session_id,
    MIN(created_at_ms) AS created_at_ms,
    MAX(created_at_ms) AS updated_at_ms,
    MAX(scope_workspace_id) AS scope_workspace_id,
    MAX(scope_repo_id) AS scope_repo_id,
    CASE
        WHEN MAX(task_id) IS NOT NULL THEN 'task'
        ELSE 'chat'
    END AS scope_kind,
    MAX(task_id) AS task_id,
    'shell' AS agent_kind,
    'shell_tmux' AS interface_mode,
    'stopped' AS status,
    '{"type":"none"}' AS external_session_ref
FROM session_events
WHERE scope_workspace_id IS NOT NULL AND scope_repo_id IS NOT NULL
GROUP BY session_id;

-- Add FK integrity for `session_events.session_id`.
ALTER TABLE session_events RENAME TO session_events_old;
DROP INDEX idx_session_events_session_created_at;
DROP INDEX idx_session_events_task_created_at;

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
    FOREIGN KEY (session_id) REFERENCES agent_sessions (session_id) ON DELETE CASCADE,
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
CREATE INDEX idx_session_events_session_kind_created_at ON session_events (session_id, kind, created_at_ms);
CREATE INDEX idx_session_events_turn_id ON session_events (turn_id);

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
