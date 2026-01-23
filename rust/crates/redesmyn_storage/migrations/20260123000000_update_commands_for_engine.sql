-- Update command persistence schema for the command engine (T-19).
--
-- Changes:
-- - Add `queued` as an explicit initial command state.
-- - Add `idempotency_key` and `created_by` fields to commands.
-- - Add a partial unique index for idempotency keys scoped by (scope + kind).
--
-- SQLite cannot alter CHECK constraints in place, so we rebuild affected tables.

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
    idempotency_key TEXT,
    created_by TEXT,
    payload BLOB NOT NULL,
    FOREIGN KEY (scope_workspace_id) REFERENCES workspaces (id) ON DELETE CASCADE,
    FOREIGN KEY (scope_repo_id) REFERENCES repositories (id) ON DELETE CASCADE,
    FOREIGN KEY (target_task_id) REFERENCES tasks (id) ON DELETE SET NULL,
    CHECK (scope_kind IN ('none', 'repo')),
    CHECK (
        (scope_kind = 'none' AND scope_workspace_id IS NULL AND scope_repo_id IS NULL)
        OR (scope_kind = 'repo' AND scope_workspace_id IS NOT NULL AND scope_repo_id IS NOT NULL)
    ),
    CHECK (state IN ('queued', 'accepted', 'running', 'blocked', 'resumable', 'succeeded', 'failed', 'canceled'))
);

CREATE INDEX idx_commands_scope_created_at ON commands (
    scope_kind,
    scope_workspace_id,
    scope_repo_id,
    created_at_ms
);
CREATE INDEX idx_commands_state_created_at ON commands (state, created_at_ms);
CREATE INDEX idx_commands_target_task_created_at ON commands (target_task_id, created_at_ms);
CREATE UNIQUE INDEX idx_commands_idempotency_key ON commands (
    scope_kind,
    scope_workspace_id,
    scope_repo_id,
    kind,
    idempotency_key
) WHERE idempotency_key IS NOT NULL;

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
    idempotency_key,
    created_by,
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
    NULL,
    NULL,
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
    CHECK (state IN ('queued', 'accepted', 'running', 'blocked', 'resumable', 'succeeded', 'failed', 'canceled'))
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
