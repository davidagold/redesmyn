-- Adjust `tasks.parent_task_id` delete behavior.
--
-- We want a DB-level guarantee that `parent_task_id` never crosses epics, but we
-- do *not* want deleting a parent to implicitly delete an entire subtree.
--
-- Approach (SQLite, no triggers):
-- - Keep the composite FK `(epic_id, parent_task_id) -> tasks(epic_id, id)` to
--   prevent cross-epic parents.
-- - Add a separate FK `parent_task_id -> tasks(id) ON DELETE SET NULL` so
--   deleting a parent detaches its children.
--
-- SQLite cannot alter FK constraints in place, so we rebuild the table.

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
    FOREIGN KEY (parent_task_id) REFERENCES tasks_new (id) ON DELETE SET NULL DEFERRABLE INITIALLY DEFERRED,
    FOREIGN KEY (epic_id, parent_task_id) REFERENCES tasks_new (epic_id, id) DEFERRABLE INITIALLY DEFERRED,
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
