-- Add durable task state for graph rendering (T-21).
--
-- Conventions:
-- - State values align with the legacy Python enum (`todo`, `in_progress`, `blocked`, `done`).
-- - Keep values in a `CHECK` constraint to avoid stringly-typed typos.

ALTER TABLE tasks
ADD COLUMN state TEXT NOT NULL DEFAULT 'todo'
CHECK (state IN ('todo', 'in_progress', 'blocked', 'done'));

