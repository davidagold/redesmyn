-- Rename legacy chat visibility column to archive semantics.
ALTER TABLE agent_sessions RENAME COLUMN closed_at_ms TO archived_at_ms;
