-- Add optional durable epic association for chat sessions.
ALTER TABLE agent_sessions
ADD COLUMN epic_id BLOB(16) REFERENCES epics(id) ON DELETE SET NULL;

CREATE INDEX idx_agent_sessions_epic_created_at
ON agent_sessions (epic_id, created_at_ms);

-- Best-effort backfill from existing pin records.
UPDATE agent_sessions
SET epic_id = (
    SELECT session_pins.epic_id
    FROM session_pins
    WHERE session_pins.session_id = agent_sessions.session_id
    LIMIT 1
)
WHERE scope_kind = 'chat' AND epic_id IS NULL;
