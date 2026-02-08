ALTER TABLE agent_sessions
ADD COLUMN runner_host_id BLOB(16);

ALTER TABLE agent_sessions
ADD COLUMN runner_host_instance_id BLOB(16);

CREATE INDEX idx_agent_sessions_runner_host_active
    ON agent_sessions (runner_host_id, ended_at_ms);

CREATE INDEX idx_agent_sessions_runner_host_instance_active
    ON agent_sessions (runner_host_instance_id, ended_at_ms);
