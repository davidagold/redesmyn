CREATE TABLE director_wake_state (
    epic_id BLOB(16) PRIMARY KEY NOT NULL,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    director_session_ref TEXT,
    ack_cursor_event_id BLOB(16),
    in_flight_wake_id TEXT,
    in_flight_high_water_event_id BLOB(16),
    in_flight_reason_mask INTEGER NOT NULL DEFAULT 0,
    pending_high_water_event_id BLOB(16),
    pending_reason_mask INTEGER NOT NULL DEFAULT 0,
    last_wake_reason_mask INTEGER NOT NULL DEFAULT 0,
    last_wake_at_ms INTEGER,
    director_mode TEXT NOT NULL DEFAULT 'paused',
    FOREIGN KEY (epic_id) REFERENCES epics (id) ON DELETE CASCADE,
    CHECK (director_mode IN ('active', 'paused', 'error', 'resume_required')),
    CHECK (
        (in_flight_wake_id IS NULL AND in_flight_high_water_event_id IS NULL)
        OR (in_flight_wake_id IS NOT NULL AND in_flight_high_water_event_id IS NOT NULL)
    )
);

CREATE INDEX idx_director_wake_state_mode ON director_wake_state (director_mode);
