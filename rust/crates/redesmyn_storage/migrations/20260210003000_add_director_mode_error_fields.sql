-- Add explicit director error metadata as an additive migration.
--
-- SQLite cannot add CHECK constraints in-place, so rebuild director_mode_state
-- with new columns + invariants and copy existing rows.

CREATE TABLE director_mode_state_new (
    epic_id BLOB(16) PRIMARY KEY NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    lifecycle TEXT NOT NULL DEFAULT 'inactive',
    director_session_id BLOB(16),
    activation_intent TEXT,
    resume_required_reason TEXT,
    resume_required_at_ms INTEGER,
    error_reason TEXT,
    error_at_ms INTEGER,
    FOREIGN KEY (epic_id) REFERENCES epics (id) ON DELETE CASCADE,
    FOREIGN KEY (director_session_id) REFERENCES agent_sessions (session_id) ON DELETE SET NULL,
    CHECK (
        lifecycle IN ('inactive', 'active', 'paused', 'resume_required', 'error')
    ),
    CHECK (
        activation_intent IS NULL
        OR activation_intent IN ('run_in_current_session', 'run_in_new_session')
    ),
    CHECK (
        (
            lifecycle = 'resume_required'
            AND resume_required_at_ms IS NOT NULL
        )
        OR (
            lifecycle <> 'resume_required'
            AND resume_required_reason IS NULL
            AND resume_required_at_ms IS NULL
        )
    ),
    CHECK (
        (
            lifecycle = 'error'
            AND error_at_ms IS NOT NULL
        )
        OR (
            lifecycle <> 'error'
            AND error_reason IS NULL
            AND error_at_ms IS NULL
        )
    )
);

INSERT INTO director_mode_state_new (
    epic_id,
    updated_at_ms,
    lifecycle,
    director_session_id,
    activation_intent,
    resume_required_reason,
    resume_required_at_ms,
    error_reason,
    error_at_ms
)
SELECT
    epic_id,
    updated_at_ms,
    lifecycle,
    director_session_id,
    activation_intent,
    resume_required_reason,
    resume_required_at_ms,
    NULL AS error_reason,
    CASE
        WHEN lifecycle = 'error' THEN updated_at_ms
        ELSE NULL
    END AS error_at_ms
FROM director_mode_state;

DROP TABLE director_mode_state;
ALTER TABLE director_mode_state_new RENAME TO director_mode_state;
