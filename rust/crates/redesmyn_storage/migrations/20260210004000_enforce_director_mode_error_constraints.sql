-- Rebuild director_mode_state to restore DB-level lifecycle/error invariants.
-- This migration is additive and leaves prior migration files immutable.

PRAGMA foreign_keys = OFF;
PRAGMA legacy_alter_table = ON;

ALTER TABLE director_mode_state RENAME TO director_mode_state_old;

CREATE TABLE director_mode_state (
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

INSERT INTO director_mode_state (
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
    error_reason,
    CASE
        WHEN lifecycle = 'error' THEN COALESCE(error_at_ms, updated_at_ms)
        ELSE error_at_ms
    END
FROM director_mode_state_old;

DROP TABLE director_mode_state_old;

PRAGMA legacy_alter_table = OFF;
PRAGMA foreign_keys = ON;
