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

CREATE TABLE director_merge_authority_policy_default (
    singleton INTEGER PRIMARY KEY NOT NULL CHECK (singleton = 1),
    updated_at_ms INTEGER NOT NULL,
    yolo_merge INTEGER NOT NULL CHECK (yolo_merge IN (0, 1))
);

INSERT INTO director_merge_authority_policy_default (
    singleton,
    updated_at_ms,
    yolo_merge
)
VALUES (1, 0, 0);

CREATE TABLE epic_merge_authority_policy_override (
    epic_id BLOB(16) PRIMARY KEY NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    yolo_merge INTEGER NOT NULL CHECK (yolo_merge IN (0, 1)),
    FOREIGN KEY (epic_id) REFERENCES epics (id) ON DELETE CASCADE
);
