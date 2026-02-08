CREATE TABLE session_policy_projection (
    session_id BLOB(16) PRIMARY KEY NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    permissions_mode TEXT,
    permissions_mode_observed_at_ms INTEGER,
    codex_approval_policy TEXT,
    codex_approval_policy_observed_at_ms INTEGER,
    codex_sandbox_policy_json TEXT,
    codex_sandbox_policy_observed_at_ms INTEGER,
    model_id TEXT,
    model_reasoning_effort TEXT,
    model_observed_at_ms INTEGER,
    FOREIGN KEY (session_id) REFERENCES agent_sessions (session_id) ON DELETE CASCADE,
    CHECK (
        permissions_mode IS NULL
        OR permissions_mode IN ('ask', 'auto_approve', 'deny')
    ),
    CHECK (
        codex_approval_policy IS NULL
        OR codex_approval_policy IN ('untrusted', 'on_failure', 'on_request', 'never')
    ),
    CHECK (
        model_reasoning_effort IS NULL
        OR model_reasoning_effort IN ('minimal', 'low', 'medium', 'high', 'xhigh')
    ),
    CHECK (
        (permissions_mode IS NULL AND permissions_mode_observed_at_ms IS NULL)
        OR permissions_mode_observed_at_ms IS NOT NULL
    ),
    CHECK (
        (codex_approval_policy IS NULL AND codex_approval_policy_observed_at_ms IS NULL)
        OR codex_approval_policy_observed_at_ms IS NOT NULL
    ),
    CHECK (
        (codex_sandbox_policy_json IS NULL AND codex_sandbox_policy_observed_at_ms IS NULL)
        OR codex_sandbox_policy_observed_at_ms IS NOT NULL
    ),
    CHECK (
        (
            model_id IS NULL
            AND model_reasoning_effort IS NULL
            AND model_observed_at_ms IS NULL
        )
        OR model_observed_at_ms IS NOT NULL
    )
);

-- Backfill historical per-session policy from existing durable session event rows.
--
-- NOTE: `session_events.payload` is protobuf bytes, so SQL migration backfill uses
-- `kind` + `message_preview` mappings. For structured sandbox variants where full
-- payload details are not represented in preview text, we backfill a normalized
-- policy variant with default fields.
WITH
policy_sessions AS (
    SELECT DISTINCT e.session_id
    FROM session_events e
    JOIN agent_sessions s ON s.session_id = e.session_id
    WHERE e.kind IN (
        'permissions_mode_changed',
        'codex_approval_policy_changed',
        'codex_sandbox_policy_changed',
        'session_model_changed'
    )
),
latest_permissions AS (
    SELECT session_id, observed_at_ms, value
    FROM (
        SELECT
            e.session_id,
            e.created_at_ms AS observed_at_ms,
            CASE e.message_preview
                WHEN 'ask' THEN 'ask'
                WHEN 'auto_approve' THEN 'auto_approve'
                WHEN 'deny' THEN 'deny'
                ELSE NULL
            END AS value,
            ROW_NUMBER() OVER (
                PARTITION BY e.session_id
                ORDER BY e.created_at_ms DESC, e.id DESC
            ) AS rn
        FROM session_events e
        JOIN agent_sessions s ON s.session_id = e.session_id
        WHERE e.kind = 'permissions_mode_changed'
          AND e.message_preview IN ('ask', 'auto_approve', 'deny')
    ) ranked
    WHERE rn = 1
),
latest_approval AS (
    SELECT session_id, observed_at_ms, value
    FROM (
        SELECT
            e.session_id,
            e.created_at_ms AS observed_at_ms,
            CASE e.message_preview
                WHEN 'untrusted' THEN 'untrusted'
                WHEN 'on_failure' THEN 'on_failure'
                WHEN 'on_request' THEN 'on_request'
                WHEN 'never' THEN 'never'
                WHEN 'default' THEN NULL
                ELSE NULL
            END AS value,
            ROW_NUMBER() OVER (
                PARTITION BY e.session_id
                ORDER BY e.created_at_ms DESC, e.id DESC
            ) AS rn
        FROM session_events e
        JOIN agent_sessions s ON s.session_id = e.session_id
        WHERE e.kind = 'codex_approval_policy_changed'
          AND e.message_preview IN ('untrusted', 'on_failure', 'on_request', 'never', 'default')
    ) ranked
    WHERE rn = 1
),
latest_sandbox AS (
    SELECT session_id, observed_at_ms, value
    FROM (
        SELECT
            e.session_id,
            e.created_at_ms AS observed_at_ms,
            CASE e.message_preview
                WHEN 'danger_full_access' THEN '{"type":"dangerFullAccess"}'
                WHEN 'read_only' THEN '{"type":"readOnly"}'
                WHEN 'external_sandbox' THEN '{"type":"externalSandbox"}'
                WHEN 'workspace_write' THEN '{"type":"workspaceWrite"}'
                WHEN 'default' THEN NULL
                ELSE NULL
            END AS value,
            ROW_NUMBER() OVER (
                PARTITION BY e.session_id
                ORDER BY e.created_at_ms DESC, e.id DESC
            ) AS rn
        FROM session_events e
        JOIN agent_sessions s ON s.session_id = e.session_id
        WHERE e.kind = 'codex_sandbox_policy_changed'
          AND e.message_preview IN (
              'danger_full_access',
              'read_only',
              'external_sandbox',
              'workspace_write',
              'default'
          )
    ) ranked
    WHERE rn = 1
),
latest_model AS (
    SELECT
        session_id,
        observed_at_ms,
        CASE
            WHEN model_raw IS NULL OR TRIM(model_raw) = '' OR TRIM(model_raw) = 'default'
                THEN NULL
            ELSE TRIM(model_raw)
        END AS model_id,
        CASE LOWER(TRIM(effort_raw))
            WHEN 'minimal' THEN 'minimal'
            WHEN 'low' THEN 'low'
            WHEN 'medium' THEN 'medium'
            WHEN 'high' THEN 'high'
            WHEN 'xhigh' THEN 'xhigh'
            ELSE NULL
        END AS reasoning_effort
    FROM (
        SELECT
            e.session_id,
            e.created_at_ms AS observed_at_ms,
            CASE
                WHEN e.message_preview IS NULL THEN NULL
                WHEN instr(e.message_preview, ':') > 0
                    THEN substr(e.message_preview, 1, instr(e.message_preview, ':') - 1)
                ELSE e.message_preview
            END AS model_raw,
            CASE
                WHEN e.message_preview IS NULL THEN NULL
                WHEN instr(e.message_preview, ':') > 0
                    THEN substr(e.message_preview, instr(e.message_preview, ':') + 1)
                ELSE NULL
            END AS effort_raw,
            ROW_NUMBER() OVER (
                PARTITION BY e.session_id
                ORDER BY e.created_at_ms DESC, e.id DESC
            ) AS rn
        FROM session_events e
        JOIN agent_sessions s ON s.session_id = e.session_id
        WHERE e.kind = 'session_model_changed'
    ) ranked
    WHERE rn = 1
)
INSERT INTO session_policy_projection (
    session_id,
    updated_at_ms,
    permissions_mode,
    permissions_mode_observed_at_ms,
    codex_approval_policy,
    codex_approval_policy_observed_at_ms,
    codex_sandbox_policy_json,
    codex_sandbox_policy_observed_at_ms,
    model_id,
    model_reasoning_effort,
    model_observed_at_ms
)
SELECT
    ps.session_id,
    MAX(
        COALESCE(lp.observed_at_ms, 0),
        COALESCE(la.observed_at_ms, 0),
        COALESCE(ls.observed_at_ms, 0),
        COALESCE(lm.observed_at_ms, 0)
    ) AS updated_at_ms,
    lp.value,
    lp.observed_at_ms,
    la.value,
    la.observed_at_ms,
    ls.value,
    ls.observed_at_ms,
    lm.model_id,
    lm.reasoning_effort,
    lm.observed_at_ms
FROM policy_sessions ps
LEFT JOIN latest_permissions lp ON lp.session_id = ps.session_id
LEFT JOIN latest_approval la ON la.session_id = ps.session_id
LEFT JOIN latest_sandbox ls ON ls.session_id = ps.session_id
LEFT JOIN latest_model lm ON lm.session_id = ps.session_id
WHERE
    lp.observed_at_ms IS NOT NULL
    OR la.observed_at_ms IS NOT NULL
    OR ls.observed_at_ms IS NOT NULL
    OR lm.observed_at_ms IS NOT NULL;

CREATE INDEX idx_session_policy_projection_permissions_mode_observed
    ON session_policy_projection (permissions_mode_observed_at_ms DESC);

CREATE INDEX idx_session_policy_projection_codex_approval_observed
    ON session_policy_projection (codex_approval_policy_observed_at_ms DESC);

CREATE INDEX idx_session_policy_projection_codex_sandbox_observed
    ON session_policy_projection (codex_sandbox_policy_observed_at_ms DESC);

CREATE INDEX idx_session_policy_projection_model_observed
    ON session_policy_projection (model_observed_at_ms DESC);
