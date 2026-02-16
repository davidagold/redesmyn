-- Add explicit director error metadata as an additive migration.
--
-- Some environments may already include these columns from earlier migration
-- variants. We rely on migration-repair logic in sqlite.rs to mark this
-- migration as applied when duplicate-column errors are encountered.

ALTER TABLE director_mode_state
ADD COLUMN error_reason TEXT;

ALTER TABLE director_mode_state
ADD COLUMN error_at_ms INTEGER;

-- Preserve existing values and only backfill missing timestamp metadata.
UPDATE director_mode_state
SET error_at_ms = updated_at_ms
WHERE lifecycle = 'error' AND error_at_ms IS NULL;
