-- Legacy DB import support (T-67).
--
-- During the split-codebase period we need a deterministic, queryable mapping
-- from legacy integer primary keys to Rust ULID identifiers.
--
-- This table lives in the Rust DB only; the importer never mutates the legacy DB.

CREATE TABLE legacy_id_map (
    table_name TEXT NOT NULL,
    legacy_id INTEGER NOT NULL,
    new_id BLOB(16) NOT NULL,
    created_at_ms INTEGER NOT NULL,
    PRIMARY KEY (table_name, legacy_id),
    UNIQUE (new_id)
);

CREATE INDEX idx_legacy_id_map_table_name ON legacy_id_map (table_name);
