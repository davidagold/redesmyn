CREATE TABLE events (
    id BLOB(16) PRIMARY KEY NOT NULL,
    created_at_ms INTEGER NOT NULL,
    kind TEXT NOT NULL,
    payload BLOB NOT NULL
);

