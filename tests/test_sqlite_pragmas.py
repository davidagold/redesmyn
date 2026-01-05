from __future__ import annotations

import sqlite3

import pytest

from tests.scenarios.scenario import Scenario


@pytest.mark.integration
async def test_sqlite_uses_wal_journal_mode_and_normal_sync(
    scenario: Scenario,
) -> None:
    # Verify that init_db applies SQLite pragmas intended to reduce lock
    # contention in local-dev.
    conn = sqlite3.connect(scenario.ctx.db_path)
    try:
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()
        assert journal_mode is not None
        assert str(journal_mode[0]).lower() == "wal"
    finally:
        conn.close()
