from __future__ import annotations

import re

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


async def migrate_agent_status_to_stopped(conn: AsyncConnection) -> None:
    # SQLite can't alter CHECK constraints in place, so when the Enum-backed
    # constraint still references `'idle'` we rebuild the table to update it.
    rows = (
        await conn.exec_driver_sql(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='agents'"
        )
    ).fetchall()
    create_sql = rows[0][0] if rows and rows[0][0] else None

    if create_sql is not None and "'idle'" in create_sql and "'stopped'" not in create_sql:
        index_rows = (
            await conn.exec_driver_sql(
                "SELECT sql FROM sqlite_master "
                "WHERE type='index' AND tbl_name='agents' AND sql IS NOT NULL"
            )
        ).fetchall()
        index_sql = [row[0] for row in index_rows if row and row[0]]

        table_sql = create_sql
        table_sql = re.sub(
            r'^(CREATE\s+TABLE\s+)("?)agents("?)',
            r"\1agents_new",
            table_sql,
            count=1,
        )
        table_sql = table_sql.replace("'idle'", "'stopped'")

        column_rows = (await conn.exec_driver_sql("PRAGMA table_info(agents)")).fetchall()
        columns = [row[1] for row in column_rows if row and row[1]]
        if not columns:
            return

        quoted_cols = ", ".join(f'"{c}"' for c in columns)
        select_cols: list[str] = []
        for col in columns:
            if col == "status":
                select_cols.append(
                    'CASE WHEN "status" = \'idle\' THEN \'stopped\' ELSE "status" END AS "status"'
                )
            else:
                select_cols.append(f'"{col}"')

        insert_sql = (
            f"INSERT INTO agents_new ({quoted_cols}) "
            f"SELECT {', '.join(select_cols)} FROM agents"
        )

        await conn.exec_driver_sql("PRAGMA foreign_keys=OFF")
        try:
            await conn.exec_driver_sql(table_sql)
            await conn.exec_driver_sql(insert_sql)
            await conn.exec_driver_sql("DROP TABLE agents")
            await conn.exec_driver_sql("ALTER TABLE agents_new RENAME TO agents")
            for stmt in index_sql:
                await conn.exec_driver_sql(stmt)
        finally:
            await conn.exec_driver_sql("PRAGMA foreign_keys=ON")

    await conn.execute(text("UPDATE agents SET status='stopped' WHERE status='idle'"))

