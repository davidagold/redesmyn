from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from alembic import command
from alembic.script import ScriptDirectory

from redesmyn.db.migrate import alembic_config_for_db


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="redesmyn-migrations-") as tmpdir:
        db_path = Path(tmpdir) / "test.sqlite3"
        cfg = alembic_config_for_db(db_path)
        script = ScriptDirectory.from_config(cfg)

        heads = script.get_heads()
        if len(heads) != 1:
            joined = ", ".join(heads) if heads else "(none)"
            print(
                "error: alembic has multiple heads (rebase/migration conflict likely): "
                f"{joined}",
                file=sys.stderr,
            )
            return 2

        try:
            command.upgrade(cfg, "head")
        except Exception as e:
            print(f"error: alembic upgrade head failed: {e}", file=sys.stderr)
            return 1

    print(f"ok: migrations apply cleanly (head={heads[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
