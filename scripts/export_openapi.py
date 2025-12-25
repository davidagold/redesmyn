from __future__ import annotations

# ruff: noqa: E402

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.openapi.utils import get_openapi

from redesmyn import __version__
from redesmyn.api import app


def main() -> None:
    spec = get_openapi(title=app.title, version=__version__, routes=app.routes)
    out_path = ROOT / "openapi" / "openapi.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
