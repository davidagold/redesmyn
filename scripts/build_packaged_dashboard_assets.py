from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--copy-only",
        action="store_true",
        help="Skip `npm ci`/`npm run build` and only copy an existing dashboard/dist.",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    dashboard_root = repo_root / "dashboard"
    dashboard_dist = dashboard_root / "dist"
    packaged_dist = repo_root / "redesmyn" / "dashboard_assets" / "dist"

    if not dashboard_root.is_dir():
        raise SystemExit(f"error: dashboard directory not found: {dashboard_root}")

    if not args.copy_only:
        if not shutil.which("npm"):
            raise SystemExit(
                "error: npm not found (required to build dashboard assets)"
            )
        _run(["npm", "ci"], cwd=dashboard_root)
        _run(["npm", "run", "build"], cwd=dashboard_root)

    index_html = dashboard_dist / "index.html"
    if not index_html.is_file():
        raise SystemExit(
            f"error: missing {index_html} (run `cd dashboard && npm run build`)"
        )

    if packaged_dist.exists():
        shutil.rmtree(packaged_dist)
    packaged_dist.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(dashboard_dist, packaged_dist)
    print(f"ok: copied {dashboard_dist} -> {packaged_dist}")


if __name__ == "__main__":
    main()
