from __future__ import annotations

import typer

from redesmyn import __version__

app = typer.Typer(add_completion=False, help="Redesmyn CLI (`rn`).")


@app.command()
def version() -> None:
    print(__version__)


def main() -> None:
    app()

