import sys
from typing import Optional

import typer
from loguru import logger

from epf import __app_name__, __version__

app = typer.Typer()

def _version_callback(version: bool):
    if version:
        typer.echo(f"{__app_name__} v{__version__}")
        raise typer.Exit()

@app.callback()
def main(
    version: Optional[bool] = typer.Option(
            None,
            "--version",
            "-v",
            callback=_version_callback,
            is_eager=True,
            help="Show app version.",
        ),
) -> None:
    return

if __name__ == "__main__":
    app()