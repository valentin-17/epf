from pathlib import Path
from typing import Optional

import typer
from loguru import logger

from epf import __app_name__, __version__
from epf.config import PREDICTIONS_DIR
from epf.pipeline import EpfPipeline

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

@app.command()
def train(
    model_builder: str  =typer.Option(
        exists = True,
        help= "The model builder to use for training. Options: lstm, cnn, gru.",
    ),
    model_dir: Path = typer.Option(
        default = "models",
        help= "Path to save the trained model.",
    ),
    model_name: str = typer.Option(
        default= "new_model",
        help= "Name of the model to train.",
    ),
) -> None:
    """
    Train the model.


    """
    pipeline = EpfPipeline()
    pipeline.train()

@app.command()
def predict(
    data_path: Path = typer.Argument(
        default= PREDICTIONS_DIR / "samples.csv",
        exists= True,
        help= "Path to file used for prediction. Default is 'samples.csv'",
    ),
    model: Path = typer.Option(
        exists= True,
        help= "Model to use for prediction.",
    ),
    output_dir: Path = typer.Option(
        default = PREDICTIONS_DIR,
        help= "Path to save the predictions.",
    ),
    output_name: str = typer.Option(
        default = "predictions.csv",
        help= "Name of the output file.",
    ),
    overwrite: bool = typer.Option(False, "--overwrite", "-o", help="Overwrite existing files"),
) -> None:
    """
    Predict prices for the given dataset using the specified model.

    :param data_path: Path to file used for prediction.
    :param model: Path to the model to use for prediction.
    :param output_dir: Path to save the predictions.
    :param output_name: Name of the output file.
    :param overwrite: Overwrite existing files.
    """
    if not Path.exists(data_path):
        raise FileNotFoundError(
            f"Data path {data_path} does not exist."
        )
    if not Path.exists(model):
        raise FileNotFoundError(
            f"Model path {model} does not exist."
        )
    if model is None:
        raise ValueError(
            "Model path is required for prediction."
        )
    if Path.exists(output_dir) and not overwrite:
        raise FileExistsError(
            f"Output path {output_dir} already exists. Use -o to overwrite."
        )

    output_path = output_dir / output_name

    pipeline = EpfPipeline()
    pipeline.predict(data_path, model)
    pipeline.save_predictions(output_path)
    # TODO: complete predict command with methods from pipeline

if __name__ == "__main__":
    app()