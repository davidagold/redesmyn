from io import FileIO
from pathlib import Path
from typing import cast

import mlflow
import polars as pl
from sklearn.pipeline import Pipeline


class SepalLengthPredictor:
    def __init__(self, model_uri: str) -> None:
        self._pipeline: Pipeline = mlflow.sklearn.load_model(model_uri)

    def include_predictions(self, records_df: pl.DataFrame) -> pl.DataFrame:
        predictions = records_df.with_columns(
            prediction=pl.Series(self._pipeline.predict(records_df)),
        )
        return predictions

    @classmethod
    def load_model(cls, loadable: str | Path | bytes | FileIO) -> "SepalLengthPredictor":
        return SepalLengthPredictor(cast(str, loadable))


def handle(model: SepalLengthPredictor, records_df: pl.DataFrame) -> pl.DataFrame:
    return model.include_predictions(records_df=records_df)
