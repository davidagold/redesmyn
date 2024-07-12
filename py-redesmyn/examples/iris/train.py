import logging
from io import FileIO
import os
from pathlib import Path
from typing import cast, Optional

import mlflow
import polars as pl
from redesmyn.artifacts import ArtifactSpec
from redesmyn.schema import Schema
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline

from common import load_irises, project_dir


class Input(Schema):
    sepal_width: pl.Float64
    petal_length: pl.Float64
    petal_width: pl.Float64


class Output(Schema):
    sepal_length: pl.Float64

    @staticmethod
    def field_name() -> str:
        return "prediction"


class SepalLengthPredictor:
    def __init__(self, model_uri: str) -> None:
        self._pipeline: Pipeline = mlflow.sklearn.load_model(model_uri)

    @classmethod
    def pipeline(cls) -> Pipeline:
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "scaler",
                    StandardScaler(),
                    ["sepal_width", "petal_length", "petal_width"],
                ),
            ],
            remainder="drop",
        )
        regressor = Lasso(alpha=0.001)
        return Pipeline(
            steps=[("preprocessor", preprocessor), ("regressor", regressor)]
        )

    @classmethod
    def fit(cls, df: pl.DataFrame, pipeline: Optional[Pipeline] = None) -> Pipeline:
        input_cols = [field.name for field in Input.to_struct_type().fields]
        return (pipeline or cls.pipeline()).fit(df[input_cols], df["sepal_length"])

    def include_predictions(self, records_df: Input.DataFrame) -> Output.DataFrame:
        predictions = records_df.with_columns(
            **{Output.field_name(): pl.Series(self._pipeline.predict(records_df))},
        )
        return predictions

class SepalLengthPredictorSpec(ArtifactSpec[SepalLengthPredictor]):
    # model_version: str
    model_id: str

    @classmethod
    def load_model(cls, loadable: str | Path | bytes | FileIO) -> SepalLengthPredictor:
        return SepalLengthPredictor(cast(str, loadable))


def main():
    irises = load_irises()
    mlflow.set_tracking_uri(project_dir() / "models/mlflow/iris")
    experiment_id = (
        experiment.experiment_id
        if (experiment := mlflow.get_experiment_by_name(name="iris"))
        else mlflow.create_experiment(name="iris")
    )
    run_id: str
    with mlflow.start_run(experiment_id=experiment_id):
        assert (active_run := mlflow.active_run()) is not None
        run_id = active_run.info.run_id
        fit = SepalLengthPredictor.fit(df=irises)
        mlflow.sklearn.log_model(fit, artifact_path="model")


if __name__ == "__main__":
    main()
