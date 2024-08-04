import asyncio
import sys
from argparse import ArgumentParser
from datetime import timedelta
from io import FileIO
from pathlib import Path
from typing import Optional, cast

import mlflow
import polars as pl
import redesmyn.artifacts as afs
from pydantic import Field, field_validator
from redesmyn.py_redesmyn import LogConfig
from redesmyn.schema import Schema
from redesmyn.service import Endpoint, Server, endpoint
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tests.common import DIR_PYREDESMYN, DIR_TESTS


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
        return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])

    @classmethod
    def fit(cls, df: pl.DataFrame, pipeline: Optional[Pipeline] = None) -> Pipeline:
        input_cols = [field.name for field in Input.to_struct_type().fields]
        return (pipeline or cls.pipeline()).fit(df[input_cols], df["sepal_length"])

    def include_predictions(self, records_df: Input.DataFrame) -> Output.DataFrame:
        predictions = records_df.with_columns(
            **{Output.field_name(): pl.Series(self._pipeline.predict(records_df))},
        )
        return predictions


class SepalLengthPredictorSpec(afs.ArtifactSpec[SepalLengthPredictor]):
    run_id: str = Field(pattern=r"^\d+$")
    model_id: str

    @classmethod
    def load_model(cls, loadable: str | Path | bytes | FileIO) -> SepalLengthPredictor:
        return SepalLengthPredictor(cast(str, loadable))


def get_handle(**kwargs) -> Endpoint:
    @endpoint(
        path="/predictions/{run_id}/{model_id}",
        batch_max_delay_ms=10,
        batch_max_size=64,
        cache_config=afs.CacheConfig(
            client=afs.FsClient(
                base_path=DIR_TESTS / "fixtures/models/mlflow/iris",
                path_template="/{run_id}/{model_id}/artifacts/model",
            ),
            load_model=lambda *args: SepalLengthPredictor(*args),
            spec=SepalLengthPredictorSpec,
        ),
        **kwargs,
    )
    def handle(model: SepalLengthPredictor, records_df: Input.DataFrame) -> Output.DataFrame:
        return model.include_predictions(records_df=records_df)

    return handle
