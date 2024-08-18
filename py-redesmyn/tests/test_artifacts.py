import math
import time
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from io import FileIO
from pathlib import Path
from typing import Annotated, Any, DefaultDict, Iterable, List, Self, cast

import mlflow
import pytest
from annotated_types import Ge, Predicate
from more_itertools import first, one, pairwise
from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
)
from redesmyn import artifacts as afs
from redesmyn.py_redesmyn import LogConfig


class RegionalModel:
    _load_timestamps_by_uri: DefaultDict[str, List[datetime]] = defaultdict(list)

    @classmethod
    def load(cls, uri: str) -> None:
        cls._load_timestamps_by_uri[uri].append(datetime.now())


class FromString(Enum):
    @classmethod
    def from_string(cls, v: str) -> Self:
        return one(variant for variant in cls if variant.value == v)


class Iso3166_1(FromString):
    US = "US"
    GB = "GB"


class Iso3166_2(FromString):
    US_CA = "US-CA"
    US_NY = "US-NY"
    GB_ENG = "GB-ENG"
    GB_NIR = "GB-NIR"

    def is_subdivision(self, of: Iso3166_1) -> bool:
        return first(self.value.split("-")) == of.value


# You can apply the :function:`artifact_spec` decorator to a class that
# inherits from `pydantic.BaseModel` to achieve the
# @afs.artifact_spec(
#     load_fn=mlflow.sklearn.load_model,
#     cache_path="s3://model-bucket/{iso3166_1}/{iso3166_2}/{id}",
# )
class RegionalModelSpec(afs.ArtifactSpec[None]):
    # These fields are parts of the artifact specification.
    iso3166_1: Annotated[Iso3166_1, BeforeValidator(Iso3166_1.from_string)]
    iso3166_2: Iso3166_2
    id: Annotated[int, Ge(0), afs.LatestKey] = Field(default=None)

    @field_validator("iso3166_2", mode="before")
    @classmethod
    def validate_iso3166_2(cls, v: str, info: ValidationInfo) -> Iso3166_2:
        iso3166_1 = info.data.get("iso3166_1")
        iso3166_2 = Iso3166_2.from_string(v)
        if not isinstance(iso3166_1, Iso3166_1) or not iso3166_2.is_subdivision(of=iso3166_1):
            raise ValueError(f"'{iso3166_2} is not a subdivision of {iso3166_1}")

        return iso3166_2


# class TestArtifactSpec:
#     def test(self):
#         path = "s3://model-bucket/{iso3166_1}/{iso3166_2}/{id}"
#         assert RegionalModelSpec.cache_path is not None
#         assert (
#             cast(afs.PathTemplate, RegionalModelSpec.cache_path).template
#             == afs.PathTemplate(path).template
#         )


# class TestModelCache:
#     client = DummyClient()
#     path = afs.path("s3://model-bucket/{iso3166_1}/{iso3166_2}/{id}/")

#     def test_init_with_type_annotation(self):
#         _ = afs.ModelCache[RegionalModelSpec, object](
#             client=self.client, path=self.path, spec=RegionalModelSpec[object]
#         )

#     def test_get(self):
#         cache = afs.ModelCache(client=self.client, path=self.path, spec=RegionalModelSpec)
#         cache.get(iso3166_1="US", iso3166_2="US-CA", id=123)

#     def test_get_latest(self):
#         print(f"{RegionalModelSpec.latest_key=}")
#         cache = afs.ModelCache(client=self.client, path=self.path, spec=RegionalModelSpec)
#         cache.get_latest(iso3166_1="US", iso3166_2="US-CA")
#         with pytest.raises(Exception):
#             cache.get_latest(iso3166_1="US", iso3166_2="US-CA", id=123)

#     def test_invalid_iso3166_1(self):
#         cache = afs.ModelCache(client=self.client, path=self.path, spec=RegionalModelSpec)
#         with pytest.raises(ValidationError):
#             cache.get_latest(iso3166_1="FR", iso3166_2="FR-17", id=123)

#     def test_invalid_iso3166_2(self):
#         cache = afs.ModelCache(client=self.client, path=self.path, spec=RegionalModelSpec)
#         with pytest.raises(ValidationError) as e:
#             cache.get_latest(iso3166_1="US", iso3166_2="GB-ENG", id=123)

#         error = one(e.value.errors())
#         assert error["msg"] == "Value error, 'Iso3166_2.GB_ENG is not a subdivision of Iso3166_1.US"

#     def test_invalid_id(self):
#         cache = afs.ModelCache(client=self.client, path=self.path, spec=RegionalModelSpec)
#         with pytest.raises(ValidationError) as e:
#             cache.get(iso3166_1="US", iso3166_2="US-CA", id=-1)

#         error = one(e.value.errors())
#         assert error["msg"] == "Input should be greater than or equal to 0"

#     def test_path_mismatch(self):
#         with pytest.raises(ValueError) as e:

#             @afs.artifact_spec(
#                 load_fn=mlflow.sklearn.load_model,
#                 cache_path="s3://model-bucket/{iso3166_1}/{iso3166_2}/{id}/",
#             )
#             class MismatchedSpec(BaseModel):
#                 iso3166_1: Annotated[Iso3166_1, BeforeValidator(Iso3166_1.from_string)]
#                 iso3166_2: Annotated[Iso3166_1, BeforeValidator(Iso3166_2.from_string)]

#         assert e.exconly() == "ValueError: Mismatch between path template and spec fields: {'id'}"


class TestCache:
    def test_schedule(self):
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        refresh_interval_seconds = 2
        cache = afs.Cache(
            client=afs.FsClient(
                base_path=Path(__file__).parent / "fixtures",
                path_template="models/{iso_3166_1}/{iso_3166_2}",
            ),
            load_model=lambda x: RegionalModel.load(x),
            interval=timedelta(seconds=refresh_interval_seconds),
            pre_fetch_all=True,
        )
        cache.start()
        time.sleep(5)

        expected_uris = ["models/us/us-ca"]
        for uri in expected_uris:
            assert uri in RegionalModel._load_timestamps_by_uri
            assert len(RegionalModel._load_timestamps_by_uri[uri]) >= 2
            lags = [y - x for x, y in pairwise(RegionalModel._load_timestamps_by_uri[uri])]
            # Look into this
            # print(lags)
            # assert all(math.isclose(lag.total_seconds(), refresh_interval_seconds) for lag in lags)
