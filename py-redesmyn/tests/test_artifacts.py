from enum import Enum
from typing import Annotated, Any, Optional, Self, cast
from annotated_types import Ge
from more_itertools import first, one
from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
)
import pytest

from redesmyn import artifacts as afs


class DummyClient(afs.ArtifactsClient):
    def __init__(self) -> None:
        pass

    def _list(self, path: str) -> afs.Iterable[str]:
        pass

    def _get(self, path: str) -> Any:
        print(f"{path=}")


class TestEndpointPath:
    def test_with_query_params(self):
        assert False

        # path.with_query_params(use_feature=bool)
        # assert (
        #     path.__repr__()
        #     == "EndpointPath('/predictions/{source}/{region}/{id}?use_feature=<bool>')"
        # )


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


@afs.spec(cache_path="s3://model-bucket/{iso3166_1}/{iso3166_2}/{id}")
class DummyArtifact(BaseModel):
    iso3166_1: Annotated[Iso3166_1, BeforeValidator(Iso3166_1.from_string)]
    iso3166_2: Iso3166_2
    id: Annotated[int, Ge(0), afs.LatestKey] = Field(default=None)

    @field_validator("iso3166_2", mode="before")
    @classmethod
    def validate_iso3166_2(cls, v: str, info: ValidationInfo) -> Iso3166_2:
        iso3166_1: Optional[Iso3166_1] = info.data.get("iso3166_1")
        iso3166_2 = Iso3166_2.from_string(v)
        if iso3166_1 is None or not iso3166_2.is_subdivision(of=iso3166_1):
            raise ValueError(f"'{iso3166_2} is not a subdivision of {iso3166_1}")

        return iso3166_2


class TestArtifactSpec:
    def test(self):
        path = "s3://model-bucket/{iso3166_1}/{iso3166_2}/{id}"
        assert DummyArtifact.cache_path() is not None
        assert (
            cast(afs.PathTemplate, DummyArtifact.cache_path()).template
            == afs.PathTemplate(path).template
        )


class TestModelCache:
    client = DummyClient()
    path = "s3://model-bucket/{iso3166_1}/{iso3166_2}/{id}/"

    def test_get(self):
        cache = afs.ModelCache(client=self.client, spec=DummyArtifact, latest_key="id")
        cache.get(iso3166_1="US", iso3166_2="US-CA", id=123)

    def test_get_latest(self):
        print(f"{DummyArtifact.latest_key()=}")
        cache = afs.ModelCache(client=self.client, spec=DummyArtifact, latest_key="id")
        cache.get_latest(iso3166_1="US", iso3166_2="US-CA")
        with pytest.raises(Exception):
            cache.get_latest(iso3166_1="US", iso3166_2="US-CA", id=123)

    def test_invalid_iso3166_1(self):
        cache = afs.ModelCache(client=self.client, spec=DummyArtifact, latest_key="id")
        with pytest.raises(ValidationError):
            cache.get_latest(iso3166_1="FR", iso3166_2="FR-17", id=123)

    def test_invalid_iso3166_2(self):
        cache = afs.ModelCache(client=self.client, spec=DummyArtifact, latest_key="id")
        with pytest.raises(ValidationError) as e:
            cache.get_latest(iso3166_1="US", iso3166_2="GB-ENG", id=123)

        error = one(e.value.errors())
        assert error["msg"] == "Value error, 'Iso3166_2.GB_ENG is not a subdivision of Iso3166_1.US"

    def test_invalid_id(self):
        cache = afs.ModelCache(client=self.client, spec=DummyArtifact, latest_key="id")
        with pytest.raises(ValidationError) as e:
            cache.get(iso3166_1="US", iso3166_2="US-CA", id=-1)

        error = one(e.value.errors())
        assert error["msg"] == "Input should be greater than or equal to 0"

    def test_path_mismatch(self):
        with pytest.raises(ValueError) as e:

            @afs.spec("s3://model-bucket/{iso3166_1}/{iso3166_2}/{id}/")
            class MismatchedSpec(BaseModel):
                iso3166_1: Annotated[Iso3166_1, BeforeValidator(Iso3166_1.from_string)]
                iso3166_2: Annotated[Iso3166_1, BeforeValidator(Iso3166_2.from_string)]

        assert e.exconly() == "ValueError: Mismatch between path template and spec fields: {'id'}"
