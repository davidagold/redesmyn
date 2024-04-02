import abc
from typing import Dict, Generic, Iterable, List, Optional, TypeVar, cast

from more_itertools import last


class EndpointPath:
    def __init__(self, parts: Iterable[str] = [], query: Dict[str, type] = {}) -> None:
        self.parts: List[str] = list(parts)
        self.query = query

    def __truediv__(self, other: str) -> "EndpointPath":
        return EndpointPath(parts=[*self.parts, other], query=self.query)

    def __rtruediv__(self, other: str) -> "EndpointPath":
        return EndpointPath(parts=[other, *self.parts], query=self.query)

    def __str__(self):
        return self.path

    def __repr__(self) -> str:
        query_params = "&".join(f"{k}=<{v.__name__}>" for k, v in self.query.items())
        debug_path = f"{self.path}?{query_params}"
        return f"EndpointPath('{debug_path}')"

    @property
    def path(self) -> str:
        return "/".join(self.parts)

    def with_query_params(self, **query_params: type):
        self.query = self.query | query_params


class ArtifactSpecMeta(abc.ABCMeta):
    def __truediv__(self, other: str) -> EndpointPath:
        # return "/".join([cast(ArtifactSpec, self).as_url_params(), other])
        return cast(ArtifactSpec, self).as_url_params() / other

    def __rtruediv__(self, other: str) -> EndpointPath:
        # return "/".join([other, cast(ArtifactSpec, self).as_url_params()])
        return other / cast(ArtifactSpec, self).as_url_params()


class ArtifactSpec(metaclass=ArtifactSpecMeta):
    def __init__(self, **kwargs: type) -> None:
        self.fields = list(kwargs.keys())

    def as_url_params(self) -> EndpointPath:
        # return "/".join(f"{{{k}}}" for k in get_type_hints(cls))
        return EndpointPath(parts=[f"{{{k}}}" for k in self.fields])

    def __truediv__(self, other: str) -> EndpointPath:
        # return "/".join([cast(ArtifactSpec, self).as_url_params(), other])
        return self.as_url_params() / other

    def __rtruediv__(self, other: str) -> EndpointPath:
        # return "/".join([other, cast(ArtifactSpec, self).as_url_params()])
        return other / self.as_url_params()


T = TypeVar("T")
class ModelCache(Generic[T]):
    def __init__(
        self,
        artifact_spec: ArtifactSpec,
        storage: str,
        latest_key: Optional[str] = None,
    ) -> None:
        self.artifact_spec = artifact_spec
        self.storage = storage
        self.latest_key: str = latest_key or last(self.artifact_spec.fields)

    def get(self, **kwargs) -> T:
        # assert all(isinstance(v, self.artifact_spec))
        errors = set(self.artifact_spec.fields) ^ set(kwargs)
        assert len(errors) == 0, f"{errors=}"
        return
    
    def get_latest(self, **kwargs) -> T:
        # assert all(isinstance(v, self.artifact_spec))
        errors = set([k for k in self.artifact_spec.fields if k != self.latest_key]) ^ set(kwargs)
        assert len(errors) == 0, f"{errors=}"
        return

    def referesh(self, **kwargs):
        pass

    def refresh_all(self, **kwargs):
        pass