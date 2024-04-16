from abc import abstractmethod
import re
from string import Template
from typing import (
    Callable,
    ClassVar,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    get_args,
)
from more_itertools import filter_map, only

from pydantic import BaseModel, ConfigDict, create_model
from pydantic.fields import FieldInfo


def _ensure_trailing_slash(path_component: str) -> str:
    return re.sub(r"^(.*)(?!/)$", r"\1/", path_component)


class PathTemplate(Template):
    delimiter = ""
    idpattern = r"(?a: )"
    braceidpattern = r"(?a:[_a-z][_a-z0-9]*)"


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


class LatestKey:
    pass


class ArtifactSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _cache_path: ClassVar[Optional[PathTemplate]]
    _latest_key: ClassVar[Optional[str]]

    @classmethod
    def latest_key(cls) -> Optional[str]:
        return cls._latest_key

    def has_latest(self) -> bool:
        latest_key = self.latest_key()
        return latest_key is not None and getattr(self, latest_key) is None

    @classmethod
    def cache_path(cls) -> Optional[PathTemplate]:
        return cls._cache_path

    @classmethod
    def from_path(cls, path: str | PathTemplate, spec_name: str) -> Type["ArtifactSpec"]:
        param_ids = (
            path.get_identifiers()
            if isinstance(path, PathTemplate)
            else PathTemplate(path).get_identifiers()
        )
        params = {k: (Optional[str], None) for k in param_ids}
        return create_model(spec_name, __base__=(ArtifactSpec,), **params)

    @staticmethod
    def generate_subclass(base: Type[BaseModel]) -> Type["ArtifactSpec"]:
        def is_latest_key(key_field_tuple: Tuple[str, FieldInfo]) -> Optional[str]:
            key, field = key_field_tuple
            key_cls = only(
                arg
                for arg in get_args(field.rebuild_annotation())
                if isinstance(arg, Type) and issubclass(arg, LatestKey)
            )
            if key_cls is not None:
                return key
            else:
                return None

        latest_key = only(filter_map(is_latest_key, base.model_fields.items()))
        spec: Type[ArtifactSpec] = create_model(f"{base.__name__}Spec", __base__=(base, ArtifactSpec))
        spec._latest_key = latest_key
        return spec

    def __truediv__(self, other: str) -> EndpointPath:
        return self.as_url_params() / other

    def __rtruediv__(self, other: str) -> EndpointPath:
        return other / self.as_url_params()

    def as_url_params(self) -> EndpointPath:
        return EndpointPath(parts=[f"{{{f}}}" for f in self.model_fields])


def spec(
    cache_path: Optional[str] = None, latest_key: Optional[str] = None
) -> Callable[[Type[BaseModel]], Type[ArtifactSpec]]:
    def wrapper(cls: Type[BaseModel]) -> Type[ArtifactSpec]:
        spec = ArtifactSpec.generate_subclass(cls)
        spec._latest_key = spec._latest_key or latest_key
        if cache_path:
            spec._cache_path = PathTemplate(cache_path)
            if errors := set(spec._cache_path.get_identifiers()) ^ set(spec.model_fields):
                msg = f"Mismatch between path template and spec fields: {errors}"
                raise ValueError(msg)

        return spec

    return wrapper


M = TypeVar("M", covariant=True)


class ArtifactsClient(Generic[M]):
    @abstractmethod
    def _list(self, path: str) -> Iterable[str]:
        pass

    @abstractmethod
    def _get(self, path: str) -> M:
        pass

    def list(self, path: PathTemplate, **kwargs) -> Iterable[str]:
        return self._list(path.substitute(**kwargs))

    def get(self, path: PathTemplate, **kwargs) -> M:
        return self._get(path.substitute(**kwargs))


T = TypeVar("T", bound=Type[ArtifactSpec])


class ModelCache(Generic[T, M]):
    def __init__(
        self,
        client: ArtifactsClient[M],
        spec: T,
        path: Optional[str] = None,
        latest_key: Optional[str] = None,
    ) -> None:
        if (path is None) ^ (spec.cache_path() is not None):
            raise ValueError("Precisely one of `path` or `spec.cache_path()` must not be `None`")
        if path and not path.endswith("/"):
            raise ValueError(f"Path must end with '/' (received '{path}').")
        if not issubclass(spec, ArtifactSpec):
            raise ValueError()

        self._client = client
        self._path = cast(PathTemplate, PathTemplate(path) if path is not None else spec.cache_path())
        self._spec = spec
        self._latest_key = latest_key

    def get(self, **kwargs) -> M:
        self._spec.model_validate(kwargs)
        return self._client.get(self._path, **kwargs)

    def get_latest(self, **kwargs) -> M:
        params = self._spec.model_validate(kwargs)
        if (latest_key := params.latest_key()) is not None:
            v = getattr(params, latest_key)
            if (v := getattr(params, latest_key)) is not None:
                msg = f"Cannot fetch latest when `latest_key='{latest_key}' is set to {v}`"
                raise ValueError(msg)

    def referesh(self, **kwargs):
        pass

    def refresh_all(self, **kwargs):
        pass
