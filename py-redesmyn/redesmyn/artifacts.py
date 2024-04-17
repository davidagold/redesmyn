import operator
import re
from abc import abstractmethod
from pathlib import Path
from string import Template
from typing import (
    Callable,
    ClassVar,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Self,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    get_args,
)

from annotated_types import SupportsGt, SupportsLt
from more_itertools import filter_map, only
from pydantic import BaseModel, ConfigDict, create_model, model_validator
from pydantic.fields import FieldInfo


def _ensure_trailing_slash(path_component: str) -> str:
    return re.sub(r"^(.*)(?!/)$", r"\1/", path_component)


class PathTemplate(Template):
    delimiter = ""
    idpattern = r"(?a: )"
    braceidpattern = r"(?a:[_a-z][_a-z0-9]*)"


def path(path: str) -> PathTemplate:
    return PathTemplate(path)


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


K = TypeVar("K", bound=Union[SupportsLt, SupportsGt])


class LatestKey(BaseModel, Generic[K]):
    """Designates an :class:`ArtifactSpec` field as the latest key for the latter's storage path.

    A `LatestKey` may optionally specify a manifest filename via the `manifest`
    initialization parameter.

    If a manifest file is specified, then a :class:`ModelCache` that uses
    the `ArtifactSpec` containing the given `LatestKey` will look for the manifest
    file at the path obtained by substituting the manifest filename for
    the cache's storage path parameter corresponding to the latest key
    and omitting any trailing path components.

    If no manifest file is specified, the model cache will list all available
    path components and select the latest using the comparison operator
    specified via the `compare_with` initialization parameter or the default
    comparison operator `operator.gt`.

    See the :class:`ModelCache` and :class:`redesmyn.service.Endpoint` documentation
    for further details concerning how designating a `LatestKey` influences
    the respective behavior.
    """

    manifest: Optional[str] = None
    compare_with: Callable[[K, K], bool] = operator.gt

    # TODO: Include model validator
    @model_validator(mode="after")
    def validate(self) -> Self:
        return self


ModelUri = str
M = TypeVar("M", covariant=True)
LoadFn = Callable[[ModelUri], M]


class ArtifactSpec(BaseModel):
    """A `Pydantic <https://docs.pydantic.dev/latest/>`_ `BaseModel` subclass for declaring and validating artifact specifications.

    Requests made to a :class:`redesmyn.service.Endpoint` may include URL parameters to specify
    which variant of a model to apply to the requesst payload. We use subclasses
    of `ArtifactSpec` to validate URL parameters requested of an endpoint.
    The fields of an `ArtifactSpec` correspond one-to-one with both the endpoint's
    path parameters and the storage location's path parameters. An endpoint declaring
    use of a given `ArtifactSpec` will apply the latter's `BaseModel.model_validate`
    method to the requested URL's path parameters to ensure that the specification
    of the requested model is valid.

    An `ArtifactSpec` may designate one field to serve as a *latest key*.
    A request to an endpoint whose associated `ArtifactSpec` declares a latest key
    may either include a specific value for this field or specify `latest` in the
    path component. See :class:`LatestKey` for more information.

    Since an `ArtifactSpec` derives from `pydantic.BaseModel`, it can include any
    validation mechanism available to the latter. In the following example,
    we use Pydantic validator `annotations <https://docs.pydantic.dev/latest/concepts/validators/#annotated-validators>`_
    and `decorators <https://docs.pydantic.dev/latest/concepts/validators/#field-validators>`_
    to validate prediction requests against a model with variants indexed
    by ISO 3166-1 and ISO 3166-2 codes and a model ID. The URL parameters
    of a request made to an endpoint declaring use of the example spec below
    *must* include a valid combination of supported ISO 3166-1 and ISO 3166-2
    codes and *may either* include a model ID *or* request the latest model via `latest`.

    ..  code-block:: python

        from enum import Enum
        from typing import Annotated, Self

        import mlflow
        from annotated_types import Ge
        from more_itertools import first, one
        from pydantic import BeforeValidator, Field, ValidationInfo, field_validator
        from redesmyn import artifacts as afs


        class IsoCode(Enum):
            @classmethod
            def from_string(cls, v: str) -> Self:
                return one(variant for variant in cls if variant.value == v)


        class Iso3166_1(IsoCode):
            US = "US"
            GB = "GB"


        class Iso3166_2(IsoCode):
            US_CA = "US-CA"
            US_NY = "US-NY"
            GB_ENG = "GB-ENG"
            GB_NIR = "GB-NIR"

            def is_subdivision(self, of: Iso3166_1) -> bool:
                return first(self.value.split("-")) == of.value


        class RegionalModelSpec(afs.ArtifactSpec):
            iso3166_1: Annotated[Iso3166_1, BeforeValidator(Iso3166_1.from_string)]
            iso3166_2: Iso3166_2
            id: Annotated[Optional[int], Ge(0), afs.LatestKey] = None

            # These fields are `ArtifactSpec` class variables.
            # You can omit and pass them to `@afs.artifact_spec` instead.
            load_fn = mlflow.sklearn.load_model
            cache_path = afs.path("s3://model-bucket/{iso3166_1}/{iso3166_2}/{id}/")

            @field_validator("iso3166_2", mode="before")
            @classmethod
            def validate_iso3166_2(cls, v: str, info: ValidationInfo) -> Iso3166_2:
                iso3166_1 = info.data.get("iso3166_1")
                iso3166_2 = Iso3166_2.from_string(v)
                if (
                    not isinstance(iso3166_1, Iso3166_1)
                    or not iso3166_2.is_subdivision(of=iso3166_1)
                ):
                    raise ValueError(f"'{iso3166_2} is not a subdivision of {iso3166_1}")

                return iso3166_2

    **Notes**

    - Requests may also include query parameters, but the latter are generally
      expected to specify optional functionality rather than determine which model
      to apply to a given request. See :class:`redesmyn.service.Endpoint`
      for more information.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    load_fn: ClassVar[LoadFn]
    cache_path: ClassVar[Optional[PathTemplate]]
    """(Optional) The (templated) path that, with parameters substituted, directs to an artifact location.

    You can specify the storage location format for model artifacts either 
    in the definition of an `ArtifactSpec` or in the initialization of a :class:`ModelCache`.
    """
    latest_key: ClassVar[Optional[str]]

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
            return key if key_cls is not None else None

        err = ValueError(f"Too many fields in `{base.__name__}` annotated as `LatestKey`")
        latest_key = only(filter_map(is_latest_key, base.model_fields.items()), too_long=err)

        spec: Type[ArtifactSpec] = create_model(f"{base.__name__}Spec", __base__=(base, ArtifactSpec))
        spec.latest_key = latest_key
        return spec

    def __truediv__(self, other: str) -> EndpointPath:
        return self.as_url_params() / other

    def __rtruediv__(self, other: str) -> EndpointPath:
        return other / self.as_url_params()

    def as_url_params(self) -> EndpointPath:
        return EndpointPath(parts=[f"{{{f}}}" for f in self.model_fields])


def artifact_spec(
    load_fn: Callable[[ModelUri], M],
    cache_path: Optional[str] = None,
    latest_key: Optional[str] = None,
) -> Callable[[Type[BaseModel]], Type[ArtifactSpec]]:
    """Generate an :class:`ArtifactSpec` by decorating a Pydantic `BaseModel` subclass.

    This decorator generates a new class inheriting from both `ArtifactSpec`
    and the decorated class.

    ..  code-block:: python

        import re
        from typing import Annotated, Any

        from annotated_types import Predicate
        from handlers.model import Model
        from pydantic import BaseModel

        from redesmyn import artifacts as afs


        @afs.artifact_spec(
            load_fn=lambda run_id: Model().load(run_id=run_id),
            cache_path="s3://model-bucket/{model_name}/{model_version}/",
        )
        class ModelArtifact(BaseModel):
            @staticmethod
            def validate_version(v: Any) -> bool:
                version_match = re.match(r"v^\d\.\d\.\d", v)
                return True if version_match is not None else False

            model_name: str
            model_version: Annotated[str, Predicate(validate_version)]
            run_id: Annotated[str, afs.LatestKey]
    """

    def wrapper(cls: Type[BaseModel]) -> Type[ArtifactSpec]:
        spec = ArtifactSpec.generate_subclass(cls)
        spec.latest_key = spec.latest_key or latest_key
        spec.load_fn = load_fn

        if not cache_path:
            return spec

        spec.cache_path = PathTemplate(cache_path)
        if errors := set(spec.cache_path.get_identifiers()) ^ set(spec.model_fields):
            msg = f"Mismatch between path template and spec fields: {errors}"
            raise ValueError(msg)

        return spec

    return wrapper


class ArtifactsClient(Generic[M]):
    @abstractmethod
    def _list(self, path: str) -> Iterable[str]:
        pass

    @abstractmethod
    def _get(self, path: str) -> Path:
        pass

    def list(self, path: PathTemplate, **kwargs) -> Iterable[str]:
        return self._list(path.substitute(**kwargs))

    def get(self, path: PathTemplate, **kwargs) -> Path:
        return self._get(path.substitute(**kwargs))


class FsClient(ArtifactsClient):
    def _list(self, path: str) -> Iterable[str]:
        yield from (fp.as_posix() for fp in Path(path).glob("*"))

    def _get(self, path: str) -> Path:
        return Path(path).resolve()


T = TypeVar("T", bound=ArtifactSpec)


class ModelCache(Generic[T, M]):
    """Model cache with asynchronous updating by interval or cron scheduling.

    ..  code-block:: python

        import re
        from typing import Annotated, Any

        import polars as pl
        from annotated_types import Predicate
        from handlers.model import Model
        from pydantic import BaseModel

        from redesmyn import artifacts as afs
        from redesmyn import service as svc

        class Input(svc.Schema):
            a = pl.Float64()
            b = pl.Float64()


        class Output(svc.Schema):
            prediction = pl.Float64()


        @afs.artifact_spec(
            load_fn=lambda run_id: Model().load(run_id=run_id),
            cache_path="s3://model-bucket/{model_name}/{model_version}/",
        )
        class ModelArtifact(BaseModel):
            @staticmethod
            def validate_version(v: Any) -> bool:
                version_match = re.match(r"v^\d\.\d\.\d", v)
                return True if version_match is not None else False

            model_name: str
            model_version: Annotated[str, Predicate(validate_version)]
            run_id: Annotated[str, afs.LatestKey]


        @svc.endpoint(
            path="/predictions/{model_name}/{model_version}",
            cache=afs.ModelCache[ModelArtifact, Model](
                client=afs.FsClient()
            ),
            batch_max_delay_ms=10,
            batch_max_size=64,
        )
        def handler(model: Model, records_df: Input.DataFrame) -> Output.DataFrame:
            return model.predict(records_df=records_df)
    """

    def __init__(
        self,
        client: ArtifactsClient[M],
        spec: Optional[Type[T]] = None,
        path: Optional[str] = None,
        latest_key: Optional[str] = None,
    ) -> None:
        if len(type_param_args := get_args(type(self))) > 0:
            spec, _ = type_param_args
            self._spec = cast(Type[ArtifactSpec], spec)
        elif spec is None or not issubclass(spec, ArtifactSpec):
            msg = f"Argument `spec={spec}` of type `{type(spec)}` is not a subclass of `ArtifactSpec`"
            raise ValueError(msg)
        else:
            self._spec = spec

        if (path is None) ^ (self._spec.cache_path is not None):
            raise ValueError("Precisely one of `path` or `spec.cache_path()` must not be `None`")
        if path and not path.endswith("/"):
            raise ValueError(f"Path must end with '/' (received '{path}').")

        self._client = client
        self._path = self._spec.cache_path or PathTemplate(cast(str, path))
        self._latest_key = self._spec.latest_key or latest_key

    def get(self, **kwargs) -> M:
        self._spec.model_validate(kwargs)
        return self._client.get(self._path, **kwargs)

    def get_latest(self, **kwargs) -> M:
        spec = self._spec.model_validate(kwargs)
        if not ((latest_key := spec.latest_key) is None or (v := getattr(spec, latest_key)) is None):
            msg = f"Cannot fetch latest when `latest_key='{latest_key}' is set to {v}`"
            raise ValueError(msg)

    def referesh(self, all: bool = False, **kwargs):
        pass
