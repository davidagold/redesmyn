import inspect
from asyncio import Future
from itertools import islice
from typing import Callable, Generic, List, Optional, Self, Tuple, Type, TypeVar, get_args, overload

import polars as pl
from more_itertools import first, one
from typing_extensions import Coroutine

from redesmyn.artifacts import ArtifactSpec, CacheConfig
from redesmyn.py_redesmyn import Cache, PyEndpoint, PyServer, ServerHandle
from redesmyn.schema import Schema


def extract_schema(annotation: Type) -> pl.Struct:
    schema_cls = one(e for e in islice(get_args(annotation), 1, None) if issubclass(e, Schema))
    return schema_cls.to_struct_type()


def get_signature(f: Callable) -> Tuple[pl.Struct, pl.Struct]:
    s = inspect.signature(f)
    print(f"{list(get_args(t.annotation) for t in s.parameters.values())}")
    param_df = one(
        param
        for param in s.parameters.values()
        if (
            len((type_args := get_args(param.annotation))) > 0
            and issubclass(first(type_args), pl.DataFrame)
        )
    )
    return (extract_schema(param_df.annotation), extract_schema(s.return_annotation))


M = TypeVar("M")


class Endpoint(Generic[M]):
    def __init__(
        self,
        handler: Callable[[M, pl.DataFrame], pl.DataFrame],
        signature: Tuple[pl.Struct, pl.Struct],
        path: str,
        cache_config: Optional[CacheConfig[M]],
        batch_max_delay_ms: int,
        batch_max_size: int,
        validate_artifact_params: bool = False,
    ) -> None:
        self._handler = handler
        self._pyendpoint = PyEndpoint(
            signature=signature,
            path=path,
            batch_max_delay_ms=batch_max_delay_ms,
            batch_max_size=batch_max_size,
            handler=handler,
            validate_artifact_params=validate_artifact_params,
        )
        self._cache_config = cache_config

    def __call__(self, model: M, records: pl.DataFrame, **kwargs) -> pl.DataFrame:
        return self._handler(model, records)


@overload
def endpoint(
    path: str,
    *,
    batch_max_delay_ms: int = 10,
    batch_max_size: int = 32,
    validate_artifact_params: bool = False,
) -> Callable[[Callable[[pl.DataFrame], pl.DataFrame]], Endpoint]: ...


@overload
def endpoint(
    path: str,
    cache_config: CacheConfig[M],
    *,
    batch_max_delay_ms: int = 10,
    batch_max_size: int = 32,
    validate_artifact_params: bool = False,
) -> Callable[[Callable[[M, pl.DataFrame], pl.DataFrame]], Endpoint]: ...


def endpoint(
    path: str,
    cache_config: Optional[CacheConfig[M]] = None,
    batch_max_delay_ms: int = 10,
    batch_max_size: int = 32,
    validate_artifact_params: bool = False,
) -> (
    Callable[[Callable[[M, pl.DataFrame], pl.DataFrame]], Endpoint]
    | Callable[[Callable[[pl.DataFrame], pl.DataFrame]], Endpoint]
):
    def wrapper(handler: Callable[..., pl.DataFrame]) -> Endpoint:
        return Endpoint(
            handler=handler,
            signature=get_signature(handler),
            path=path,
            cache_config=cache_config,
            batch_max_delay_ms=batch_max_delay_ms,
            batch_max_size=batch_max_size,
            validate_artifact_params=validate_artifact_params,
        )

    return wrapper


SchemaLike = pl.Struct | Type[Schema]
Signature = Tuple[pl.Struct, pl.Struct]


class Server:
    """An HTTP server.

    Use :py:func:`Server.register` to register endpoints with a given server.

    Run the server with :py:func:`Server.serve`.
    """

    def __init__(self) -> None:
        self._pyserver: PyServer = PyServer()
        self._handle: ServerHandle = self._pyserver.handle()
        self._endpoints: List[Endpoint] = []

    def register(self, endpoint: Endpoint) -> Self:
        """Register an endpoint with the present `Server`.

        ..  code-block:: python

            import polars as pl
            from redesmyn import service as svc
            from sklearn.linear_model import Lasso


            @svc.endpoint(
                ...
            )
            def handler(model: Lasso, records: pl.DataFrame) -> pl.DataFrame:
                return model.predict(records)


            server = svc.Server()
            server.register(handler)
        """
        self._endpoints.append(endpoint)
        self._pyserver.register(endpoint._pyendpoint, endpoint._cache_config)
        return self

    async def serve(self) -> Coroutine:
        """Run the present `Server`.

        ..  code-block:: python

            import asyncio

            from redesmyn import service as svc


            async def main():
                server = svc.Server()
                await server.serve()


            asyncio.run(main())
        """
        return await self._pyserver.serve()

    async def stop(self, graceful: bool = True) -> Coroutine:
        return await self._handle.stop(graceful=graceful)

    def __repr__(self) -> str:
        tab = " " * 4
        repr_endpoints = "\n".join(f"{tab}{endpoint}" for endpoint in self._endpoints)
        repr = f"Endpoints\n{repr_endpoints}"
        return repr
