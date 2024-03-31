import inspect
from itertools import islice
from typing import Callable, Tuple, Type, get_args

from more_itertools import first, one
import polars as pl

from redesmyn.py_redesmyn import PyEndpoint
from redesmyn.schema import Schema


def extract_schema(annotation: Type) -> pl.Struct:
    schema_cls = one(
        e for e in islice(get_args(annotation), 1, None) if issubclass(e, Schema)
    )
    return schema_cls.to_struct_type()


def get_signature(f: Callable) -> Tuple[pl.Struct, pl.Struct]:
    s = inspect.signature(f)
    param_df = one(
        param
        for param in s.parameters.values()
        if issubclass(first(get_args(param.annotation)), pl.DataFrame)
    )
    return (extract_schema(param_df.annotation), extract_schema(s.return_annotation))


class Endpoint:
    def __init__(
        self,
        handler: Callable[..., pl.DataFrame],
        signature: Tuple[pl.Struct, pl.Struct],
        path: str,
        batch_max_delay_ms: int,
        batch_max_size: int,
    ) -> None:
        self._handler = handler
        self._pyendpoint = PyEndpoint(
            signature=signature,
            path=path,
            batch_max_delay_ms=batch_max_delay_ms,
            batch_max_size=batch_max_size,
            handler=handler,
        )
    
    def __call__(self, *args, **kawrgs) -> pl.DataFrame:
        return self._handler(*args, **kawrgs)


def endpoint(
    path: str,
    batch_max_delay_ms: int = 10,
    batch_max_size: int = 32,
) -> Callable[[Callable], Endpoint]:
    def wrapper(handler: Callable) -> Endpoint:
        signature = get_signature(handler)
        return Endpoint(
            handler=handler,
            signature=signature,
            path=path,
            batch_max_delay_ms=batch_max_delay_ms,
            batch_max_size=batch_max_size,
        )

    return wrapper
