import abc
from typing import Annotated, Optional, Self, Type, cast

import polars as pl

from redesmyn.py_redesmyn import PySchema


class SchemaMeta(abc.ABCMeta):
    @property
    def DataFrame(cls) -> Type[Annotated[pl.DataFrame, Self]]:
        return cast(Type[Annotated[pl.DataFrame, Self]], Annotated[pl.DataFrame, cls])


class Schema(metaclass=SchemaMeta):
    _pyschema: PySchema
    
    def __init__(self, pyschema: Optional[PySchema]) -> None:
        self._pyschema = pyschema or PySchema()

    @classmethod
    def from_struct_type(cls, struct_type: pl.Struct) -> "Schema":
        return Schema(pyschema=PySchema.from_struct_type(struct_type))
    
    def __repr__(self) -> str:
        return self._pyschema.as_str()

    @classmethod
    def to_struct_type(cls) -> pl.Struct:
        schema_dict = {k: v for k, v in vars(cls).items() if issubclass(type(v), pl.DataType)}
        return pl.Struct(schema_dict)
