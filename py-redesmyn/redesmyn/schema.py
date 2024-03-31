import abc
from typing import Annotated, Optional, Self, Type, cast

import polars as pl

from redesmyn.py_redesmyn import PySchema


class SchemaMeta(abc.ABCMeta):
    @property
    def DataFrame(cls) -> Type[Annotated[pl.DataFrame, Self]]:
        return cast(Type[Annotated[pl.DataFrame, Self]], Annotated[pl.DataFrame, cls])


class Schema(metaclass=SchemaMeta):
    """Base class to facilitate handler signature specification.

    .. code-block:: python

        import polars as pl
        from redesmyn.schema import Schema


        class Iris(Schema):
            sepal_length = pl.Float64()
            sepal_width = pl.Float64()


        class Output(Schema):
            species = pl.Categorical()


        @endpoint(path="/predictions/iris/species/{model_version})
        def handler(model, records: Iris.DataFrame) -> Output.DataFrame:
            return model.predict(records)

    """

    def __init__(self, pyschema: Optional[PySchema]) -> None:
        self._pyschema = pyschema or PySchema()

    @classmethod
    def from_struct_type(cls, struct_type: pl.Struct) -> "Schema":
        """Generate a `Schema` object from a `polars.Struct` object."""
        return Schema(pyschema=PySchema.from_struct_type(struct_type))

    def __repr__(self) -> str:
        return self._pyschema.as_str()

    @classmethod
    def to_struct_type(cls) -> pl.Struct:
        """Generate a `polars.Struct` object from a `Schema` subclass."""
        schema_dict = {
            k: v for k, v in vars(cls).items() if issubclass(type(v), pl.DataType)
        }
        return pl.Struct(schema_dict)
