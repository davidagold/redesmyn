from typing import Optional
from redesmyn.py_redesmyn import PySchema
import polars as pl


class Schema:
    _pyschema: PySchema
    
    def __init__(self, pyschema: Optional[PySchema]) -> None:
        self._pyschema = pyschema or PySchema()

    @classmethod
    def from_struct_type(cls, struct_type: pl.Struct) -> "Schema":
        return Schema(pyschema=PySchema.from_struct_type(struct_type))
    
    def __repr__(self) -> str:
        return self._pyschema.as_str()
