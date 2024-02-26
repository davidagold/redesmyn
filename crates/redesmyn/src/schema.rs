use std::{any::Any, collections::HashMap};

use arrow::array::Array;
use polars::datatypes::{AnyValue, DataType};
// use polars::prelude::Schema;
use pyo3::prelude::*;
use serde::{
    de::{DeserializeSeed, MapAccess, Visitor},
    ser::Error,
    Deserialize, Deserializer,
};
use serde_json::Value;

type ScalarType<'a> = AnyValue<'a>;
type ColumnType<'a> = Vec<ScalarType<'a>>;


fn parse<'a>(dtype: &DataType, v: Value) -> ScalarType<'a> {
    match dtype {
        DataType::Float64 => v.as_f64().map(|v| AnyValue::Float64(v)).unwrap_or(AnyValue::Null),
        _ => todo!(),
    }
}

#[derive(Clone, Debug)]
pub struct Field {
    pub name: String,
    pub data_type: DataType,
}

#[derive(Clone)]
pub struct Schema {
    pub fields: Vec<Field>,
}

impl Schema {
    pub fn columns<'a>(&self) -> HashMap<String, (DataType, ColumnType<'a>)> {
        let mut columns =
            HashMap::<String, (DataType, ColumnType<'a>)>::with_capacity((&self).fields.len());
        for field in self.fields.iter() {
            columns.insert(field.name.clone(), (field.data_type.clone(), ColumnType::<'a>::new()));
        }
        columns
    }
}


#[derive(Debug)]
pub struct Columns<'a>(pub HashMap<String, (DataType, ColumnType<'a>)>);

impl<'de, 'a> DeserializeSeed<'de> for Columns<'a> {
    type Value = Self;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        pub struct ColumnsVisitor<'a>(Columns<'a>);

        impl<'de, 'a> Visitor<'de> for ColumnsVisitor<'a> {
            type Value = Columns<'a>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a mapping of fields to values")
            }

            fn visit_map<M>(mut self, mut visited: M) -> Result<Self::Value, M::Error>
            where
                M: MapAccess<'de>,
            {
                while let Some((key, value)) = visited.next_entry::<String, Value>()? {
                    self.0.0.get_mut(&key).map(|(dtype, col)| {
                        col.push(parse(dtype, value))
                    });
                }
                Ok(self.0)
            }
        }

        deserializer.deserialize_map(ColumnsVisitor(self))
    }
}
