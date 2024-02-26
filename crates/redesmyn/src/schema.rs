use std::{any::Any, collections::HashMap};

use arrow::array::Array;
use polars::datatypes::AnyValue;
// use polars::prelude::Schema;
use pyo3::prelude::*;
use serde::{
    de::{DeserializeSeed, MapAccess, Visitor},
    ser::Error,
    Deserialize, Deserializer,
};
use serde_json::Value;
// use polars::prelude::{Field};

#[derive(Clone, Debug, Deserialize)]
pub enum DataType {
    // Int16,
    // Int32,
    // F32,
    F64,
    // String,
}

impl DataType {
    fn parse<'a>(&self) -> impl Fn(Value) -> AnyValue<'a> {
        match self {
            // Self::Int16 => |v: Value| AnyValue::UInt16(v.into()),
            // Self::Int32 => Box::new(arrow::array::Int32Array::from(Vec::<i32>::new())),
            Self::F64 => |v: Value| AnyValue::Float64(v.as_f64().unwrap()),
            // Self::Int32 => Box::new(Vec::<i32>::new()),
            // Self::F32 => Box::new(Vec::<f32>::new()),
            // Self::F64 => Box::new(Vec::<f64>::new()),
            // Self::String => Box::new(Vec::<String>::new()),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Field {
    pub name: String,
    pub data_type: DataType,
}

#[derive(Clone)]
pub struct Schema {
    pub fields: Vec<Field>,
}

impl Schema {
    pub fn columns(&self) -> HashMap<String, (DataType, Vec<AnyValue>)> {
        let mut columns =
            HashMap::<String, (DataType, Vec<AnyValue>)>::with_capacity((&self).fields.len());
        for field in self.fields.iter() {
            columns.insert(field.name.clone(), (field.data_type.clone(), Vec::<AnyValue>::new()));
        }
        columns
    }
}

#[derive(Debug)]
pub struct Columns<'a>(pub HashMap<String, (DataType, Vec<AnyValue<'a>>)>);

impl<'de, 'a> DeserializeSeed<'de> for Columns<'a> {
    type Value = Self;

    fn deserialize<D>(mut self, deserializer: D) -> Result<Self::Value, D::Error>
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
                // let mut columns = self.columns();
                while let Some((key, value)) = visited.next_entry::<String, Value>()? {
                    println!("{key}: {:#?}", value);
                    self.0.0.get_mut(&key).map(|(dtype, col)| {
                        println!("{:#?}", *col);
                        col.push(dtype.parse()(value))
                    });
                }
                Ok(self.0)
            }
        }

        deserializer.deserialize_map(ColumnsVisitor(self))
    }
}
