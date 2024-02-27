use std::iter::repeat;

use polars::datatypes::{AnyValue, DataType};
use serde::{
    de::{DeserializeSeed, MapAccess, Visitor},
    Deserializer,
};
use serde_json::Value;

type ScalarType<'a> = AnyValue<'a>;
type ColumnType<'a> = Vec<ScalarType<'a>>;

fn parse<'a>(dtype: &DataType, v: Value) -> ScalarType<'a> {
    match dtype {
        DataType::Int64 => v.as_i64().map(|v| AnyValue::Int64(v)).unwrap_or(AnyValue::Null),
        DataType::Float64 => v.as_f64().map(|v| AnyValue::Float64(v)).unwrap_or(AnyValue::Null),
        DataType::String => {
            v.as_str().map(move |v| AnyValue::StringOwned(v.into())).unwrap_or(AnyValue::Null)
        }
        _ => todo!(),
    }
}

#[derive(Clone, Debug)]
pub struct Field<'a> {
    pub name: &'a str,
    pub data_type: DataType,
    pub index: u8,
}

#[derive(Clone, Debug)]
pub struct Schema<'a> {
    pub fields: Vec<Field<'a>>,
    pub columns: Vec<ColumnType<'a>>,
}

impl<'a> Schema<'a> {
    pub fn new(fields: Vec<Field<'a>>, capacity: Option<usize>) -> Schema<'a> {
        let column_factory: Box<dyn FnOnce() -> Vec<AnyValue<'a>>> = match capacity {
            Some(capacity) => Box::new(move || ColumnType::<'a>::with_capacity(capacity)),
            None => Box::new(|| ColumnType::<'a>::new())
        };
        let n_fields = &fields.len();
        Schema {
            fields,
            columns: repeat(column_factory()).take(*n_fields).collect(),
        }
    }
}

struct SchemaVisitor<'a>(Schema<'a>);

impl<'de, 'a> Visitor<'de> for SchemaVisitor<'a> {
    type Value = Schema<'a>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a mapping of fields to values")
    }

    fn visit_map<M>(mut self, mut visited: M) -> Result<Schema<'a>, M::Error>
    where
        M: MapAccess<'de>,
    {
        while let Some((key, value)) = visited.next_entry::<&str, Value>()? {
            let Ok(index) = &self.0.fields[..].binary_search_by_key(&key, |f| f.name) else {
                continue;
            };
            self.0
                .columns
                .get_mut(*index)
                .map(|col| col.push(parse(&self.0.fields[*index].data_type, value)));
        }
        Ok(self.0)
    }
}

impl<'de, 'a> DeserializeSeed<'de> for Schema<'a> {
    type Value = Schema<'a>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(SchemaVisitor(self))
    }
}
