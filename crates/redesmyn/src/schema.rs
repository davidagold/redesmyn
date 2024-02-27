use std::iter::repeat;

use polars::{
    datatypes::{AnyValue, DataType},
    frame::DataFrame,
    series::Series,
};
use serde::{
    de::{DeserializeSeed, MapAccess, Visitor},
    Deserializer,
};
use serde_json::Value;

use crate::error::ServiceError;

type ScalarType<'a> = AnyValue<'a>;
type ColumnType<'a> = Vec<ScalarType<'a>>;

fn parse<'a>(dtype: &DataType, v: Value) -> ScalarType<'a> {
    match dtype {
        DataType::Int64 => v.as_i64().map(AnyValue::Int64).unwrap_or(AnyValue::Null),
        DataType::Float64 => v.as_f64().map(AnyValue::Float64).unwrap_or(AnyValue::Null),
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
    pub index: usize,
}

#[derive(Clone, Debug)]
#[derive(Default)]
pub struct Schema<'a> {
    pub fields: Vec<Field<'a>>,
    pub columns: Vec<ColumnType<'a>>,
}

impl<'a> Schema<'a> {
    pub fn new(fields: Vec<Field<'a>>, capacity: Option<usize>) -> Schema<'a> {
        let column_factory: Box<dyn FnOnce() -> Vec<AnyValue<'a>>> = match capacity {
            Some(capacity) => Box::new(move || ColumnType::<'a>::with_capacity(capacity)),
            None => Box::new(ColumnType::<'a>::new),
        };
        let n_fields = &fields.len();
        Schema {
            fields,
            columns: repeat(column_factory()).take(*n_fields).collect(),
        }
    }

    fn len(&self) -> usize {
        assert!(self.fields.len() == self.columns.len());
        self.fields.len()
    }

    fn add_field(mut self, name: &'a str, data_type: DataType) -> Self {
        let index = self.len() + 1;
        self.fields.push(Field { name, data_type, index });
        self.columns.push(Vec::new());
        self
    }

    pub fn dataframe_from_records(self, records: Vec<&'a str>) -> Result<DataFrame, ServiceError> {
        let fields = self.fields.clone();
        let series = records
            .into_iter()
            .fold(Ok(self), |schema, record| {
                let mut de = serde_json::Deserializer::from_str(record);
                schema?.deserialize(&mut de)
            })
            .map_err(|err| ServiceError::Error(err.to_string()))?
            .columns
            .iter()
            .zip(fields)
            .filter_map(|(col, field)| {
                match Series::from_any_values_and_dtype(field.name, col, &field.data_type, true) {
                    Ok(series) => Some(series),
                    Err(err) => {
                        println!("Error; {err}");
                        None
                    }
                }
            })
            .collect::<Vec<_>>();

        DataFrame::new(series).map_err(Into::into)
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
            if let Some(col) = self.0
                .columns
                .get_mut(*index) { col.push(parse(&self.0.fields[*index].data_type, value)) }
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
