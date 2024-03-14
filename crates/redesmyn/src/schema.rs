use crate::common::Sized128String;
use crate::error::ServiceError;
use heapless::{self, FnvIndexMap};
use polars::prelude::*;
use polars::{
    datatypes::{AnyValue, DataType},
    frame::DataFrame,
    series::Series,
};
use rayon::prelude::*;
use serde::{
    de::{DeserializeSeed, MapAccess, Visitor},
    Deserializer,
};
use serde_json::Value;

pub trait Relation {
    type Serialized;

    fn schema(rel: Option<&Self>) -> Option<Schema>;

    fn parse(records: Vec<Self::Serialized>, schema: &Schema) -> Result<DataFrame, ServiceError>
    where
        Self: Sized;
}

#[derive(Clone, Debug)]
pub struct Field {
    // pub name: &'a str,
    pub name: Sized128String,
    pub data_type: DataType,
    pub index: usize,
}

type ScalarType<'a> = AnyValue<'a>;
type ColumnValues<'a> = Vec<ScalarType<'a>>;
pub struct Column<'a> {
    pub field: Field,
    pub raw_values: ColumnValues<'a>,
}

impl<'a> Column<'a> {
    #[inline]
    fn parse(&self, v: Value) -> ScalarType<'a> {
        match self.field.data_type {
            DataType::Int64 => match v {
                Value::Number(v) => {
                    v.as_i64().map(AnyValue::Int64).unwrap_or_else(|| AnyValue::Null)
                }
                _ => AnyValue::Null,
            },
            DataType::Float64 => match v {
                Value::Number(v) => {
                    v.as_f64().map(AnyValue::Float64).unwrap_or_else(|| AnyValue::Null)
                }
                _ => AnyValue::Null,
            },
            DataType::String => match v {
                Value::String(v) => AnyValue::StringOwned(v.into()),
                _ => AnyValue::Null,
            },
            _ => todo!(),
        }
    }
    fn push(&mut self, v: ScalarType<'a>) {
        self.raw_values.push(v)
    }
}

const MAX_FIELDS: usize = 256;

// type Columns<'a> = Vec<Column<'a>>;
type Columns<'a> = FnvIndexMap<&'a str, Column<'a>, MAX_FIELDS>;

#[derive(Clone, Debug, Default)]
pub struct Schema {
    pub fields: heapless::Vec<Field, MAX_FIELDS>,
}

impl Schema {
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    pub fn columns<'a>(&'a self, capacity: Option<usize>) -> Columns<'a> {
        self.fields
            .iter()
            .map(|field| {
                let raw_values = match capacity {
                    Some(capacity) => ColumnValues::<'a>::with_capacity(capacity),
                    None => ColumnValues::<'a>::new(),
                };
                (field.name.as_str(), Column { field: field.clone(), raw_values })
            })
            .collect()
    }

    pub fn add_field(&mut self, name: &str, data_type: DataType) -> &Self {
        let index = self.len();
        // TODO: Handle possible error
        self.fields.push(Field {
            name: Sized128String::try_from(name).unwrap(),
            data_type,
            index,
        });
        self
    }
}

impl Relation for Schema {
    type Serialized = String;

    fn schema(rel: Option<&Self>) -> Option<Schema> {
        match rel {
            Some(schema) => Some(schema.clone()),
            None => None,
        }
    }

    fn parse(records: Vec<<Self as Relation>::Serialized>, schema: &Schema) -> Result<DataFrame, ServiceError>
    where
        Self: Sized,
    {
        let series = records
            .iter()
            .fold(schema.columns(None), |mut columns, record| {
                let mut de = serde_json::Deserializer::from_str(&record);
                match ColumnsWrapper(&mut columns).deserialize(&mut de) {
                    _ => columns,
                }
            })
            .into_iter()
            .filter_map(|(_, col)| {
                // TODO: Handle better.
                let dtype = &col.field.data_type.clone();
                match Series::from_any_values_and_dtype(
                    &col.field.name,
                    &col.raw_values,
                    dtype,
                    true,
                ) {
                    Ok(series) => Some(series),
                    Err(err) => {
                        println!("Error; {err}");
                        None
                    }
                }
            })
            .collect::<Vec<_>>();

        DataFrame::new(series.to_vec()).map_err(Into::into)
    }
}

struct ColumnsWrapper<'w, 'cols: 'w>(&'w mut Columns<'cols>);
struct SchemaVisitor<'v, 'cols: 'v>(&'v mut Columns<'cols>);

impl<'v, 'de, 'cols> Visitor<'de> for &mut SchemaVisitor<'v, 'cols>
where
    'cols: 'v,
    'de: 'cols,
{
    type Value = ();

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a mapping of fields to values")
    }

    fn visit_map<M>(self, mut visited: M) -> Result<<Self as Visitor<'de>>::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        while let Some((key, value)) = visited.next_entry::<&str, Value>()? {
            self.0.get_mut(key).map(|col| col.push(col.parse(value)));
        }
        Ok(())
    }
}

impl<'v, 'w, 'de, 'cols> DeserializeSeed<'de> for &'v mut ColumnsWrapper<'w, 'cols>
where
    'de: 'cols,
{
    type Value = ();

    fn deserialize<D>(
        self,
        deserializer: D,
    ) -> Result<<Self as DeserializeSeed<'de>>::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mut visitor = SchemaVisitor::<'v, 'cols>(self.0);
        deserializer.deserialize_map(&mut visitor)
    }
}
