use heapless;
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

use crate::error::ServiceError;

type Size4String = heapless::String<128>;

#[derive(Clone, Debug)]
pub struct Field<'a> {
    pub name: &'a str,
    pub data_type: DataType,
    pub index: usize,
}

type ScalarType<'a> = AnyValue<'a>;
type ColumnValues<'a> = Vec<ScalarType<'a>>;
struct Column<'a> {
    pub field: Field<'a>,
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

    fn extend(&mut self, other: Column<'a>) {
        self.raw_values.extend(other.raw_values)
    }
}

type Columns<'a> = Vec<Column<'a>>;

#[derive(Clone, Debug, Default)]
pub struct Schema<'a> {
    pub fields: heapless::Vec<Field<'a>, 256>,
}

impl<'a> Schema<'a> {
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    pub fn columns(&'a self, capacity: Option<usize>) -> Columns<'a> {
        self.fields
            .iter()
            .map(|field| {
                let raw_values = match capacity {
                    Some(capacity) => ColumnValues::<'a>::with_capacity(capacity),
                    None => ColumnValues::<'a>::new(),
                };
                // (field.name, Column { field: field.clone(), values })
                Column { field: field.clone(), raw_values }
            })
            .collect()
    }

    pub fn add_field(mut self, name: &'a str, data_type: DataType) -> Self {
        let index = self.len();
        // TODO: Handle possible error
        self.fields.push(Field { name, data_type, index });
        self
    }
}

pub fn dataframe_from_records(
    schema: &Schema,
    records: Vec<&str>,
) -> Result<DataFrame, ServiceError> {
    let series = records
        .into_par_iter()
        .fold(
            || schema.columns(None),
            |mut columns, record| {
                let mut de = serde_json::Deserializer::from_str(record);
                match ColumnsWrapper(&mut columns).deserialize(&mut de) {
                    _ => columns,
                }
            },
        )
        .reduce(
            || schema.columns(None),
            |mut acc, x| {
                acc.iter_mut().zip(x).for_each(|(col, other)| col.extend(other));
                acc
            },
        )
        .iter()
        .filter_map(|col| {
            // TODO: Handle better.
            let dtype = &col.field.data_type.clone();
            match Series::from_any_values_and_dtype(col.field.name, &col.raw_values, dtype, true) {
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
            let Ok(index) =
                self.0.binary_search_by_key(&Size4String::try_from(key).unwrap(), |col| {
                    Size4String::try_from(col.field.name).unwrap()
                })
            else {
                continue;
            };
            let col: &mut Column = self.0.get_mut(index).unwrap();
            col.push(col.parse(value))
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
