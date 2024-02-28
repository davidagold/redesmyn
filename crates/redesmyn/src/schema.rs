use std::{time::Instant};

use heapless;
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

type Size4String = heapless::String<4>;

#[derive(Clone, Debug)]
pub struct Field<'a> {
    pub name: &'a str,
    pub data_type: DataType,
    pub index: usize,
}

type ScalarType<'a> = AnyValue<'a>;
// type ColumnType<'a> = Vec<ScalarType<'a>>;
type ColumnValues<'a> = Vec<ScalarType<'a>>;
struct Column<'a> {
    pub field: Field<'a>,
    pub values: ColumnValues<'a>,
}

impl<'a> Column<'a> {
    fn parse(&self, v: Value) -> ScalarType<'a> {
        match self.field.data_type {
            DataType::Int64 => v.as_i64().map(AnyValue::Int64).unwrap_or(AnyValue::Null),
            DataType::Float64 => v.as_f64().map(AnyValue::Float64).unwrap_or(AnyValue::Null),
            DataType::String => {
                v.as_str().map(move |v| AnyValue::StringOwned(v.into())).unwrap_or(AnyValue::Null)
            }
            _ => todo!(),
        }
    }

    fn push(&mut self, v: ScalarType<'a>) {
        self.values.push(v)
    }
}

// type Columns<'a> = HashMap<&'a str, Column<'a>>;
type Columns<'a> = heapless::Vec<Column<'a>, 256>;

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
                let values = match capacity {
                    Some(capacity) => ColumnValues::<'a>::with_capacity(capacity),
                    None => ColumnValues::<'a>::new(),
                };
                Column { field: field.clone(), values }
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
    let start = Instant::now();
    let mut columns = &mut schema.columns(None);
    let wrapped = &mut InnerWrapper(columns);

    for record in records {
        let mut de = serde_json::Deserializer::from_str(record);
        wrapped.deserialize(&mut de);
    }

    let series = columns
        .iter()
        .filter_map(|col| {
            // TODO: Handle better.
            match Series::from_any_values_and_dtype(
                col.field.name,
                &col.values,
                &col.field.data_type,
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

    let df = DataFrame::new(series.to_vec()).map_err(Into::into);
    println!("{:#?}", start.elapsed());
    df
}

// struct SchemaVisitor<'a>(&'a mut Schema<'a>);
// type Inner<'a> = &'a mut Schema<'a>;
type Inner<'cols> = Columns<'cols>;

struct InnerWrapper<'w, 'cols: 'w>(&'w mut Inner<'cols>);
struct SchemaVisitor<'v, 'cols: 'v>(&'v mut Inner<'cols>);

impl<'de, 'v: 'de, 'cols: 'v> Visitor<'de> for &mut SchemaVisitor<'v, 'cols> {
    // type Value = Inner<'a>;
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

impl<'de, 'v: 'de, 'w: 'de, 'cols: 'w> DeserializeSeed<'de> for &'v mut InnerWrapper<'w, 'cols> {
    // type Value = Inner<'a>;
    type Value = ();

    // fn deserialize<D>(mut self, deserializer: D) -> Result<&'a mut Schema<'a>, D::Error>
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
