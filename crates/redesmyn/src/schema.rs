use std::{collections::HashMap, iter::repeat, time::Instant};

use polars::{
    datatypes::{AnyValue, DataType},
    frame::DataFrame,
    series::Series,
};
use serde::{
    de::{DeserializeSeed, MapAccess, Visitor}, Deserializer
};
use serde_json::Value;

use crate::error::ServiceError;



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
    pub field: &'a Field<'a>,
    pub values: ColumnValues<'a>
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


type Columns<'a> = HashMap<&'a str, Column<'a>>;

#[derive(Clone, Debug)]
#[derive(Default)]
pub struct Schema<'a> {
    pub fields: Vec<Field<'a>>,
}

impl<'a> Schema<'a> {
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    pub fn columns(&'a self, capacity: Option<usize>) -> Columns<'a> {
        let mut columns = Columns::<'a>::new();
        for field in self.fields.iter() {
            let values = match capacity {
                Some(capacity) => ColumnValues::<'a>::with_capacity(capacity),
                None => ColumnValues::<'a>::new()
            };
            columns.insert(field.name, Column{field, values});
        }
        columns
    }

    pub fn add_field(mut self, name: &'a str, data_type: DataType) -> Self {
        let index = self.len() + 1;
        self.fields.push(Field { name, data_type, index });
        self
    }
}

pub fn dataframe_from_records(schema: &Schema, records: Vec<&str>) -> Result<DataFrame, ServiceError> {
    let start = Instant::now();
    let mut columns = &mut schema.columns(None);
    let wrapped = &mut InnerWrapper(&mut columns);
    
    for record in records {
        let mut de = serde_json::Deserializer::from_str(record);
        wrapped.deserialize(&mut de);
    }

    let series = &columns
        .iter()
        .filter_map(|(col_name, col)| {
            // TODO: Handle better.
            match Series::from_any_values_and_dtype(col_name, &col.values, &col.field.data_type, true) {
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
// type VisitorInner<'a> = &'a mut Schema<'a>;
type VisitorInner<'a> = Columns<'a>;

struct InnerWrapper<'w, 'cols: 'w>(&'w mut VisitorInner<'cols>);
struct SchemaVisitor<'a, 'cols: 'a>(&'a mut VisitorInner<'cols>);

impl<'de> Visitor<'de> for &mut SchemaVisitor<'_, '_> {
    // type Value = VisitorInner<'a>;
    type Value = ();

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a mapping of fields to values")
    }

    fn visit_map<M>(self, mut visited: M) -> Result<<Self as Visitor<'de>>::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        while let Some((key, value)) = visited.next_entry::<&str, Value>()? {
            let Some(col) = self.0.get_mut(key) else {
                continue
            };
            col.push(col.parse(value))
        }
        // Ok(self.0)
        Ok(())
    }
}

impl<'de, 'v: 'de, 'w: 'v> DeserializeSeed<'de> for &'v mut InnerWrapper<'w, '_> {
    // type Value = VisitorInner<'a>;
    type Value = ();

    // fn deserialize<D>(mut self, deserializer: D) -> Result<&'a mut Schema<'a>, D::Error>
    fn deserialize<D>(self, deserializer: D) -> Result<<Self as DeserializeSeed<'de>>::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mut visitor = SchemaVisitor::<'v, '_>(self.0);
        deserializer.deserialize_map(&mut visitor)
    }
}

