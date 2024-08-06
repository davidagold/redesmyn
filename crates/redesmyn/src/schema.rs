use std::sync::Arc;

use crate::common::{OkOrLogErr, Sized128String, Wrap};
use crate::error::{ServiceError, ServiceResult};
use heapless::{self, FnvIndexMap};
use indexmap::IndexMap;
use polars::prelude::NamedFrom;
use polars::{
    datatypes::{AnyValue, DataType},
    frame::DataFrame,
    series::Series,
};
use pyo3::exceptions::PyTypeError;
use pyo3::{intern, FromPyObject, PyAny, PyResult};
use serde::{
    de::{DeserializeSeed, MapAccess, Visitor},
    Deserializer,
};
use serde_json::Value;
use tracing::error;

pub trait Relation {
    type Serialized;

    fn schema(rel: Option<&Self>) -> Option<Schema>;

    fn parse(
        records: Vec<Self::Serialized>,
        schema: Option<&Arc<Schema>>,
    ) -> Result<DataFrame, ServiceError>
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

    pub fn columns(&self, capacity: Option<usize>) -> Columns<'_> {
        self.fields
            .iter()
            .map(|field| {
                let raw_values = match capacity {
                    Some(capacity) => ColumnValues::<'_>::with_capacity(capacity),
                    None => ColumnValues::<'_>::new(),
                };
                (field.name.as_str(), Column { field: field.clone(), raw_values })
            })
            .collect()
    }

    pub fn add_field(&mut self, name: &str, data_type: DataType) -> &Self {
        let index = self.len();
        // TODO: Handle possible error
        let _ = self.fields.push(Field {
            name: Sized128String::try_from(name).unwrap(),
            data_type,
            index,
        });
        self
    }

    // pub fn parse_untyped(records: Vec<String>) -> ServiceResult<DataFrame> {
    //     // TODO: Deserialize directly into `columns` with custom `DeserializeSeed` impl
    //     let mut columns = IndexMap::<String, Vec<f64>>::new();
    //     for record_str in records {
    //         let record = match serde_json::from_str::<IndexMap<String, f64>>(record_str.as_str()) {
    //             Ok(record) => record,
    //             Err(err) => {
    //                 error!("Failed to deserialize record `{record_str}`: {err}");
    //                 continue;
    //             }
    //         };
    //         for (key, val) in record.into_iter() {
    //             columns.entry(key).or_insert_with(|| Vec::new()).push(val);
    //         }
    //     }
    //     DataFrame::new(
    //         columns.into_iter().map(|(field, col)| Series::new(field.as_str(), col)).collect(),
    //     )
    //     .map_err(ServiceError::from)
    // }
}

impl Relation for Schema {
    type Serialized = String;

    fn schema(rel: Option<&Self>) -> Option<Schema> {
        rel.cloned()
    }

    fn parse(
        records: Vec<<Self as Relation>::Serialized>,
        schema: Option<&Arc<Schema>>,
    ) -> Result<DataFrame, ServiceError>
    where
        Self: Sized,
    {
        if let Some(schema) = schema {
            let series = records
                .iter()
                .fold(schema.columns(None), |mut columns, record| {
                    let mut de = serde_json::Deserializer::from_str(record);
                    // TODO: Handle
                    let _ = ColumnsWrapper(&mut columns).deserialize(&mut de);
                    columns
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
        } else {
            let mut columns = IndexMap::<String, Vec<f64>>::new();
            for record_str in records {
                let record =
                    match serde_json::from_str::<IndexMap<String, f64>>(record_str.as_str()) {
                        Ok(record) => record,
                        Err(err) => {
                            error!("Failed to deserialize record `{record_str}`: {err}");
                            continue;
                        }
                    };
                for (key, val) in record.into_iter() {
                    columns.entry(key).or_insert_with(|| Vec::new()).push(val);
                }
            }
            DataFrame::new(
                columns.into_iter().map(|(field, col)| Series::new(field.as_str(), col)).collect(),
            )
            .map_err(ServiceError::from)
        }
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
            if let Some(col) = self.0.get_mut(key) {
                col.push(col.parse(value))
            }
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

impl From<Wrap<Schema>> for Schema {
    fn from(wrapped: Wrap<Schema>) -> Self {
        wrapped.0
    }
}

impl FromPyObject<'_> for Schema {
    fn extract(ob: &'_ PyAny) -> PyResult<Self> {
        let py = ob.py();
        let name = ob.get_type().name()?;
        if name != "Struct" {
            return Err(PyTypeError::new_err(format!(
                "Cannot convert object of type `{name}` into Schema"
            )));
        };
        let mut schema = Schema::default();
        let fields = ob.getattr(intern!(py, "fields"))?.extract::<Vec<&PyAny>>()?;
        for field in fields {
            let field_name = field.getattr(intern!(py, "name"))?.extract::<&str>()?;
            let dtype = field.getattr("dtype")?;

            let dtype =
                dtype.get_type().name().and_then(|name| get_dtype_from_name(name.as_ref()))?;
            schema.add_field(field_name, dtype);
        }

        // Ok(Wrap(schema))
        Ok(schema)
    }
}

fn get_dtype_from_name(dtype_name: &str) -> PyResult<DataType> {
    match dtype_name {
        "Int8" => Ok(DataType::Int8),
        "Int16" => Ok(DataType::Int16),
        "Int32" => Ok(DataType::Int32),
        "Int64" => Ok(DataType::Int64),
        "UInt8" => Ok(DataType::UInt8),
        "UInt16" => Ok(DataType::UInt16),
        "UInt32" => Ok(DataType::UInt32),
        "UInt64" => Ok(DataType::UInt64),
        "String" => Ok(DataType::String),
        "Binary" => Ok(DataType::Binary),
        "Boolean" => Ok(DataType::Boolean),
        "Float32" => Ok(DataType::Float32),
        "Float64" => Ok(DataType::Float64),
        dt => Err(PyTypeError::new_err(format!("'{dt}' is not a Polars data type",))),
    }
}
