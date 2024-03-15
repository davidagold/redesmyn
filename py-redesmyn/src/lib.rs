use futures_util::future::TryFutureExt;
use ::redesmyn::predictions::{BatchPredictor, Configurable, Service, ServiceConfig};
use ::redesmyn::schema::Schema;
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::intern;

use ::redesmyn::server::{Serve, Server};
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::PyType;

struct Wrap<T>(T);

#[pyclass]
#[repr(transparent)]
pub struct PySchema {
    pub schema: Schema,
}

impl FromPyObject<'_> for Wrap<Schema> {
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

            let dtype = dtype.get_type().name().and_then(get_dtype_from_name)?;
            schema.add_field(field_name, dtype);
        }

        Ok(Wrap(schema))
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
        dt => return Err(PyTypeError::new_err(format!("'{dt}' is not a Polars data type",))),
    }
}

#[pymethods]
impl PySchema {
    #[new]
    pub fn __init__() -> Self {
        PySchema { schema: Schema::default() }
    }

    #[classmethod]
    pub fn from_struct_type(_cls: &PyType, struct_type: &PyAny) -> PyResult<Self> {
        match struct_type.extract::<Wrap<Schema>>() {
            Ok(wrapped) => Ok(PySchema { schema: wrapped.0 }),
            Err(err) => Err(err),
        }
    }

    pub fn as_str(&self) -> String {
        format!("{:#?}", self.schema)
    }
}

#[pyclass]
#[derive(Clone)]
struct Endpoint {
    signature: (Schema, Schema),
    config: ServiceConfig,
}

#[pymethods]
impl Endpoint {
    #[new]
    #[pyo3(signature = (signature, path, handler, batch_max_delay_ms = 10, batch_max_size = 50))]
    pub fn __new__(
        signature: (Wrap<Schema>, Wrap<Schema>),
        path: String,
        handler: String,
        batch_max_delay_ms: u32,
        batch_max_size: usize,
    ) -> Self {
        let config = ServiceConfig::default()
            .path(path)
            .py_handler(handler)
            .batch_max_delay_ms(batch_max_delay_ms)
            .batch_max_size(batch_max_size);
        let (schema_in, schema_out) = signature;
        Endpoint {
            signature: (schema_in.0, schema_out.0),
            config,
        }
    }

    pub fn __repr__(&self) -> String {
        // let str_config = format!("{:#?}", self.config);
        let ServiceConfig {
            path,
            batch_max_delay_ms,
            batch_max_size,
            py_handler,
        } = self.config.clone();
        format!("Endpoint {{ path: \"{path}\", handler: `{py_handler}` }}")
    }
}

#[pyclass]
#[repr(transparent)]
struct PyServer {
    server: Server,
}

#[pymethods]
impl PyServer {
    #[new]
    pub fn __new__() -> Self {
        PyServer { server: Server::default() }
    }

    pub fn register(&mut self, endpoint: Endpoint) -> PyResult<()> {
        let (schema_in, _) = endpoint.signature;
        let service = BatchPredictor::<String, Schema>::new(schema_in);
        self.server.register(service);
        Ok(())
    }

    pub fn serve<'py>(&'py mut self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let mut server = self.server.clone();
        pyo3_asyncio::tokio::future_into_py(py, async move {
            server.serve()?.await.map_err(|err| PyRuntimeError::new_err(err))
        })

    }
}

#[pymodule]
#[pyo3(name = "py_redesmyn")]
fn redesmyn(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySchema>().unwrap();
    m.add_class::<PyServer>().unwrap();
    m.add_class::<Endpoint>().unwrap();

    Ok(())
}
