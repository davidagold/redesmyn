use std::path::PathBuf;

use ::redesmyn::common::{Wrap, LogConfig};
use ::redesmyn::handler::{Handler, HandlerConfig};
use ::redesmyn::predictions::{BatchPredictor, ServiceConfig};
use ::redesmyn::schema::Schema;
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::intern;

use ::redesmyn::server::Server;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::{PyFunction, PyType};


#[pyclass]
#[repr(transparent)]
pub struct PySchema {
    pub schema: Schema,
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
struct PyEndpoint {
    signature: (Schema, Schema),
    config: ServiceConfig,
}

#[pymethods]
impl PyEndpoint {
    #[new]
    #[pyo3(signature = (signature, path, handler, batch_max_delay_ms = 10, batch_max_size = 64))]
    pub fn __new__(
        signature: (Wrap<Schema>, Wrap<Schema>),
        path: String,
        handler: &PyFunction,
        batch_max_delay_ms: u32,
        batch_max_size: usize,
    ) -> Self {
        let config = ServiceConfig {
            schema: signature.0.clone().into(),
            path,
            batch_max_delay_ms,
            batch_max_size,
            handler_config: HandlerConfig::Function(handler.into()),
            handler: Some(Handler::Python(handler.into())),
        };
        let (schema_in, schema_out) = signature;
        PyEndpoint {
            signature: (schema_in.0, schema_out.0),
            config,
        }
    }

    pub fn __repr__(&self) -> String {
        let ServiceConfig { path, .. } = self.config.clone();
        format!("Endpoint {{ path: \"{path}\", handler: `{:#?}` }}", self.config.handler)
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
        let mut server = Server::default();
        let mut path: PathBuf = ["logs", "this_run"].iter().collect();
        path.set_extension("txt");
        server.log_config(LogConfig::File(path));
        PyServer { server }
    }

    pub fn register(&mut self, endpoint: PyEndpoint) -> PyResult<()> {
        let service = BatchPredictor::<String, Schema>::new(endpoint.config);
        self.server.register(service);
        Ok(())
    }

    pub fn serve<'py>(&'py mut self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let mut server = self.server.clone();
        pyo3_asyncio::tokio::future_into_py(py, async move {
            server.serve()?.await.map_err(PyRuntimeError::new_err)
        })
    }
}

#[pymodule]
#[pyo3(name = "py_redesmyn")]
fn redesmyn(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySchema>().unwrap();
    m.add_class::<PyServer>().unwrap();
    m.add_class::<PyEndpoint>().unwrap();

    Ok(())
}
