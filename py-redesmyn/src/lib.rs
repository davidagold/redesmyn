use ::redesmyn::cache::{validate_schedule, Cache, FsClient};
use ::redesmyn::common::Wrap;
use ::redesmyn::error::ServiceError;
use ::redesmyn::handler::{Handler, HandlerConfig};
use ::redesmyn::logging::LogConfig;
use ::redesmyn::predictions::{BatchPredictor, ServiceConfig};
use ::redesmyn::schema::Schema;
use ::redesmyn::server::{Server, ServerHandle};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDelta, PyType};
use std::sync::{Arc, OnceLock};

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
    pub fn from_struct_type(
        _cls: &Bound<'_, PyType>,
        struct_type: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
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
        handler: &Bound<'_, PyAny>,
        batch_max_delay_ms: u32,
        batch_max_size: usize,
    ) -> Self {
        let config = ServiceConfig {
            schema: signature.0.clone().into(),
            path,
            batch_max_delay_ms,
            batch_max_size,
            handler_config: HandlerConfig::Function(handler.clone().unbind()),
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
struct PyServer {
    server: OnceLock<Server>,
    handle: Arc<ServerHandle>,
}

#[pymethods]
impl PyServer {
    #[new]
    pub fn __new__() -> Self {
        let server = Server::new(None);
        let handle = server.handle();
        PyServer {
            server: OnceLock::from(server),
            handle: handle.into(),
        }
    }

    pub fn register(
        &mut self,
        endpoint: PyEndpoint,
        cache_config: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let fs_client = cache_config.getattr("client")?.extract::<FsClient>()?;
        let load_model: Py<_> = cache_config.getattr("load_model")?.clone().unbind();
        let schedule = cache_config.getattr("schedule").ok();
        let interval = cache_config
            .getattr("interval")
            .and_then(|obj| {
                obj.downcast::<PyDelta>()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                    .cloned()
            })
            .ok();
        let max_size: Option<usize> = cache_config.getattr("max_size")?.extract().ok();
        let cache = Cache::new(
            fs_client,
            max_size,
            validate_schedule(schedule, interval)?,
            Some(true),
            load_model,
        );

        let service = BatchPredictor::<String, Schema>::new(endpoint.config, cache.into());
        self.server
            .get_mut()
            .ok_or_else(|| {
                let msg = "Cannot register a service with a server that is already running";
                ServiceError::from(msg)
            })?
            .register(service);
        Ok(())
    }

    pub async fn serve<'py>(&'py mut self) -> PyResult<()> {
        let runtime = TOKIO_RUNTIME.get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("Failed to initialize Tokio Runtime.")
        });
        // TODO: Need to gaurd against any potentially untoward consequences of passing ownership of the
        //       server to the future. For one, we should keep a handle.
        let mut server = self.server.take().ok_or_else(|| {
            PyRuntimeError::new_err("Cannot start server that has previously been started")
        })?;
        runtime
            .spawn(async move { server.serve()?.await.map_err(PyRuntimeError::new_err) })
            .await
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
    }

    pub fn handle(&self) -> PyResult<ServerHandle> {
        Ok(self
            .server
            .get()
            .ok_or_else(|| PyRuntimeError::new_err("Failed to obtain `Server`"))?
            .handle())
    }
}

static TOKIO_RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

#[pymodule]
#[pyo3(name = "py_redesmyn")]
fn redesmyn(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySchema>().unwrap();
    m.add_class::<PyServer>().unwrap();
    m.add_class::<PyEndpoint>().unwrap();
    m.add_class::<Cache>().unwrap();
    m.add_class::<FsClient>().unwrap();
    m.add_class::<LogConfig>().unwrap();
    m.add_class::<ServerHandle>().unwrap();
    Ok(())
}
