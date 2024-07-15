use std::cell::OnceCell;
use std::path::PathBuf;

use ::redesmyn::cache::{ArtifactsClient, Cache, FsClient};
use ::redesmyn::common::{consume_and_log_err, LogConfig as RsLogConfig, Wrap};
use ::redesmyn::error::ServiceError;
use ::redesmyn::handler::{Handler, HandlerConfig};
use ::redesmyn::predictions::{BatchPredictor, ServiceConfig};
use ::redesmyn::schema::Schema;
use ::redesmyn::server::Server;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyFunction, PyType};
use tracing::{error, info};
use tracing_subscriber::{self, layer::SubscriberExt, prelude::*, EnvFilter};

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
        handler: &Bound<'_, PyFunction>,
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
#[repr(transparent)]
struct PyServer {
    server: OnceCell<Server>,
}

#[pymethods]
impl PyServer {
    #[new]
    pub fn __new__() -> Self {
        let mut server = Server::default();
        let mut path: PathBuf = ["logs", "this_run"].iter().collect();
        path.set_extension("txt");
        server.log_config(RsLogConfig::File(path));
        let cell = OnceCell::new();
        consume_and_log_err(cell.set(server));
        PyServer { server: cell }
    }

    pub fn register(&mut self, endpoint: PyEndpoint, cache_config: Py<PyAny>) -> PyResult<()> {
        let cache = match Python::with_gil(|py| -> PyResult<_> {
            let fs_client = cache_config.getattr(py, "client")?.extract::<FsClient>(py)?;
            let load_model: Py<_> = cache_config
                .getattr(py, "load_model")?
                .downcast_bound::<PyFunction>(py)?
                .clone()
                .unbind();
            let client = ArtifactsClient::FsClient { client: fs_client, load_model };
            Ok(Cache::new(client, None, None, Some(true)))
        }) {
            Ok(cache) => {
                info!("Successfully initialized model cache {}", cache);
                cache
            }
            Err(err) => {
                error!("Failed to initialize model cache: {}", err);
                return Err(err);
            }
        };
        let service = BatchPredictor::<String, Schema>::new(endpoint.config, cache.into());
        self.server
            .get_mut()
            .ok_or_else(|| {
                ServiceError::from(
                    "Cannot register a service with a server that is already running",
                )
            })?
            .register(service);
        Ok(())
    }

    pub async fn serve<'py>(&'py mut self) -> PyResult<()> {
        // TODO: Need to gaurd against any potentially untoward consequences of passing ownership of the
        //       server to the future. For one, we should keep a handle.
        let mut server = self.server.take().ok_or_else(|| {
            PyRuntimeError::new_err("Cannot start server that has previously  been started")
        })?;
        server.serve()?.await.map_err(PyRuntimeError::new_err)
    }
}

#[pyclass]
#[derive(Default)]
struct LogConfig {
    config: OnceCell<RsLogConfig>,
}

#[pymethods]
impl LogConfig {
    #[new]
    fn __new__(path: Py<PyAny>) -> PyResult<LogConfig> {
        let config = LogConfig::default();
        Python::with_gil(|py| {
            config
                .config
                .set(RsLogConfig::File(path.extract::<PathBuf>(py)?))
                .map_err(|_| PyRuntimeError::new_err("Failed to set log config"))?;
            PyResult::Ok(())
        })?;
        Ok(config)
    }

    fn init(&mut self) -> PyResult<()> {
        let config = self
            .config
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Failed to take log config."))?;
        tracing_subscriber::registry()
            .with(EnvFilter::from_default_env())
            .with(config.layer())
            .init();
        Ok(())
    }
}

#[pymodule]
#[pyo3(name = "py_redesmyn")]
fn redesmyn(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySchema>().unwrap();
    m.add_class::<PyServer>().unwrap();
    m.add_class::<PyEndpoint>().unwrap();
    m.add_class::<Cache>().unwrap();
    m.add_class::<FsClient>().unwrap();
    m.add_class::<LogConfig>().unwrap();

    Ok(())
}
