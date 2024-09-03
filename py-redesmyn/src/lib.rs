use ::redesmyn::artifacts::FsClient;
use ::redesmyn::cache::{validate_schedule, Cache};
use ::redesmyn::common::{from_optional, OkOrLogErr, Wrap};
use ::redesmyn::error::ServiceError;
use ::redesmyn::handler::Handler;
use ::redesmyn::logging::LogConfig;
use ::redesmyn::predictions::{Endpoint, ServiceConfig};
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
        match struct_type.extract::<Schema>() {
            Ok(schema) => Ok(PySchema { schema }),
            Err(err) => Err(err),
        }
    }

    pub fn as_str(&self) -> String {
        format!("{:#?}", self.schema)
    }
}

#[pyclass]
struct PyEndpoint {
    signature: (Option<Schema>, Option<Schema>),
    config: ServiceConfig,
}

#[pymethods]
impl PyEndpoint {
    #[new]
    #[pyo3(
        signature = (
            signature,
            path,
            handler,
            batch_max_delay_ms = 10,
            batch_max_size = 64,
            validate_artifact_params = false
        )
    )]
    pub fn __new__(
        signature: (Bound<'_, PyAny>, Bound<'_, PyAny>),
        path: String,
        handler: &Bound<'_, PyAny>,
        batch_max_delay_ms: u32,
        batch_max_size: usize,
        validate_artifact_params: bool,
    ) -> Self {
        let schema_in = from_optional::<Schema>(signature.0);
        let schema_out = from_optional::<Schema>(signature.1);

        let config = ServiceConfig {
            schema: schema_in.clone(),
            path,
            batch_max_delay_ms,
            batch_max_size,
            handler: Handler::Python(handler.clone().unbind()),
            validate_artifact_params,
        };
        PyEndpoint {
            signature: (schema_in, schema_out),
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
        endpoint: &PyEndpoint,
        cache_config: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let cache = match cache_config {
            None => None,
            Some(config) => {
                let fs_client: FsClient = config.getattr("client")?.extract::<FsClient>()?;
                let load_model: Py<_> = config.getattr("load_model")?.clone().unbind();
                let schedule = config.getattr("schedule").ok();
                let interval = config
                    .getattr("interval")
                    .and_then(|obj| {
                        obj.downcast::<PyDelta>()
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                            .cloned()
                    })
                    .ok();
                let max_size: Option<usize> = config.getattr("max_size")?.extract().ok();
                let pre_fetch_all: bool = config.getattr("pre_fetch_all")?.extract()?;
                let artifact_spec = config
                    .getattr("spec")
                    .ok_or_log_err()
                    .and_then(|obj| (!obj.is_none()).then_some(obj.unbind()));
                Some(Cache::new(
                    fs_client,
                    max_size,
                    validate_schedule(schedule, interval)?,
                    Some(pre_fetch_all),
                    load_model,
                    artifact_spec,
                ))
            }
        };

        let service = Endpoint::<String, Schema>::new(endpoint.config.clone(), cache.into());
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
        // TODO: Need to guard against any potentially untoward consequences of passing ownership of the
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
