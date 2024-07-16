use ::redesmyn::cache::{ArtifactsClient, Cache, FsClient, Schedule};
use ::redesmyn::common::{consume_and_log_err, LogConfig as RsLogConfig, Wrap};
use ::redesmyn::error::ServiceError;
use ::redesmyn::handler::{Handler, HandlerConfig};
use ::redesmyn::predictions::{BatchPredictor, ServiceConfig};
use ::redesmyn::schema::Schema;
use ::redesmyn::server::Server;

use chrono::Duration;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyFunction, PyType};
use std::cell::OnceCell;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::OnceLock;
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

    pub fn register(
        &mut self,
        endpoint: PyEndpoint,
        cache_config: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let fs_client = cache_config.getattr("client")?.extract::<FsClient>()?;
        let load_model: Py<_> =
            cache_config.getattr("load_model")?.downcast::<PyFunction>()?.clone().unbind();
        let schedule = cache_config.getattr("schedule").ok().and_then(|obj| {
            (!obj.is_none()).then(|| {
                let sched_str = obj.call_method0("as_str").ok()?.extract::<String>().ok();
                cron::Schedule::from_str(sched_str?.as_str()).ok()
            })?
        });
        let interval = cache_config
            .getattr("interval")
            .ok()
            .and_then(|obj| (!obj.is_none()).then(|| obj.extract::<Duration>().ok())?);
        let sched = match (schedule, interval) {
            (Some(cron_schedule), None) => Some(Schedule::Cron(cron_schedule)),
            (None, Some(duration)) => Some(Schedule::Interval(duration)),
            (None, None) => None,
            _ => {
                return Err(PyRuntimeError::new_err(
                    "At most one of `schedule` or `interval` may be specified",
                ));
            }
        };
        let max_size: Option<usize> = cache_config.getattr("max_size")?.extract().ok();
        let client = ArtifactsClient::FsClient { client: fs_client, load_model };
        let cache = Cache::new(client, max_size, sched, Some(true));

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
}

static TOKIO_RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

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
