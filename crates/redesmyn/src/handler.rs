use pyo3::types::PyFunction;
use pyo3::{exceptions::PyTypeError, prelude::*};
use pyo3_polars::PyDataFrame;
use tracing::{error, info};

use crate::{
    config_methods,
    error::{PredictionError, ServiceError},
    validate_param,
};

#[derive(Clone, Debug)]
pub enum HandlerConfig {
    PySpec(PySpec),
    Function(Py<PyFunction>),
}

impl From<PySpec> for HandlerConfig {
    fn from(spec: PySpec) -> Self {
        HandlerConfig::PySpec(spec)
    }
}

impl From<Py<PyFunction>> for HandlerConfig {
    fn from(func: Py<PyFunction>) -> Self {
        HandlerConfig::Function(func)
    }
}

#[derive(Debug, Clone, Default)]
pub struct PySpec {
    module: Option<String>,
    class: Option<String>,
    obj: Option<String>,
    method: Option<String>,
}

impl PySpec {
    config_methods! {
        module: &str,
        class: &str,
        obj: &str,
        method: &str
    }

    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Clone, Debug)]
pub struct PyHandler {
    pub handler: Py<PyFunction>,
}

impl From<&PyFunction> for PyHandler {
    fn from(handler: &PyFunction) -> Self {
        PyHandler { handler: handler.into() }
    }
}

impl PyHandler {
    fn get_func(spec: &PySpec, obj: &PyAny) -> PyResult<Py<PyFunction>> {
        let method_name = validate_param!(&spec, method);
        let handler = obj.getattr(method_name.as_str()).inspect_err(|err| {
            error!("Failed to read handler function `{method_name}`: {err}");
        })?;

        match handler.get_type().name()? {
            "function" => handler.extract(),
            // TODO: What if the Python `Endpoint` is configured differently from the native endpoint?
            "Endpoint" => handler.getattr("_handler")?.extract(),
            name => Err(PyTypeError::new_err(format!(
                "Object of type `{name}` cannot be used as handler."
            ))),
        }
    }

    pub fn try_new(config: &HandlerConfig) -> PyResult<Self> {
        let handler: Py<PyFunction> = match config {
            HandlerConfig::PySpec(spec) => Python::with_gil(|py| {
                info!("Importing Python handler with spec {:?}", spec);
                let module_name = validate_param!(&spec, module);
                let module = py.import(module_name.as_str()).inspect_err(|err| {
                    error!("Failed to import handler module `{module_name}`: {err}")
                })?;
                let obj = spec.obj.as_ref().map(|obj_name| module.getattr(obj_name.as_str()));
                match obj {
                    Some(obj) => Self::get_func(spec, obj?),
                    None => Self::get_func(spec, module),
                }
            })?,
            HandlerConfig::Function(func) => func.clone(),
        };
        Ok(PyHandler { handler })
    }

    pub fn invoke(&self, py: Python<'_>, model: Py<PyAny>, df: PyDataFrame) -> PyResult<PyObject> {
        self.handler.call(py, (model, df), None)
    }
}

#[derive(Clone, Debug)]
pub enum Handler {
    Rust,
    Python(PyHandler),
}

impl Handler {
    pub fn invoke(
        &self,
        df: PyDataFrame,
        model: Py<PyAny>,
        py: Option<Python<'_>>,
    ) -> Result<PyObject, ServiceError> {
        match self {
            Handler::Python(pyhandler) => {
                let err = || {
                    PredictionError::Error(
                        "Cannot invoke Python handler without GIL guard.".to_string(),
                    )
                };
                pyhandler.invoke(py.ok_or_else(err)?, model, df).map_err(|err| err.into())
            }
            Handler::Rust => todo!(),
        }
    }
}
