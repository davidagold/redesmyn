use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::{common::__Str__, error::ServiceError};

#[derive(Clone, Debug)]
pub enum Handler {
    Rust,
    Python(Py<PyAny>),
}

impl std::fmt::Display for Handler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str_repr = match self {
            Self::Rust => unimplemented!(),
            Self::Python(handler) => handler.__str__(),
        };
        f.write_str(str_repr.as_str())
    }
}

impl TryFrom<Py<PyAny>> for Handler {
    type Error = ServiceError;

    fn try_from(obj: Py<PyAny>) -> Result<Self, Self::Error> {
        Python::with_gil(|py| match obj.bind(py).is_callable() {
            true => Ok(Handler::Python(obj)),
            false => Err(ServiceError::from(format!(
                "Object of type {} is not callable",
                obj.bind(py).get_type().__str__()
            ))),
        })
    }
}

impl Handler {
    /// Invoke the specified handler function with Python arguments determined by the endpoint configuration.
    /// If the endpoint is configured to use a model cache, this function passes arguments of Python signature
    /// `(M, polars.DataFrame)`, where `M` is the parameter of the cache's respective `ArtifactSpec`.
    /// If the endpoint is configured not to use a model cache, this function passes arguments of Python signature
    /// `(polars.DataFrame)`.
    pub(crate) fn invoke(
        &self,
        df: PyDataFrame,
        model: Option<Py<PyAny>>,
    ) -> Result<PyObject, ServiceError> {
        match self {
            Handler::Python(handler_fn) => {
                Python::with_gil(|py| {
                    let result = match model {
                        // NOTE: See https://github.com/davidagold/redesmyn/issues/74
                        Some(model) => handler_fn.call_bound(py, (model, df), None),
                        None => handler_fn.call_bound(py, (df,), None),
                    };
                    result.map_err(ServiceError::from)
                })
            }
            Handler::Rust => todo!(),
        }
    }
}
