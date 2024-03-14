use thiserror::Error;

use polars::prelude::PolarsError;
use pyo3::{exceptions::PyRuntimeError, prelude::PyErr};
use std::env::VarError;
use tokio::{
    sync::{mpsc::error::SendError, oneshot::error::RecvError},
    task::JoinError,
};

#[derive(Error, Debug)]
pub enum ServiceError {
    #[error(transparent)]
    PredictionError(#[from] PredictionError),
    #[error("Environment variable not found: {0}.")]
    VarError(#[from] VarError),
    #[error(transparent)]
    IoError(#[from] std::io::Error),
    #[error(transparent)]
    JoinError(#[from] JoinError),
    #[error("{0}")]
    Error(String),
    #[error("Failed to forward request to prediction service: {0}")]
    SendError(String),
    #[error("Failed to received result: {0}")]
    ReceiveError(#[from] RecvError),
    #[error("Polars operation failed: {0}")]
    ParseError(#[from] PolarsError),
    #[error("Failed to serialize result: {0}")]
    JsonError(#[from] serde_json::Error),
}

impl From<ServiceError> for PyErr {
    fn from(err: ServiceError) -> Self {
        PyRuntimeError::new_err(err.to_string())
    }
}

impl<T> From<SendError<T>> for ServiceError {
    fn from(err: SendError<T>) -> Self {
        Self::SendError(err.to_string())
    }
}

impl<T> From<ServiceError> for Result<T, ServiceError> {
    fn from(err: ServiceError) -> Self {
        Err(err)
    }
}

impl From<PyErr> for ServiceError {
    fn from(err: PyErr) -> Self {
        Self::PredictionError(err.into())
    }
}

#[derive(Error, Debug)]
pub enum PredictionError {
    #[error("Prediction failed: {0}.")]
    Error(String),
    #[error("Prediction failed from Polars operation: {0}.")]
    PolarsError(#[from] polars::prelude::PolarsError),
    #[error("Prediction failed from PyO3 operation: {0}.")]
    PyError(#[from] pyo3::prelude::PyErr),
    #[error("Prediction failed during IO: {0}.")]
    IoError(#[from] std::io::Error),
}
