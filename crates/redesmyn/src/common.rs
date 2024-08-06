pub(crate) type Sized128String = heapless::String<128>;
use crate::error::{ServiceError, ServiceResult};
use pyo3::prelude::*;
use serde::Serialize;
use std::{fmt::Debug, sync::OnceLock};
use tracing::{error, info};

#[macro_export]
macro_rules! do_in {
    (|| $body:block) => {{ (|| -> Option<_> { { $body }.into() })() }};
    (|| -> $ret:ty $body:block) => {{ (|| -> $ret { { $body }.into() })() }};
}

#[macro_export]
macro_rules! config_methods {
    ($($name:ident : $type:ty),*) => {
        $(
            pub fn $name(mut self, $name: $type) -> Self {
                self.$name = Some($name.into());
                self
            }
        )*
    }
}

#[macro_export]
macro_rules! validate_param {
    ($self:expr, $name:ident) => {
        $self.$name.clone().ok_or(ServiceError::Error(format!(
            "Unset required parameter `{}`",
            stringify!($name)
        )))?
    };
}

// Make this a derivable trait, implement on wrapper structs from current type aliases
#[derive(Clone)]
pub struct Wrap<T>(pub T);

impl<T> Wrap<T> {
    pub fn inner<'a>(&'a self) -> &'a T {
        &self.0
    }

    pub fn into_inner(self) -> T {
        self.0
    }

    pub fn inner_mut<'a>(&'a mut self) -> &'a mut T {
        &mut self.0
    }
}

impl<T> From<T> for Wrap<T> {
    fn from(value: T) -> Self {
        Wrap(value)
    }
}

impl<T: Serialize> Serialize for Wrap<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.inner().serialize(serializer)
    }
}

pub fn consume_and_log_err<T, E>(result: Result<T, E>)
where
    E: Debug,
{
    if let Err(err) = result {
        error!("Result of operation failed: {:#?}", err)
    }
}

pub(crate) static TOKIO_RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

pub(crate) fn build_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to initialize Tokio Runtime.")
}

pub fn include_python_paths<'path>(
    paths: impl IntoIterator<Item = &'path str>,
) -> ServiceResult<()> {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let sys = py.import_bound("sys")?;
        let version: String = sys.getattr("version")?.extract()?;
        tracing::info!("Found Python version: {}", version);

        let insert = sys.getattr("path")?.getattr("insert")?;
        for path in paths.into_iter() {
            insert.call((0, path), None)?;
        }

        let pythonpath: Vec<String> = sys.getattr("path")?.extract()?;
        let str_python_path = serde_json::to_string_pretty(&pythonpath).map_err(|err| {
            ServiceError::from(format!("Failed to serialize `sys.path`: {}", err))
        })?;
        info!(pythonpath = format!("{}", str_python_path.as_str()));
        PyResult::<()>::Ok(())
    })
    .map_err(|err| ServiceError::from(format!("Failed to initialize Python process: {}", err)))
}

pub trait __Str__ {
    fn __str__(&self) -> String;
}

impl __Str__ for Py<PyAny> {
    fn __str__(&self) -> String {
        Python::with_gil(|py| self.call_method0(py, "__str__")?.extract::<String>(py))
            .unwrap_or("<Failure calling `__str__` for Python object>".into())
    }
}

impl<'py> __Str__ for Bound<'py, PyAny> {
    fn __str__(&self) -> String {
        do_in!(|| -> PyResult<_> { self.call_method0("__str__")?.extract::<String>() })
            .unwrap_or("<Failure calling `__str__` for Python object>".into())
    }
}

pub trait OkOrLogErr {
    type Mapped;

    fn ok_or_log_err(self) -> Self::Mapped;
}

impl<T, E: Debug> OkOrLogErr for Result<T, E> {
    type Mapped = Option<T>;

    fn ok_or_log_err(self) -> Self::Mapped {
        match self {
            Ok(val) => Some(val),
            Err(err) => {
                error!("Result not OK: {:#?}", err);
                None
            }
        }
    }
}

pub fn from_optional<'py, T: FromPyObject<'py>>(obj: Bound<'py, PyAny>) -> Option<T> {
    if !obj.is_none() { obj.extract::<T>().ok_or_log_err() } else { None }
}
