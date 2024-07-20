pub(crate) type Sized128String = heapless::String<128>;
use crate::metrics::{EmfInterest, EmfMetrics};
use pyo3::{prelude::*, pyclass, pymethods};
use serde::Serialize;
use std::{fmt::Debug, fs::File, io, path::PathBuf, sync::OnceLock};
use tracing::error;
use tracing_subscriber::{
    self, layer::Layer, layer::SubscriberExt, prelude::*, registry::LookupSpan, EnvFilter,
};

#[derive(Debug, Clone, Default)]
pub enum LogOutput {
    #[default]
    Stdout,
    File(PathBuf),
}

#[pyclass]
#[derive(Default)]
pub struct LogConfig {
    output: LogOutput,
    enable_emf: bool,
}

impl LogConfig {
    pub fn new(output: LogOutput, enable_emf: Option<bool>) -> Self {
        LogConfig {
            output,
            enable_emf: enable_emf.unwrap_or(false),
        }
    }

    pub fn layer<S>(&self) -> Box<dyn Layer<S> + Send + Sync + 'static>
    where
        S: tracing::Subscriber,
        for<'a> S: LookupSpan<'a>,
    {
        let layer = tracing_subscriber::fmt::layer().json();
        match &self.output {
            LogOutput::Stdout => Box::new(layer.with_writer(io::stdout)),
            LogOutput::File(path) => {
                println!("Creating log file at {}", path.to_string_lossy());
                let file = File::create(path).expect("Failed to create log file.");
                Box::new(layer.with_writer(file))
            }
        }
    }
}

#[pymethods]
impl LogConfig {
    #[new]
    fn __new__(path: &Bound<'_, PyAny>, enable_emf: Option<bool>) -> PyResult<LogConfig> {
        Ok(LogConfig::new(LogOutput::File(path.extract::<PathBuf>()?), enable_emf))
    }

    pub fn init(&mut self) -> () {
        tracing_subscriber::registry()
            .with(EnvFilter::from_default_env())
            .with(self.layer().with_filter(EmfInterest::Never))
            .with(self.enable_emf.then(|| EmfMetrics::new(10, "./metrics.log".into())))
            .init();
    }
}

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
