use crate::metrics::{EmfInterest, EmfMetrics, EmfOutput};
use pyo3::{prelude::*, pyclass, pymethods};
use std::{
    fmt::{Debug, Display},
    fs::File,
    io,
    path::PathBuf,
};
use tracing::info;
use tracing_subscriber::{
    self, layer::Layer, layer::SubscriberExt, prelude::*, registry::LookupSpan, EnvFilter,
};

/// `LogOutput` configures the primary logging output for the application.
/// Logs can be written either to STDOUT by initializing a `LogOutput::Stdout`
/// or to a file by initializing a `LogOutput::File(path: PathBuf)`,
/// where `path` is the path of the file to which to write.
#[derive(Debug, Clone, Default)]
pub enum LogOutput {
    #[default]
    Stdout,
    File(PathBuf),
}

impl Display for LogOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let formatted = match self {
            Self::Stdout => "STDOUT",
            Self::File(path) => path.to_str().unwrap_or("<ERROR>"),
        };
        f.write_fmt(format_args!("LogOutput `{}`", formatted))
    }
}

/// `LogConfig` configures all logging output for the application.
/// This includes specifications for the primary logging output (via `LogOutput`)
/// as well as specification of the optional Amazon Web Services (AWS) Embedded Metrics Format (EMF) metrics
/// (via `EmfOutput`).
#[pyclass]
#[derive(Default)]
pub struct LogConfig {
    output: LogOutput,
    emf_output: Option<EmfOutput>,
}

impl LogConfig {
    /// Instantiate a new `LogOutput` specification.
    pub fn new(output: LogOutput, emf_output: Option<EmfOutput>) -> Self {
        LogConfig { output, emf_output }
    }

    /// Produce a `tracing_subscriber::layer::Layer` from the present `LogConfig`.
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
    fn __new__(
        path: &Bound<'_, PyAny>,
        emf_path: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<LogConfig> {
        let emf_output = match emf_path {
            Some(path) => Some(EmfOutput::new(path.extract()?)),
            None => None,
        };
        Ok(LogConfig::new(LogOutput::File(path.extract::<PathBuf>()?), emf_output))
    }

    pub fn init(&self) -> () {
        println!("Initializing logging");
        tracing_subscriber::registry()
            .with(EnvFilter::from_default_env())
            .with(self.layer().with_filter(EmfInterest::Never))
            .with(match &self.emf_output {
                Some(EmfOutput::File(path)) => {
                    Some(EmfMetrics::new(10, path.to_string_lossy().to_string()))
                }
                None => None,
            })
            .init();
        info!("Initialized logging with main output `{}`", self.output);
    }
}
