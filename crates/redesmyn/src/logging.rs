use crate::metrics::{EmfInterest, EmfMetrics};
use pyo3::{prelude::*, pyclass, pymethods};
use std::{fmt::Debug, fs::File, io, path::PathBuf};
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
