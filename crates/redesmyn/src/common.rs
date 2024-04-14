pub(crate) type Sized128String = heapless::String<128>;
use std::{fs::File, io, path::PathBuf};

// pub(crate) type Sized256String = heapless::String<256>;
// use tracing::{error, info};
use tracing_subscriber::{layer::Layer, registry::LookupSpan};

#[derive(Debug, Clone, Default)]
pub enum LogConfig {
    #[default]
    Stdout,
    File(PathBuf),
}

impl LogConfig {
    pub fn layer<S>(&self) -> Box<dyn Layer<S> + Send + Sync + 'static>
    where
        S: tracing::Subscriber,
        for<'a> S: LookupSpan<'a>,
    {
        let layer = tracing_subscriber::fmt::layer().json();
        match self {
            LogConfig::Stdout => Box::new(layer.with_writer(io::stdout)),
            LogConfig::File(path) => {
                println!("Creating log file at {}", path.to_string_lossy());
                let file = File::create(path).expect("Failed to create log file.");
                Box::new(layer.with_writer(file))
            }
        }
    }
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

#[derive(Clone)]
pub struct Wrap<T>(pub T);
