use crate::common::consume_and_log_err;

use bytes::{Buf, BufMut, BytesMut};
use indexmap::IndexMap;
use pyo3::types::{PyAnyMethods, PyIterator};
use pyo3::Bound;
use pyo3::{
    exceptions::PyTypeError,
    types::{PyByteArray, PyNone, PyString},
    IntoPy, Py, PyAny, PyResult, Python,
};
use serde::Serialize;
use std::{io::Read, path::PathBuf, sync::Arc};
use strum::Display;
use tracing::info;

pub trait Pydantic {
    fn fields(&self) -> PyResult<Vec<String>>;
}

impl<'bound> Pydantic for &Bound<'bound, PyAny> {
    fn fields(&self) -> PyResult<Vec<String>> {
        let mut keys_iter = self
            .call_method0("model_fields")?
            .call_method0("keys")?
            .downcast_into::<PyIterator>()?;
        keys_iter.try_fold(Vec::<String>::with_capacity(self.len()?), |mut fields, f| {
            fields.push(f?.extract()?);
            Ok(fields)
        })
    }
}

pub trait ArtifactSpec: Send + Sync {
    fn spec(&self) -> Box<&dyn erased_serde::Serialize>;

    fn as_map(&self) -> Result<IndexMap<String, String>, serde_json::Error> {
        let mut writer = BytesMut::new().writer();
        serde_json::to_writer(&mut writer, self.spec().as_ref())?;
        serde_json::from_reader(writer.into_inner().reader())
    }

    fn as_key(&self) -> Result<String, serde_json::Error> {
        let parts = self.as_map()?.into_iter().map(|(k, v)| [k, v].join("="));
        let key = parts.collect::<Vec<_>>().join("/");
        Ok(key)
    }
}

impl Serialize for dyn ArtifactSpec {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.spec().as_ref().serialize(serializer)
    }
}

impl std::fmt::Display for dyn ArtifactSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let map = self.as_map().map_err(|_| std::fmt::Error::default())?;
        let display = map.into_iter().map(|(k, v)| [k, v].join("=")).collect::<Vec<_>>().join(", ");
        f.write_str(display.as_str())
    }
}

impl std::fmt::Debug for dyn ArtifactSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "<impl ArtifactSpec, fields: {{ {:#?} }}>",
            serde_json::to_string(self.spec().as_ref())
        ))
    }
}

impl<T: Serialize + Send + Sync> ArtifactSpec for T {
    fn spec(&self) -> Box<&dyn erased_serde::Serialize> {
        Box::new(self as &dyn erased_serde::Serialize)
    }
}

pub type BoxedSpec = Box<dyn ArtifactSpec + Send + Sync + 'static>;

#[derive(Clone)]
pub enum Uri {
    Path(Option<PathBuf>),
    Id { extractor: Option<Arc<dyn Fn(PathBuf) -> String + Send + Sync>>, id: Option<String> },
}

impl Default for Uri {
    fn default() -> Self {
        Uri::Path(None)
    }
}

impl std::fmt::Debug for Uri {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            &Uri::Path(Some(path)) => f.write_fmt(format_args!(
                "<Uri::Path path = '{}'>",
                path.to_str().unwrap_or("[error]")
            )),
            &Uri::Path(None) => f.write_fmt(format_args!("<Uri::Path path = unset'>")),
            &Uri::Id { extractor: _, id: Some(id_str) } => {
                f.write_fmt(format_args!("<Uri::Id id = '{}'>", id_str))
            }
            &Uri::Id { extractor: _, id: None } => {
                f.write_fmt(format_args!("<Uri::Id in = [unset]'>"))
            }
        }
    }
}

impl Uri {
    // TODO: Figure out more parsimonious ownership model (not pressing)
    pub(crate) fn parse(self, path: &PathBuf) -> Self {
        match self {
            Uri::Path(None) => Uri::Path(Some(path.clone())),
            Uri::Id { extractor: Some(func), id: None } => {
                let id = func(path.clone());
                Uri::Id { extractor: None, id: Some(id.clone()) }
            }
            Uri::Path(Some(old_path)) => {
                info!("Received path: {:#?}", old_path);
                Uri::Path(Some(path.clone()))
            }
            _ => panic!(),
        }
    }
}

#[derive(Debug, Display)]
pub enum FetchAs {
    Uri(Option<Uri>),
    Bytes(Option<BytesMut>),
}

impl IntoPy<PyResult<Py<PyAny>>> for FetchAs {
    fn into_py(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            FetchAs::Uri(Some(Uri::Path(Some(path)))) => {
                let res: Py<PyAny> = match path.to_str() {
                    Some(path_str) => PyString::new_bound(py, path_str).as_any().clone().unbind(),
                    None => PyNone::get_bound(py).as_any().clone().unbind(),
                };
                Ok(res)
            }
            FetchAs::Uri(Some(Uri::Id { id: Some(id), .. })) => {
                Ok(PyString::new_bound(py, &id).as_any().clone().unbind())
            }
            FetchAs::Bytes(Some(bytes)) => {
                let mut buf = Vec::<u8>::new();
                consume_and_log_err(bytes.reader().read_to_end(&mut buf));
                Ok(PyByteArray::new_bound(py, buf.as_slice()).into())
            }
            FetchAs::Uri(Some(Uri::Id { id: None, .. }))
            | FetchAs::Uri(Some(Uri::Path(None)))
            | FetchAs::Uri(None)
            | FetchAs::Bytes(None) => {
                Err(PyTypeError::new_err("Cannot load model from type `None`"))
            }
        }
    }
}

impl Clone for FetchAs {
    fn clone(&self) -> Self {
        use FetchAs::*;
        match self {
            Uri(maybe_uri) => FetchAs::Uri(maybe_uri.clone()),
            Bytes(maybe_bytes) => FetchAs::Bytes(maybe_bytes.clone()),
        }
    }
}

impl From<Uri> for FetchAs {
    fn from(uri: Uri) -> Self {
        FetchAs::Uri(Some(uri))
    }
}
impl From<BytesMut> for FetchAs {
    fn from(bytes: BytesMut) -> Self {
        FetchAs::Bytes(Some(bytes))
    }
}

impl FetchAs {
    pub(crate) fn empty_like(fetch_as: &FetchAs) -> FetchAs {
        match &fetch_as {
            &FetchAs::Uri(_) => FetchAs::Uri(None),
            &FetchAs::Bytes(_) => FetchAs::Bytes(None),
        }
    }
}
