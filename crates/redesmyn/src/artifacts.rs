use crate::common::{__Str__, consume_and_log_err};
use crate::do_in;
use crate::error::{ArtifactsError, ArtifactsResult};

use bytes::{Buf, BufMut, BytesMut};
use indexmap::IndexMap;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::{IntoPyDict, PyAnyMethods, PyIterator};
use pyo3::{
    exceptions::PyTypeError,
    types::{PyByteArray, PyNone, PyString},
    IntoPy, Py, PyAny, PyResult, Python,
};
use pyo3::{pyclass, pymethods};
use serde::Serialize;
use std::{collections::VecDeque, future::Future, io::Read, path::PathBuf, pin::Pin, sync::Arc};
use strum::Display;
use tracing::info;

pub trait Pydantic: __Str__ + Send + Sync {
    fn fields(&self) -> PyResult<Vec<String>>;

    fn validate(&self, data: &IndexMap<String, String>) -> PyResult<Py<PyAny>>;
}

impl std::fmt::Debug for dyn Pydantic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.__str__().as_str())
    }
}

impl<'bound> Pydantic for Py<PyAny> {
    fn fields(&self) -> PyResult<Vec<String>> {
        Python::with_gil(|py| -> PyResult<_> {
            let mut keys_iter = self
                .bind(py)
                .call_method0("model_fields")?
                .call_method0("keys")?
                .downcast_into::<PyIterator>()
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
            keys_iter.try_fold(
                Vec::<String>::with_capacity(self.bind(py).len()?),
                |mut fields, f| {
                    fields.push(f?.extract()?);
                    Ok(fields)
                },
            )
        })
    }

    fn validate(&self, data: &IndexMap<String, String>) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            self.call_method_bound(py, "model_validate", (data.into_py_dict_bound(py),), None)
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

pub trait Client: std::fmt::Debug + Send + Sync {
    fn substitute(&self, args: IndexMap<String, String>) -> ArtifactsResult<String>;

    fn list_parametrizations(
        &self,
        base_path: PathBuf,
        path_template: PathTemplate,
    ) -> Pin<Box<dyn Future<Output = Vec<(IndexMap<String, String>, PathBuf)>> + Send>> {
        let spec = IndexMap::<String, String>::default();
        let paths_by_spec =
            list_parametrizations_impl([(spec, base_path)].into(), path_template.components());
        Box::pin(std::future::ready(paths_by_spec))
    }

    fn list_from_path(&self, path: &PathBuf) -> Option<impl Iterator<Item = String>>;

    fn fetch_bytes(
        &self,
        spec: BoxedSpec,
        bytes: BytesMut,
    ) -> Pin<Box<dyn Future<Output = ArtifactsResult<BytesMut>> + Send + 'static>>;

    fn fetch<'this>(
        &'this self,
        spec: BoxedSpec,
        fetch_as: FetchAs,
    ) -> Pin<Box<dyn Future<Output = ArtifactsResult<FetchAs>> + Send + 'this>> {
        // TODO: This may be fine for now but we should wrap in an `Arc``
        match fetch_as {
            FetchAs::Uri(None) => {
                let uri = do_in!(|| -> ArtifactsResult<_> {
                    let args = spec.as_map()?;
                    let path = self.substitute(args)?;
                    let uri = Uri::Path(Some(PathBuf::from(path)));
                    Ok(FetchAs::Uri(Some(uri)))
                });
                Box::pin(std::future::ready(uri))
            }
            FetchAs::Bytes(None) => {
                Box::pin(async move { Ok(self.fetch_bytes(spec, BytesMut::new()).await?.into()) })
            }
            // TODO: Don't panic, just return Error
            _ => panic!(),
        }
    }
}

fn list_parametrizations_impl(
    mut paths_by_spec: Vec<(IndexMap<String, String>, PathBuf)>,
    mut remaining_components: VecDeque<PathComponent>,
) -> Vec<(IndexMap<String, String>, PathBuf)> {
    match remaining_components.pop_front() {
        Some(PathComponent::Fixed(dir_name)) => {
            paths_by_spec.iter_mut().for_each(|(_, ref mut path)| path.push(dir_name.clone()));
            list_parametrizations_impl(paths_by_spec, remaining_components)
        }
        Some(PathComponent::Identifier(identifier)) => {
            let updated_paths_by_spec = paths_by_spec
                .into_iter()
                .filter_map(|(spec, path)| {
                    let dir = std::fs::read_dir(&path).ok()?;
                    let object_names = dir.filter_map(|entry| {
                        Some(entry.ok()?.file_name().to_string_lossy().to_string())
                    });
                    Some(((spec, path.clone()), object_names))
                })
                .flat_map(|((spec, path), names)| {
                    let cloned_identifier = identifier.clone();
                    names.filter_map(move |name| {
                        let (mut cloned_spec, mut cloned_path) = (spec.clone(), path.clone());
                        cloned_spec.insert(cloned_identifier.clone(), name.clone());
                        cloned_path.push(name);
                        Some((cloned_spec, cloned_path))
                    })
                })
                .collect();
            list_parametrizations_impl(updated_paths_by_spec, remaining_components)
        }
        None => paths_by_spec,
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct FsClient {
    base_path: PathBuf,
    path_template: PathTemplate,
}

impl Client for FsClient {
    fn substitute(&self, args: IndexMap<String, String>) -> ArtifactsResult<String> {
        self.path_template.substitute(args)
    }

    fn list_from_path(&self, path: &PathBuf) -> Option<impl Iterator<Item = String>> {
        let dir = std::fs::read_dir(&path).ok()?;
        let object_names = dir
            .into_iter()
            .filter_map(|entry| Some(entry.ok()?.file_name().to_string_lossy().to_string()));
        Some(object_names)
    }

    fn fetch_bytes(
        &self,
        spec: BoxedSpec,
        mut bytes: BytesMut,
    ) -> Pin<Box<dyn Future<Output = ArtifactsResult<BytesMut>> + Send + 'static>> {
        let path = do_in!(|| -> ArtifactsResult<_> {
            Ok(PathBuf::from(self.substitute(spec.as_map()?)?))
        });
        Box::pin(async move {
            bytes.extend(tokio::fs::read(path?).await?);
            Ok(bytes)
        })
    }
}

impl FsClient {
    pub fn new(base_path: PathBuf, path_template: String) -> FsClient {
        FsClient {
            // TODO: Remove redundant `base_path` field somewhere
            base_path: base_path.clone(),
            path_template: PathTemplate { template: path_template, base: base_path },
        }
    }
}

#[pymethods]
impl FsClient {
    #[new]
    pub fn __new__(base_path: Py<PyAny>, path_template: Py<PyString>) -> PyResult<FsClient> {
        Python::with_gil(|py| {
            let client = FsClient::new(
                base_path.extract::<PathBuf>(py)?,
                path_template.extract::<String>(py)?,
            );
            Ok(client)
        })
    }
}

#[pyclass]
#[derive(Clone, Debug)]
struct S3Client {
    base_path: PathBuf,
    path_template: PathTemplate,
    client: aws_sdk_s3::Client,
}

impl Client for S3Client {
    fn substitute(&self, args: IndexMap<String, String>) -> ArtifactsResult<String> {
        self.path_template.substitute(args)
    }

    fn list_from_path(&self, path: &PathBuf) -> Option<impl Iterator<Item = String>> {
        //
    }

    fn fetch_bytes(
        &self,
        spec: BoxedSpec,
        mut bytes: BytesMut,
    ) -> Pin<Box<dyn Future<Output = ArtifactsResult<BytesMut>> + Send + 'static>> {
        let path = do_in!(|| -> ArtifactsResult<_> {
            Ok(PathBuf::from(self.substitute(spec.as_map()?)?))
        });
        Box::pin(async move {
            bytes.extend(tokio::fs::read(path?).await?);
            Ok(bytes)
        })
    }
}

impl S3Client {
    pub fn new(base_path: PathBuf, path_template: String) -> S3Client {
        S3Client {
            // TODO: Remove redundant `base_path` field somewhere
            base_path: base_path.clone(),
            path_template: PathTemplate { template: path_template, base: base_path },
        }
    }
}

#[pymethods]
impl S3Client {
    #[new]
    pub fn __new__(base_path: Py<PyAny>, path_template: Py<PyString>) -> PyResult<S3Client> {
        Python::with_gil(|py| {
            let client = S3Client::new(
                base_path.extract::<PathBuf>(py)?,
                path_template.extract::<String>(py)?,
            );
            Ok(client)
        })
    }
}

#[derive(Clone, Debug)]
pub struct PathTemplate {
    template: String,
    base: PathBuf,
}

enum PathComponent {
    Fixed(String),
    Identifier(String),
}

impl PathTemplate {
    fn components(&self) -> VecDeque<PathComponent> {
        self.template
            .split("/")
            .filter_map(|part| {
                if part.starts_with("{") && part.ends_with("}") {
                    let n_chars = part.len();
                    let identifier: String = part.chars().skip(1).take(n_chars - 2).collect();
                    Some(PathComponent::Identifier(identifier))
                } else if part != "" {
                    Some(PathComponent::Fixed(part.to_string()))
                } else {
                    None
                }
            })
            .collect()
    }

    fn parse(&self, path: &PathBuf) -> IndexMap<String, String> {
        let mut base_parts: VecDeque<_> = self.base.iter().collect();
        let rel_path: PathBuf = path
            .into_iter()
            .skip_while(|&part| {
                let Some(base_part) = base_parts.pop_front() else { return false };
                base_part == part
            })
            .collect();

        self.components()
            .into_iter()
            .filter_map(|part| match part {
                PathComponent::Fixed(_) => None,
                PathComponent::Identifier(identifier) => Some(identifier),
            })
            .zip(rel_path.into_iter().filter_map(|part| part.to_str().map(str::to_string)))
            .collect()
    }

    fn substitute(&self, args: IndexMap<String, String>) -> ArtifactsResult<String> {
        let path = self
            .components()
            .iter()
            .try_fold(None, |path: Option<String>, component| {
                let next_path_component = match component {
                    PathComponent::Fixed(dir_name) => dir_name,
                    PathComponent::Identifier(identifier) => {
                        args.get(identifier).ok_or_else(|| {
                            ArtifactsError::from(format!(
                                "Substitution identifier `{}` is not present in the provided args",
                                identifier
                            ))
                        })?
                    }
                };
                let path = match path {
                    None => next_path_component.to_string(),
                    Some(path) => vec![path, next_path_component.to_string()].join("/"),
                };
                ArtifactsResult::<Option<String>>::Ok(Some(path))
            })?
            .ok_or_else(|| ArtifactsError::from("Failed to substitute args into path template"))?;

        let mut abs_path = self.base.clone();
        abs_path.push(path);
        Ok(abs_path
            .to_str()
            .ok_or_else(|| {
                ArtifactsError::from(format!("Failed to stringify path {:#?}", abs_path))
            })?
            .to_string())
    }
}
