use std::marker::PhantomData;

use crate::error::{ServiceError, ServiceResult};
use pyo3::{
    types::{PyDict, PyFunction, PyTuple},
    IntoPy, Py, PyErr, PyObject, PyResult, ToPyObject,
};

pub trait Runtime {
    type Expecting;
    type Error: std::error::Error;
}

pub struct Rust<T> {
    expecting: PhantomData<T>,
}
impl<T> Runtime for Rust<T> {
    type Expecting = T;
    type Error = ServiceError;
}

pub struct Python {}
impl Runtime for Python {
    type Expecting = PyObject;
    type Error = PyErr;
}

enum EitherImpl<A, K, R> {
    Python(Py<PyFunction>),
    Rust(Box<dyn Fn(A, Option<K>) -> PyResult<R>>),
}

pub trait Invoke<Args, Kwargs, Returned, For: Runtime>
where
    For::Expecting: From<Returned>,
{
    fn invoke(&self, args: Args, kwargs: Option<Kwargs>) -> Result<For::Expecting, For::Error>;
}

impl<A, K, R> Invoke<A, K, R, self::Python> for EitherImpl<A, K, R>
where
    A: IntoPy<Py<PyTuple>>,
    for<'kw> &'kw PyDict: From<K>,
    R: ToPyObject,
    PyObject: From<R>,
{
    fn invoke(&self, args: A, kwargs: Option<K>) -> PyResult<PyObject> {
        match self {
            EitherImpl::Python(f) => {
                pyo3::Python::with_gil(|py| f.call(py, args, kwargs.map(|val| val.into())))
            }
            EitherImpl::Rust(f) => {
                // Don't hold the GIL if we don't need to
                let res = f(args, kwargs)?;
                Ok(pyo3::Python::with_gil(|py| res.to_object(py)))
            }
        }
    }
}

impl<A, K, R> Invoke<A, K, PyObject, self::Rust<R>> for EitherImpl<A, K, R>
where
    A: IntoPy<Py<PyTuple>>,
    for<'kw> &'kw PyDict: From<K>,
    R: From<PyObject>,
{
    fn invoke(&self, args: A, kwargs: Option<K>) -> ServiceResult<R> {
        let res: R = match self {
            EitherImpl::Python(f) => {
                pyo3::Python::with_gil(|py| f.call(py, args, kwargs.map(|kwargs| kwargs.into())))?
                    .into()
            }
            EitherImpl::Rust(f) => f(args, kwargs)?,
        };
        Ok(res)
    }
}
