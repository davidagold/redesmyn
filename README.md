> redes-man -- An adviser, a counselor. [[quod.lib.umich.edu]](https://quod.lib.umich.edu/m/middle-english-dictionary/dictionary/MED36316)

# Redesmyn: serve ML model predictions with Python and Rust

Redesmyn (/ˈreɪd.smɪn/, REEDZ-min) helps you build services for real-time ML inference and data processing in Python and Rust:
* **Dual language**: Core Redesmyn functionality is written in Rust for safety and performance and exposed through interoperable Python and Rust APIs.
Use and extend this functionality with Rust, Python, or whichever combination suits you:
    * Specify services and handler chains in either language.
    * Implement request handlers in either or both languages:
        * Declare handler schemas via Rust structs that derive the `Record` trait to take advantage of [strongly-typed Serde parsing](https://docs.rs/serde_json/latest/serde_json/#parsing-json-as-strongly-typed-data-structures) parsing, or 
        * Declare handler schemas via [Pydantic](https://docs.pydantic.dev/latest/) models to use [untyped Serde parsing](https://docs.rs/serde_json/latest/serde_json/#operating-on-untyped-json-values).
        * Redesmyn by default passes model input parsed using either strategy above to your Python handlers as [Polars](https://pola.rs) DataFrames with zero-copy thanks to [Arrow](https://arrow.apache.org).
        * Default support for Redesmyn gRPC schema.
    * Start your server from either language.
    
* **Transparent**: Redesmyn aims to expose as much control over your HTTP server as possible:
    * Specify your endpoint paths and parameters.
    * Apply custom header, request, and response logic.
    * Middleware? Handlers are
    * If you need finer control, you can wrap a Redesmyn service in an HTTP proxy using another Python or Rust library to Redesmyn.
* **Obervable**: Redesmyn aims to help you monitor the behavior of your models in real-time and post-hoc:
    * Logging is structured by default and surfaces dimensioned metrics for latency, availability, and input distribution summaries.
    * Write input to and output from any request handler to supported data sinks.
* **Optional integrated model store**: Integrate your endpoints with local or remote filesystem as your model stores: 
    * Map path and query parameters to model specifications that determine the location of your models.
    * Configure your service to automatically fetch updated models asynchronously without service interruption.

## Safety

While Rust does offer great protection from memory exceptions, it is important to note:
1. Rust's type system does not necessarily guard against unintended behavior due to loss of state while awaiting a future in an async process that gets canceled.
This difficulty is not unique to Rust.
However, users should recognize the differences between synchronous and asynchronous Rust and how these differences can signficantly change the behavior of otherwise near-identical synchronous implementations. 
2. 
Memory safety in Rust is enabled by the latter's type system.
Care must be taken to preserve type safety when interacting with foreign data.


# Real-time inference

## Schema

Given a struct `U` that describes the structure of an incoming record, the trait `Schema<U>` implements a methods to ...

You can use the `derive(Schema)` proc macro in Rust or the `Schema` mixin in Python to automatically derive these methods.


## Service

An entity that implements the `Service<U, V>` trait/protocol receives foreign requests containing records that implement `Schema<U>` and returns records that implement `Schema<V>`.

You can apply the `service()` macro in Rust or the `@service` decorator in Python to an async function to derive these methods.

```
#[service("path/to/service")]
async fn predict(req: Request<Schema<U>>) {
    ...
}
```
<!-- 
```
impl Service<U, V> for MyService {
    fn invoke<U>(handle: Handle<MyService>, req: Request<U>) -> Schema<V>;

    fn path(service: Service<U>) -> Path;
}
``` -->


```
@service("path/to/service")
async fn predict(request: Request[Schema[U]]):
    ...

```

## Handler

A `Handler<H, H'>` acts on data received by a service.
You can chain handlers to specify a service's workflow.



# Scalable online data processing

Redesmyn supports building scalable data pipelines on top of the same constructs used to build real-time inference services.

## Why?

Spark is powerful but complex.
Orchestration requires a lot of infrastructure.
Sometimes workflows need to be distributed but not *that* distributed.
`redesmyn-cascade` allows you to 

Benefits:
* Containerize only if you need to
* Greater observability
* Greater control over distributed workflow

## Handoff

The `Handoff<U, X, Between<B>>` trait describes an interface between services in terms of their output and input schemas as well as the method by which the services should communicate.
`Handoff` is parametrized by:
* `V`, `X`:
* `B`: parameter to `Between<B>`.
The implementation of `Handoff<V, X>` requires
```
impl<B> Between<B> for service_x {
    fn invoke<B>(&mut self, df_x: TypedDataFrame<X>) -> TypedDataFrame<Y>;
};
```

Suppose we have  services `service_u: impl Service<U, V>` and `service_x: impl Service<X, Y>`, `u`:
```
[#endpoint("{path}/{to}/service_u")]
fn service_u(input_df: TypedDataFrame<U>) -> TypedDataFrame<V> {
    ...
}
```
Given such an implementation and `Handoff<V, X, B>` as above, `service_u` can *hand off* its successful result to `service_x`:
```
redesmyn::compose(service_x, handoff<V, X, B>, service_u)(input_u);  // TypedDataFrame<Y>;
```
The services `service_u`, `service_x` could be:
* In the same application (`Between<Service>`),
* In different applications but the same container (`Between<App>`),
* In different containers but on the same host (`Between<Container>`),
* On different hosts (`Between<Host>`).

The parameter `B` of `Between<B>` determines the means by which `u` obtains a handle for `x`.
In order to support cascading for a particular variant `R`, a `Service` must implement `Between<R>`.
For instance, on ECS Fargate with [*TODO: instert name*] networking mode, we can implement:
```
impl Between<Container> for service_x {
    fn invoke<Container>(&mut self, df_x: TypedDataFrame<X>) -> TypedDataFrame<Y> {
        // request to service_x.get_path() with localhost
    }
}

#[derive(Handler)]
fn v_to_x(df_v: impl Schema<V>) -> impl Schema<X> {

} 
```
This allows `u` (or any service `Service<_, V>` that implements `Handoff<V, X>`) to invoke `v` against the former's result:
```
invoke(service_x, invoke(service_u, input_df_u).into());  // -> TypedDataFrame<Y>
```

`Host` etc. are traits themselves.
`redesmyn-cascade` includes default implementations for a number of remote access patterns for those developing pure Python applications.
The default implementation of `Between<Service>` enables services to talk to communicate amongst themselves within the same application.


## PartitionedService

A `PartitionedService<U, V, P>` accepts requests that are invariant in fields belonging to the schema `Schema<P>`.


## Orchestration


```
@asset
def input_df() -> TypedDataFrame[Schema[U]]:
    pl.read_parquet()


@op
def u_to_v(df: OfSchema[U]) -> OfSchema[V]:
    return invoke(service_u, df)

@op
@handler
def v_to_x(df: OfSchema[V]) -> OfSchema

@op
def x_to_y(input_df_x) -> TypedDataFrame[Schema[Y]]:
    return invoke(service_x, input_df_x)


@graph_asset
def result(input_df) -> TypedDataFrame[Schema[Y]]:
    pipeline = compose(x_to_y, Handoff(Schema[X], Between(Container)), u_to_v)
    return pipeline(input_df)
```
