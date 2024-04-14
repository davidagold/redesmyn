use std::{
    collections::{BTreeMap, BTreeSet},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use crate::common::Wrap;
use serde::{self, Serialize, Serializer};
use serde_json::{Number, Value};
use strum::IntoStaticStr;
use tokio::{
    fs::File,
    io::AsyncWriteExt,
    sync::{
        mpsc::{Receiver, Sender},
        oneshot,
    },
};
use tracing::{
    error,
    field::{Field, Visit},
    span, Subscriber,
};
use tracing_subscriber::{registry::LookupSpan, Layer};

use crate::error::ServiceError;

#[derive(Debug, IntoStaticStr, Serialize, Clone, Eq, PartialEq, Ord, PartialOrd)]
enum Unit {
    Count,
    Percent,
    Seconds,
    Milliseconds,
    Microseconds,
    Nanoseconds,
    Tebibytes,
    Gigibytes,
    Mebibytes,
    Kibibytes,
    Bytes,
    #[strum(serialize = "Terabits/Seconds")]
    TerabitsPerSecond,
    #[strum(serialize = "Gigabits/Seconds")]
    GigabitsPerSecond,
    #[strum(serialize = "Megabits/Seconds")]
    MegabitsPerSecond,
    #[strum(serialize = "Kilobits/Seconds")]
    KilobitsPerSecond,
    #[strum(serialize = "Bits/Seconds")]
    BitsPerSecond,
    #[strum(serialize = "Count/Seconds")]
    CountPerSecond,
}

#[derive(Serialize, Default)]
struct AwsEmfDocument {
    #[serde(rename = "_aws")]
    aws: AwsEmfMetadata,
    #[serde(flatten)]
    pub dimensions: BTreeMap<&'static str, String>,
    #[serde(flatten)]
    pub properties: BTreeMap<&'static str, Value>,
    #[serde(flatten)]
    pub values: BTreeMap<&'static str, Vec<Value>>,
}

trait Metrics: Sized + Serialize {
    fn with_dimensions(self, dimensions: DimensionMapping) -> Self;

    fn put_metrics(&mut self, metrics: MetricsMapping) -> &mut Self;

    async fn write<W: AsyncWriteExt + Unpin>(&self, writer: &mut W) -> Result<(), ServiceError> {
        writer.write(serde_json::to_vec(&self)?.as_slice()).await?;
        writer.write(b"\n").await?;
        Ok(())
    }
}

impl AwsEmfDocument {
    fn new(directive: AwsEmfMetricDirective) -> AwsEmfDocument {
        AwsEmfDocument {
            aws: AwsEmfMetadata { directive, ..AwsEmfMetadata::default() },
            ..AwsEmfDocument::default()
        }
    }

    fn set_timestamp(&mut self) -> Result<(), ServiceError> {
        self.aws.timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|err| ServiceError::Error(err.to_string()))?
            .as_millis();
        Ok(())
    }
}

impl Metrics for AwsEmfDocument {
    fn with_dimensions(mut self, dimensions: DimensionMapping) -> Self {
        let mut dim_set = DimensionSet::new();
        for (name, value) in dimensions {
            self.dimensions.insert(name, value);
            dim_set.push(name);
        }
        self.aws.directive.dimensions.push(dim_set);
        self
    }

    fn put_metrics(&mut self, metrics: MetricsMapping) -> &mut Self {
        for (metric_name, metric_value) in metrics {
            self.aws.directive.define_metric(metric_name, None);
            self.values
                .entry(metric_name)
                .or_insert_with(|| Vec::<Value>::new())
                .push(metric_value);
        }
        self
    }
}

#[derive(Serialize, Default, Clone)]
struct AwsEmfMetadata {
    #[serde(rename = "Timestamp")]
    timestamp: u128,
    #[serde(rename = "CloudWatchMetrics")]
    directive: AwsEmfMetricDirective,
}

type DimensionSet = Vec<&'static str>;
type DimensionPair = (&'static str, String);
type DimensionMapping = BTreeMap<&'static str, String>;
type MetricsMapping = BTreeMap<&'static str, Value>;

#[derive(Serialize, Default, Clone, Debug)]
struct AwsEmfMetricDirective {
    #[serde(rename = "Namespace")]
    namespace: String,
    #[serde(rename = "Dimensions")]
    dimensions: Vec<DimensionSet>,
    #[serde(
        rename = "Metrics",
        serialize_with = "AwsEmfMetricDirective::serialize_metric_definitions"
    )]
    metric_definitions: BTreeSet<MetricDefinition>,
}

impl AwsEmfMetricDirective {
    fn new(namespace: String) -> AwsEmfMetricDirective {
        AwsEmfMetricDirective {
            namespace,
            ..AwsEmfMetricDirective::default()
        }
    }

    fn define_metric(&mut self, name: &'static str, unit: Option<Unit>) {
        self.metric_definitions.insert(MetricDefinition { name, unit });
    }

    fn serialize_metric_definitions<S: Serializer>(
        definitions: &BTreeSet<MetricDefinition>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(definitions.iter())
    }
}

#[derive(Serialize, Default, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct MetricDefinition {
    #[serde(rename = "Name")]
    name: &'static str,
    #[serde(rename = "Unit")]
    unit: Option<Unit>,
}

#[derive(Debug)]
struct MetricsEntry {
    dimensions: DimensionMapping,
    metrics: MetricsMapping,
}

impl MetricsEntry {
    fn new(dimensions: DimensionMapping, metrics: MetricsMapping) -> Self {
        MetricsEntry { dimensions, metrics }
    }

    fn dimension_pairs<'a>(&'a self) -> Vec<DimensionPair> {
        // TODO: Avoid cloning.
        self.dimensions.iter().map(|(k, v)| (*k, v.clone())).collect()
    }
}

#[derive(Debug, Clone)]
pub struct AwsEmfSubscriber {
    tx: Sender<MetricsEntry>,
}

impl AwsEmfSubscriber {
    pub fn new(max_delay_milliseconds: u32, fp: String) -> AwsEmfSubscriber {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<MetricsEntry>(512);
        tokio::spawn(async move {
            let mut file = File::create(fp.as_str()).await.unwrap();
            loop {
                let task =
                    AwsEmfSubscriber::receive_and_flush(&mut rx, &mut file, max_delay_milliseconds);
                if let Err(err) = task.await {
                    error!("{err}");
                };
            }
        });
        AwsEmfSubscriber { tx }
    }

    async fn receive_and_flush<W: AsyncWriteExt + Unpin>(
        rx: &mut Receiver<MetricsEntry>,
        writer: &mut W,
        max_delay_milliseconds: u32,
    ) -> Result<(), ServiceError> {
        let mut metrics_by_dims = BTreeMap::<Vec<DimensionPair>, AwsEmfDocument>::new();
        let (tx_timeout, mut rx_timeout) = oneshot::channel::<()>();

        tokio::spawn(async move {
            tokio::time::sleep(Duration::new(0, 1_000_000 * max_delay_milliseconds)).await;
            tx_timeout.send(());
        });

        loop {
            if let Ok(_) = rx_timeout.try_recv() {
                for document in metrics_by_dims.values_mut() {
                    document.set_timestamp()?;
                    document.write(writer).await?;
                }
                writer.flush().await?;
                metrics_by_dims.clear();
                return Ok(());
            };

            let entry = rx.recv().await.ok_or_else(|| {
                ServiceError::Error("Failed to receive metrics: Channel closed".into())
            })?;

            metrics_by_dims
                .entry(entry.dimension_pairs())
                .or_insert_with(|| {
                    let directive = AwsEmfMetricDirective::new("DummyNamespace".into());
                    AwsEmfDocument::new(directive).with_dimensions(entry.dimensions)
                })
                .put_metrics(entry.metrics);
        }
    }
}

impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for AwsEmfSubscriber {
    fn on_event(
        &self,
        _event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let mut dims_event = DimensionMapping::new();
        let mut metrics = BTreeMap::<&'static str, Value>::new();
        _event.record(&mut Wrap((&mut dims_event, &mut metrics)));

        // Include, but do not overwrite with, dimensions from parent span.
        // (Operate in closure for easier `None`-handling.)
        (|| {
            let span = _ctx.event_span(_event)?;
            let ext = span.extensions();
            let dims_span = ext.get::<DimensionMapping>()?;
            for (k, v) in dims_span.iter() {
                dims_event.entry(*k).or_insert(v.clone());
            }
            Some(())
        })();

        let entry = MetricsEntry::new(dims_event, metrics);
        let _ = self.tx.try_send(entry);
    }

    fn on_new_span(
        &self,
        attrs: &span::Attributes<'_>,
        id: &span::Id,
        ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let mut dims = DimensionMapping::new();

        // Include dimensions of present span.
        attrs.record(&mut Wrap(&mut dims));

        // Inherit dimensions of parent span.
        (|| {
            let parent_span = ctx.span_scope(id)?.nth(1)?;
            for (k, v) in parent_span.extensions().get::<Wrap<DimensionMapping>>()?.0.iter() {
                // TODO: Avoid cloning
                dims.entry(*k).or_insert_with(|| v.clone());
            }
            Some(())
        })();

        if dims.is_empty() {
            return;
        }
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(dims);
        };
    }
}

fn unprefix<'a>(field: &Field, prefix: &str) -> Option<&'a str> {
    let (_, key) = field.name().split_once(prefix)?;
    Some(key)
}

impl Wrap<(&mut DimensionMapping, &mut MetricsMapping)> {
    // Could make this a macro to avoid creating unnecessary `Value` variants.
    fn record_if_metric(&mut self, field: &Field, value: Value) {
        let (_, metrics) = &mut self.0;
        unprefix(field, "__Metrics.").map(|key| match value {
            // k-v pair may be of form `__Metrics.<name>.Unit = <unit>`.
            // The actual metric is recorded in a separate field,
            // so we do not need to insert the value into `metrics` here.
            Value::String(maybe_unit) => match key.rsplit_once(".Unit") {
                Some(_) => {
                    // TODO: Handle unit declaration
                }
                // Metrics cannot be strings.
                None => (),
            },
            // k-v pair is of form `__Metrics.<key> = <value>`.
            _ => metrics.insert(key, value).map(|_| ()).unwrap_or(()),
        });
    }
}

impl Visit for Wrap<&mut DimensionMapping> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        return;
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        unprefix(field, "__Dimensions").map(|dim_name| self.0.insert(dim_name, value.to_string()));
    }
}

impl Visit for Wrap<(&mut DimensionMapping, &mut MetricsMapping)> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.record_if_metric(field, Value::String(format!("{:#?}", value)));
    }

    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        self.record_if_metric(field, Value::Bool(value));
    }

    fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
        self.record_if_metric(field, Value::Number(Number::from_f64(value).unwrap()))
    }

    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        self.record_if_metric(field, Value::Number(Number::from_f64(value as f64).unwrap()))
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        let (dimensions, metrics) = &mut self.0;
        unprefix(field, "__Dimensions")
            .map(|dim_name| dimensions.insert(dim_name, value.to_string()));
        self.record_if_metric(field, Value::String(value.to_string()))
    }
}

#[macro_export]
macro_rules! metrics {
    ($($name:ident $(: $unit:path)? = $value:expr),* $(,)?) => {
        {
            tracing::info!(
                $(__Metrics.$name = $value $(, __Metrics.$name.Unit = stringify!($unit))?),*
            )
        }
    };
}
