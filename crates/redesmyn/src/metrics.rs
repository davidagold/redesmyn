use std::collections::BTreeMap;

use serde::{self, Serialize};
use serde_json::Value;
use strum::{EnumString, IntoStaticStr};
use tracing::Subscriber;
use tracing_subscriber::Layer;

#[derive(Debug, IntoStaticStr, Serialize)]
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

#[derive(Serialize)]
struct CloudWatchMetrics {}

#[derive(Serialize, Default)]
struct AwsEmfDocument<'a> {
    #[serde(rename = "_aws")]
    aws: AwsEmfMetadata,
    #[serde(flatten)]
    pub dimensions: BTreeMap<&'a str, &'a str>,
    #[serde(flatten)]
    pub properties: BTreeMap<&'a str, Value>,
    #[serde(flatten)]
    pub values: BTreeMap<&'a str, Vec<Value>>,
}

trait MetricBatch<'a>: Sized {
    fn set_dimension(&mut self, name: &'a str, value: &'a str);
    
    fn put_metric(&mut self, name: &'a str, value: Value);

    fn flush(self) {

    }
}

impl<'a> AwsEmfDocument<'a> {
    fn new(
        directives: Vec<AwsEmfMetricDirective>,
    ) -> AwsEmfDocument<'a> {
        AwsEmfDocument {
            aws: AwsEmfMetadata { directives, ..AwsEmfMetadata::default() },
            ..AwsEmfDocument::default()
        }
    }
}

impl<'a> MetricBatch<'a> for AwsEmfDocument<'a> {

    fn set_dimension(&mut self, name: &'a str, value: &'a str) {
        self.dimensions.insert(name, value);
    }

    fn put_metric(&mut self, name: &'a str, value: Value) {
        self.values.entry(name).and_modify(|values| values.push(value));
    }


    fn flush(self) {

    }
}

#[derive(Serialize, Default)]
struct AwsEmfMetadata {
    #[serde(rename = "Timestamp")]
    timestamp: u64,
    #[serde(rename = "CloudWatchMetrics")]
    directives: Vec<AwsEmfMetricDirective>,
}

type DimensionSet = Vec<String>;

#[derive(Serialize, Default)]
struct AwsEmfMetricDirective {
    #[serde(rename = "Namespace")]
    namespace: String,
    #[serde(rename = "Dimensions")]
    dimensions: Vec<DimensionSet>,
    #[serde(rename = "Metrics")]
    metrics: Vec<MetricDefinition>,
}

#[derive(Serialize, Default)]
struct MetricDefinition {
    #[serde(rename = "Name")]
    name: String,
    #[serde(rename = "Unit")]
    unit: Option<Unit>,
}

#[derive(Serialize)]
struct AwsEmfListener {}

impl<S: Subscriber> Layer<S> for AwsEmfListener {}

struct AwsEmfVisitor {}
