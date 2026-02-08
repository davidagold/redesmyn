use std::future::Future;

use tokio::runtime::Handle;
use tokio::sync::broadcast;
use tokio::task::JoinSet;

use redesmyn_logging::tracing;

#[derive(Debug)]
pub struct TaskManager {
    shutdown_tx: broadcast::Sender<()>,
    tasks: JoinSet<()>,
    runtime: Handle,
}

impl TaskManager {
    #[must_use]
    pub fn new(runtime: Handle) -> Self {
        let (shutdown_tx, _shutdown_rx) = broadcast::channel(1);
        Self {
            shutdown_tx,
            tasks: JoinSet::new(),
            runtime,
        }
    }

    pub fn spawn(&mut self, task_name: &'static str, f: impl Future<Output = ()> + Send + 'static) {
        self.tasks.spawn_on(
            async move {
                let span = tracing::info_span!("control_plane.task", task = task_name);
                let _enter = span.enter();
                f.await;
            },
            &self.runtime,
        );
    }

    #[must_use]
    pub fn subscribe_shutdown(&self) -> broadcast::Receiver<()> {
        self.shutdown_tx.subscribe()
    }

    pub fn trigger_shutdown(&self) {
        let _ = self.shutdown_tx.send(());
    }

    pub async fn join_all(mut self) {
        while let Some(join_result) = self.tasks.join_next().await {
            if let Err(err) = join_result {
                tracing::error!(error = %err, "control plane task panicked");
            }
        }
    }
}
