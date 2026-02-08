use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tokio::sync::{RwLock, mpsc};

use redesmyn_ids::{CommandId, HostId, HostInstanceId};
use redesmyn_logging::tracing;
use redesmyn_protocol::daemon::{CommandDispatch, DaemonFrame, DaemonMessage};
use redesmyn_protocol::{
    ErrorCategory, ErrorDetail, ErrorEnvelope, ProtocolEnvelope, ProtocolVersion, RepoScope, Scope,
};

#[derive(Clone, Default)]
pub struct DaemonRouter {
    inner: Arc<RwLock<DaemonRouterState>>,
}

#[derive(Default)]
struct DaemonRouterState {
    connections: HashMap<HostInstanceId, DaemonConnectionEntry>,
    assignments: HashMap<CommandId, HostInstanceId>,
}

struct DaemonConnectionEntry {
    #[allow(dead_code)]
    host_id: HostId,
    accepted_protocol: ProtocolVersion,
    outbound_tx: mpsc::Sender<DaemonFrame>,
    attached_repo_scopes: HashSet<RepoScope>,
}

impl DaemonRouter {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn register_connection(
        &self,
        host_id: HostId,
        host_instance_id: HostInstanceId,
        accepted_protocol: ProtocolVersion,
        outbound_tx: mpsc::Sender<DaemonFrame>,
    ) {
        let mut inner = self.inner.write().await;
        inner.connections.insert(
            host_instance_id,
            DaemonConnectionEntry {
                host_id,
                accepted_protocol,
                outbound_tx,
                attached_repo_scopes: HashSet::new(),
            },
        );
    }

    pub async fn unregister_connection(&self, host_instance_id: HostInstanceId) {
        let mut inner = self.inner.write().await;
        inner.connections.remove(&host_instance_id);
        inner
            .assignments
            .retain(|_command_id, assigned| *assigned != host_instance_id);
    }

    pub async fn attach_repo(&self, host_instance_id: HostInstanceId, repo_scope: RepoScope) {
        let mut inner = self.inner.write().await;
        let Some(entry) = inner.connections.get_mut(&host_instance_id) else {
            return;
        };
        entry.attached_repo_scopes.insert(repo_scope);
    }

    pub async fn detach_repo(&self, host_instance_id: HostInstanceId, repo_scope: RepoScope) {
        let mut inner = self.inner.write().await;
        let Some(entry) = inner.connections.get_mut(&host_instance_id) else {
            return;
        };
        entry.attached_repo_scopes.remove(&repo_scope);
    }

    #[must_use]
    pub async fn authorize_command_update(
        &self,
        host_instance_id: HostInstanceId,
        command_id: CommandId,
    ) -> bool {
        let inner = self.inner.read().await;
        inner
            .assignments
            .get(&command_id)
            .is_some_and(|assigned| *assigned == host_instance_id)
    }

    pub async fn dispatch_command(
        &self,
        scope: RepoScope,
        command_id: CommandId,
        command_kind: String,
        json_payload: Vec<u8>,
    ) -> Result<HostInstanceId, ErrorEnvelope> {
        let (host_instance_id, accepted_protocol, outbound_tx) = {
            let inner = self.inner.read().await;

            let selected = inner
                .connections
                .iter()
                .filter(|(_, entry)| entry.attached_repo_scopes.contains(&scope))
                .min_by_key(|(id, _)| *id)
                .or_else(|| {
                    inner
                        .connections
                        .iter()
                        .filter(|(_, entry)| entry.attached_repo_scopes.is_empty())
                        .min_by_key(|(id, _)| *id)
                })
                .map(|(id, entry)| (*id, entry.accepted_protocol, entry.outbound_tx.clone()));

            selected.ok_or_else(|| {
                ErrorEnvelope::new(
                    ErrorCategory::Unavailable,
                    "No daemon connection is available for the requested scope.",
                )
            })?
        };

        let mut envelope = ProtocolEnvelope::new();
        envelope.protocol_major = accepted_protocol.major;
        envelope.protocol_minor = accepted_protocol.minor;
        envelope.scope = Some(Scope::Repo { repo: scope });

        let frame = DaemonFrame::new(
            envelope,
            DaemonMessage::CommandDispatch(CommandDispatch {
                command_id,
                scope,
                command_kind,
                json_payload,
            }),
        );

        {
            let mut inner = self.inner.write().await;
            inner.assignments.insert(command_id, host_instance_id);
        }

        if outbound_tx.send(frame).await.is_err() {
            {
                let mut inner = self.inner.write().await;
                inner.assignments.remove(&command_id);
            }

            let detail =
                ErrorDetail::from([("host_instance_id".to_string(), host_instance_id.to_string())]);
            return Err(ErrorEnvelope::new(
                ErrorCategory::Unavailable,
                "Daemon connection closed while dispatching command.",
            )
            .with_detail(detail));
        }

        tracing::debug!(
            command_id = %command_id,
            host_instance_id = %host_instance_id,
            "dispatched command to daemon"
        );

        Ok(host_instance_id)
    }
}
