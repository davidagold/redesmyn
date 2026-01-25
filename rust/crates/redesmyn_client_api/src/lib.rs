//! Transport-agnostic client for the control-plane Client API protocol (T-12/T-20).

#![forbid(unsafe_code)]

#[cfg(all(unix, feature = "uds"))]
pub mod uds;

use std::collections::HashMap;

use tokio::sync::{mpsc, oneshot};

use redesmyn_ids::{RequestId, SessionId, SubscriptionId};
use redesmyn_logging::tracing;
use redesmyn_protocol::client::{
    ClientFrame, ClientMessage, Event, Request, RequestPayload, ResponseResult, SessionEventCursor,
    SessionEventsFilter, Subscribe, SubscriptionEvent, SubscriptionFilter, Unsubscribe,
};
use redesmyn_protocol::{ErrorCategory, ErrorEnvelope, ProtocolEnvelope};
use redesmyn_transport::client::{ClientConnection, ClientTransportError};

#[derive(Clone, Debug)]
pub struct Client {
    tx: mpsc::Sender<ClientCommand>,
}

pub struct ClientTask<C> {
    conn: C,
    rx: mpsc::Receiver<ClientCommand>,
}

enum ClientCommand {
    Request {
        envelope: ProtocolEnvelope,
        payload: RequestPayload,
        respond_to: oneshot::Sender<ResponseResult>,
    },
    Subscribe {
        envelope: ProtocolEnvelope,
        subscription_id: SubscriptionId,
        filter: SubscriptionFilter,
        events_tx: mpsc::Sender<SubscriptionEvent>,
        ack: oneshot::Sender<Result<(), ErrorEnvelope>>,
    },
    Unsubscribe {
        envelope: ProtocolEnvelope,
        subscription_id: SubscriptionId,
    },
}

impl Client {
    #[must_use]
    pub fn connect<C>(conn: C, buffer: usize) -> (Self, ClientTask<C>)
    where
        C: ClientConnection,
    {
        let (tx, rx) = mpsc::channel(buffer.max(1));
        (Self { tx }, ClientTask { conn, rx })
    }

    pub async fn request(&self, payload: RequestPayload) -> Result<ResponseResult, ErrorEnvelope> {
        self.request_with_envelope(ProtocolEnvelope::new(), payload)
            .await
    }

    pub async fn request_with_envelope(
        &self,
        envelope: ProtocolEnvelope,
        payload: RequestPayload,
    ) -> Result<ResponseResult, ErrorEnvelope> {
        let (tx, rx) = oneshot::channel();

        self.tx
            .send(ClientCommand::Request {
                envelope,
                payload,
                respond_to: tx,
            })
            .await
            .map_err(|_| unavailable("Client API channel closed."))?;

        rx.await
            .map_err(|_| unavailable("Client API response channel closed."))
    }

    pub async fn subscribe_session_events(
        &self,
        session_id: SessionId,
        after: Option<SessionEventCursor>,
    ) -> Result<(SubscriptionId, mpsc::Receiver<SubscriptionEvent>), ErrorEnvelope> {
        self.subscribe(
            ProtocolEnvelope::new(),
            SubscriptionFilter::SessionEvents(SessionEventsFilter { session_id, after }),
        )
        .await
    }

    pub async fn subscribe(
        &self,
        envelope: ProtocolEnvelope,
        filter: SubscriptionFilter,
    ) -> Result<(SubscriptionId, mpsc::Receiver<SubscriptionEvent>), ErrorEnvelope> {
        let subscription_id = SubscriptionId::new();
        let (events_tx, events_rx) = mpsc::channel(256);
        let (ack_tx, ack_rx) = oneshot::channel();

        self.tx
            .send(ClientCommand::Subscribe {
                envelope,
                subscription_id,
                filter,
                events_tx,
                ack: ack_tx,
            })
            .await
            .map_err(|_| unavailable("Client API channel closed."))?;

        ack_rx
            .await
            .map_err(|_| unavailable("Client API subscription channel closed."))??;

        Ok((subscription_id, events_rx))
    }

    pub async fn unsubscribe(&self, subscription_id: SubscriptionId) -> Result<(), ErrorEnvelope> {
        self.tx
            .send(ClientCommand::Unsubscribe {
                envelope: ProtocolEnvelope::new(),
                subscription_id,
            })
            .await
            .map_err(|_| unavailable("Client API channel closed."))?;
        Ok(())
    }
}

impl<C> ClientTask<C>
where
    C: ClientConnection,
{
    pub async fn run(mut self) {
        run_client_loop(&mut self.conn, &mut self.rx).await;
    }
}

async fn run_client_loop<C>(conn: &mut C, rx: &mut mpsc::Receiver<ClientCommand>)
where
    C: ClientConnection,
{
    let mut pending_requests: HashMap<RequestId, oneshot::Sender<ResponseResult>> = HashMap::new();
    let mut subscriptions: HashMap<SubscriptionId, mpsc::Sender<SubscriptionEvent>> =
        HashMap::new();
    let mut subscribe_acks: HashMap<SubscriptionId, oneshot::Sender<Result<(), ErrorEnvelope>>> =
        HashMap::new();

    loop {
        tokio::select! {
            cmd = rx.recv() => {
                let Some(cmd) = cmd else {
                    break;
                };

                match cmd {
                    ClientCommand::Request { envelope, payload, respond_to } => {
                        let request_id = RequestId::new();
                        pending_requests.insert(request_id, respond_to);
                        let frame = ClientFrame::new(
                            envelope,
                            ClientMessage::Request(Request {
                                request_id,
                                payload,
                            }),
                        );

                        if conn.send(frame).await.is_err() {
                            break;
                        }
                    }
                    ClientCommand::Subscribe { envelope, subscription_id, filter, events_tx, ack } => {
                        subscriptions.insert(subscription_id, events_tx);
                        subscribe_acks.insert(subscription_id, ack);
                        let frame = ClientFrame::new(
                            envelope,
                            ClientMessage::Subscribe(Subscribe { subscription_id, filter }),
                        );

                        if conn.send(frame).await.is_err() {
                            break;
                        }
                    }
                    ClientCommand::Unsubscribe { envelope, subscription_id } => {
                        subscriptions.remove(&subscription_id);
                        if let Some(ack) = subscribe_acks.remove(&subscription_id) {
                            let _ = ack.send(Err(unavailable("Subscription canceled.")));
                        }
                        let frame = ClientFrame::new(
                            envelope,
                            ClientMessage::Unsubscribe(Unsubscribe { subscription_id }),
                        );

                        if conn.send(frame).await.is_err() {
                            break;
                        }
                    }
                }
            }
            frame = conn.recv() => {
                let frame = match frame {
                    Ok(frame) => frame,
                    Err(_) => break,
                };

                match frame.message {
                    ClientMessage::Response(response) => {
                        if let Some(tx) = pending_requests.remove(&response.request_id) {
                            let _ = tx.send(response.result);
                        }
                    }
                    ClientMessage::Event(event) => {
                        handle_subscription_event(event, &mut subscriptions, &mut subscribe_acks)
                            .await;
                    }
                    ClientMessage::Request(_)
                    | ClientMessage::Subscribe(_)
                    | ClientMessage::Unsubscribe(_) => {
                        tracing::debug!("ignoring unexpected server message");
                    }
                }
            }
        }
    }

    fail_pending_requests(&mut pending_requests);
    fail_pending_subscribe_acks(&mut subscribe_acks);
    fail_active_subscriptions(&mut subscriptions).await;
}

async fn handle_subscription_event(
    event: Event,
    subscriptions: &mut HashMap<SubscriptionId, mpsc::Sender<SubscriptionEvent>>,
    subscribe_acks: &mut HashMap<SubscriptionId, oneshot::Sender<Result<(), ErrorEnvelope>>>,
) {
    if let SubscriptionEvent::Subscribed(_) = &event.event {
        if let Some(ack) = subscribe_acks.remove(&event.subscription_id) {
            let _ = ack.send(Ok(()));
        }
    }

    if let SubscriptionEvent::Error(err) = &event.event {
        if let Some(ack) = subscribe_acks.remove(&event.subscription_id) {
            let _ = ack.send(Err(err.clone()));
        }
    }

    let Some(tx) = subscriptions.get(&event.subscription_id) else {
        return;
    };

    if tx.send(event.event).await.is_err() {
        subscriptions.remove(&event.subscription_id);
    }
}

fn fail_pending_requests(pending: &mut HashMap<RequestId, oneshot::Sender<ResponseResult>>) {
    let err = ResponseResult::Error(unavailable("Client API connection closed."));
    for (_request_id, tx) in pending.drain() {
        let _ = tx.send(err.clone());
    }
}

fn fail_pending_subscribe_acks(
    pending: &mut HashMap<SubscriptionId, oneshot::Sender<Result<(), ErrorEnvelope>>>,
) {
    for (_subscription_id, tx) in pending.drain() {
        let _ = tx.send(Err(unavailable("Client API connection closed.")));
    }
}

async fn fail_active_subscriptions(
    subscriptions: &mut HashMap<SubscriptionId, mpsc::Sender<SubscriptionEvent>>,
) {
    let err = SubscriptionEvent::Error(unavailable("Client API connection closed."));
    for (_subscription_id, tx) in subscriptions.drain() {
        let _ = tx.send(err.clone()).await;
    }
}

fn unavailable(message: impl Into<String>) -> ErrorEnvelope {
    ErrorEnvelope::new(ErrorCategory::Unavailable, message)
}

#[allow(dead_code)]
fn transport_unavailable(err: ClientTransportError) -> ErrorEnvelope {
    ErrorEnvelope::new(
        ErrorCategory::Unavailable,
        format!("Client API transport error: {err}"),
    )
}
