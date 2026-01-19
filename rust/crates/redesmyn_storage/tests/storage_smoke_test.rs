use redesmyn_ids::EventId;
use redesmyn_storage::{
    events::{EventRecord, get_event, insert_event},
    in_transaction, open_test_sqlite_pool,
};

#[tokio::test(flavor = "current_thread")]
async fn applies_migrations_and_runs_queries() {
    redesmyn_logging::init();

    let pool = open_test_sqlite_pool().await.unwrap();

    let event = EventRecord::new_now(EventId::new(), "test.event", vec![1, 2, 3]);
    let event_to_insert = event.clone();

    in_transaction(&pool, |conn| {
        Box::pin(async move {
            insert_event(&mut *conn, &event_to_insert).await?;
            Ok(())
        })
    })
    .await
    .unwrap();

    let loaded = get_event(&pool, event.id).await.unwrap().unwrap();
    assert_eq!(loaded, event);
}
