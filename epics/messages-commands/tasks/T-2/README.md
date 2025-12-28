# T-2 DB + API: messages, threads, and command loop

## Metadata

```yaml
id: T-2
stacked_on: T-1
node:
  branch: rn/messages-commands/T-2-db-api
```

## Brief (local)

- Implement persistence for messages/threads and the command loop.
- Replace ad-hoc `dict` payloads with typed Pydantic payload models wherever feasible (commands, message metadata).
- Add APIs for:
  - send message
  - list thread messages (by node/session)
  - issue command
  - agent poll/ack + update command state
- Emit corresponding events for Timeline + WebSocket consumers.

## Acceptance Criteria

- Messages and command transitions are durable and queryable.
- APIs are typed and versionable; payloads are modeled (no “stringly-typed dict soup”).

