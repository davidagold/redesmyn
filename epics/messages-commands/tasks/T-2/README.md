---
rn:
  node:
    branch: rn/messages-commands/T-2-db-api
  parent: T-1
---

# T-2 DB + API: messages, threads, and command loop

## Plan

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
