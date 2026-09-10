# Runner control command boundary

This follow-up to #2318 removes API route dependencies from control command
execution. It keeps the current in-process deployment and durable command
protocol. Command decisions and their execution both belong to the runner;
there is no new execution/storage facade between them.

## Ownership

- `services/task_command_execution.py` consumes claimed MESSAGE, PAUSE, RESUME,
  and CANCEL commands. It owns execution-time actor checks, task/run checks,
  lease-owner deferral, message delivery reconciliation, resume admission,
  and terminal outcome classification. It calls the existing execution,
  orchestration, lease, and persistence services directly.
- `services/a2a_task_cancel.py` retains the A2A-specific exact-target cancellation
  rules and returns a detached task snapshot instead of an HTTP response.
  External-scope cancellation keeps its existing service and publishes events
  without importing WebSocket routes.
- WebSocket ingress retains authentication, request parsing, durable enqueue,
  acceptance/replay responses, and connection management. A2A, workforce code,
  and application startup import the same command executor directly.

## Notifications

Task-wide events use the existing task event publisher. Personal replies use
an asynchronous dictionary callback. The optional host delivery adapter resolves
that callback by `(task_id, command_id)` and releases the origin registration
when the command completes or becomes terminal. Retriable failures and deferrals
retain the registration, as before.

The WebSocket host owns the connection registry, first-registration rule,
connection validation, disconnect cleanup, and bounded LRU. It also applies
`task_id_updated` connection moves and current control-state enrichment. The
runner receives neither a socket nor a fake socket. Without a host adapter,
personal replies are discarded; durable state and task-wide event publication
still work. Socket disconnect exceptions are translated at the host boundary.

## Invariants and follow-up

Enqueue/claim behavior, actor and owner identities, command ordering, lease
fences, resume reservations, delivery acknowledgements, terminal retry budgets,
and public event fields retain their existing semantics. An accepted command
still does not imply the requested execution transition has finished.

This does not add a process role or a START command. Agent instances and task
handles remain local, and the notification adapter is in-process. Durable start
handoff, runner lifecycle configuration, and cross-process event transport are
separate steps. The legacy intervention acknowledgement is still a UI event;
this change does not implement a new intervention command.
