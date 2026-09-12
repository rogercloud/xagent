# Shared task execution foundations and handoff

This change contains the acceptance/data, event/recovery, worker handoff, and SDK/Workforce/A2A/Trigger ingress adapters for shared task execution. Application startup and task execution remain on the existing path. It does not start a shared worker or the Redis event bridge.

## Acceptance and persistence

`_AcceptedTurn` represents a persisted turn without an execution lease. With shared execution enabled, acceptance returns `_EnqueuedTurn`: the new run, transcript, file bindings, runtime inputs and START command commit in the same transaction. The request host does not acquire an execution lease or return a local background task. With the flag disabled, the existing `_ClaimedTurn` and local scheduling path is preserved.

A worker consumes START by validating its immutable task/run/version/actor references and its live command claim. Acquiring the exact execution lease and completing START are one transaction; command completion acknowledges handoff, not agent completion. Later controls route to the task's current live owner. Already claimed MESSAGE commands execute directly without nesting another START. Cancellation drains handoff through local registration so it cannot abandon a newly committed lease; process death remains subject to lease recovery.

`task_runtime_secrets` stores encrypted connector secrets and auth selectors per accepted run. Reads verify task, acceptance turn, run and the owner's stable subject. Resume uses the run binding even when the reply creates a new transcript turn. Paused and waiting-for-user runs retain their inputs across workers; completed, failed, replaced, or deleted runs lose them. A new APPEND run never inherits the old run's inputs. Exact-run terminal cleanup and the recovery-loop sweep compensate interrupted deletion without deleting a newer run's values.

No secret values are added to task APIs or event payloads. The store requires a valid, explicitly configured `ENCRYPTION_KEY` and rejects the published development key, including when copied from `example.env`. Unavailable key configuration returns `connector_runtime_unavailable` before writing. Missing accepted inputs fail with `runtime_secret_unavailable`; persisted storage does not extend upstream credential validity or bypass revocation.

SDK and A2A replies enqueue RESUME_INPUT for worker-side checkpoint preparation and reuse the existing run. Their APIs wait for the persisted preparation outcome. Legacy existing-task execution uses an explicit START variant without adding another user transcript row. Legacy and Trigger completion waits poll the exact durable run without retaining a database connection or accepting a replacement run's outcome. Workforce and Trigger projections remain part of their existing acceptance and terminal transactions.


Nullable `reply_host_id` and `reply_origin` fields identify the creating ingress and its exact socket registration. They are stored when a command is created; duplicate submissions cannot overwrite the route. The migration preserves existing commands and can be downgraded independently.

## Events and recovery

The Redis bridge provides task fanout, origin-bound private replies, socket acknowledgements, bounded deduplication and reconnect notifications. Redis remains a Pub/Sub transport, not a task queue or replay log. A private-reply timeout is delivery-unknown and must not cause command execution to retry. Replies without an origin route remain valid for non-socket ingress; they emit a warning and a `xagent.task.reply.delivery` counter with `outcome=no_route`, without logging the reply content.

State-bearing event enrichment is extracted from the WebSocket API so a future producer can capture the original run/version before transport. Each shared socket has a bounded writer queue; slow sockets cannot block the other recipients. Authorized task audiences can receive persistent state/output snapshots to reconcile missing events.

The frontend handles unavailable/resync notifications and exact-run snapshots. It marks interrupted output, avoids appending tokens to a missing prefix, and replaces the display when complete persisted output is available. Existing local output remains unchanged when these events are absent.

## Activation boundary

`XAGENT_SHARED_TASK_EXECUTION_ENABLED` defaults to **false in this intermediate change**. Application startup refuses `true`: this change does not wire the bridge into application startup and is not a deployable shared execution mode. Treat the flag as process startup configuration; changing it inside a running process is unsupported. Tests explicitly initialize the bridge to exercise its contracts. Reconciliation lifecycle tests also start and stop `ConnectionManager`'s loop explicitly; production startup must wire both the bridge and the reconciliation loop together in the activation change.

`XAGENT_TASK_EVENT_CHANNEL_PREFIX` defaults to `xagent:task-events:v1`. Redis logical DB numbers do not isolate Pub/Sub; separate deployments need different prefixes.

Follow-up changes will implement channel bots and worker/web lifecycle integration. The final integration will enable shared execution by default, after those paths are complete. No intermediate PR should turn it on prematurely.

## Validation

Tests accompany acceptance/rollback, immutable origin routing, encrypted input isolation, migration upgrade/downgrade, event identity and deduplication, ACK semantics, bounded socket queues, and frontend recovery. The Redis reconnect test accepts `XAGENT_TEST_REDIS_URL`; PostgreSQL migration tests accept `XAGENT_TEST_POSTGRES_URL` and create/drop their own disposable database. Missing service URLs skip only those external-service tests.
