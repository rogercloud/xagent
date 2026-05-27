export type TraceProcessStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "paused"
  | "waiting_for_user"

type TraceProcessEvent = {
  event_type?: string
  data?: unknown
}

const TRACE_PROCESS_STATUSES = new Set<TraceProcessStatus>([
  "pending",
  "running",
  "completed",
  "failed",
  "paused",
  "waiting_for_user",
])

const STOPPED_TRACE_PROCESS_STATUSES = new Set<TraceProcessStatus>([
  "completed",
  "failed",
  "paused",
  "waiting_for_user",
])

const TERMINAL_FAILURE_EVENTS = new Set([
  "agent_error",
  "react_task_failed",
  "task_failed",
  "task_failed_react",
])

const asRecord = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === "object" ? (value as Record<string, unknown>) : null

export const normalizeTraceProcessStatus = (
  status: unknown
): TraceProcessStatus | undefined => {
  if (typeof status !== "string") return undefined
  const normalized = status.trim().toLowerCase()
  return TRACE_PROCESS_STATUSES.has(normalized as TraceProcessStatus)
    ? (normalized as TraceProcessStatus)
    : undefined
}

export const isStoppedTraceProcessStatus = (status: unknown): boolean => {
  const normalized = normalizeTraceProcessStatus(status)
  return normalized ? STOPPED_TRACE_PROCESS_STATUSES.has(normalized) : false
}

export const getTraceProcessStatusFromEvents = (
  events?: TraceProcessEvent[]
): TraceProcessStatus | undefined => {
  if (!Array.isArray(events)) return undefined

  for (let i = events.length - 1; i >= 0; i--) {
    const event = events[i]
    const eventType = event?.event_type || ""
    const eventData = asRecord(event?.data)

    if (eventData) {
      const eventStatus = normalizeTraceProcessStatus(eventData.status)
      if (eventStatus && STOPPED_TRACE_PROCESS_STATUSES.has(eventStatus)) {
        return eventStatus
      }

      if (
        eventType === "trace_error" &&
        (eventData.error_type === "agent_error" ||
          eventData.status === "failed")
      ) {
        return "failed"
      }
    }

    if (TERMINAL_FAILURE_EVENTS.has(eventType)) {
      return "failed"
    }
  }

  return undefined
}

export const resolveTraceProcessStatus = ({
  processStatus,
  taskStatus,
  traceEvents,
}: {
  processStatus?: unknown
  taskStatus?: unknown
  traceEvents?: TraceProcessEvent[]
}): TraceProcessStatus | undefined =>
  normalizeTraceProcessStatus(processStatus) ||
  normalizeTraceProcessStatus(taskStatus) ||
  getTraceProcessStatusFromEvents(traceEvents)
