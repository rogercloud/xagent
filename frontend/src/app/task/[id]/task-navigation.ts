interface TaskBackTarget {
  agentId?: number
  workforceId?: number
}

export function getTaskBackHref(task: TaskBackTarget | null | undefined): string {
  if (task?.workforceId) {
    return `/workforces/${task.workforceId}/run`
  }
  if (task?.agentId) {
    return `/agent/${task.agentId}`
  }
  return "/task"
}
