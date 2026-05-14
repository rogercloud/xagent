"use client"

import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"
import type {
  WorkforceAgentOption,
  WorkforceBuilderApplyPayload,
  WorkforceBuilderApplyResponse,
  WorkforceBuilderMessagesResponse,
  WorkforceBuilderProposePayload,
  WorkforceBuilderProposeResponse,
  WorkforceCanvasResponse,
  WorkforceCreatePayload,
  WorkforceDetail,
  WorkforceListResponse,
  WorkforceRunResponse,
  WorkforceUpdatePayload,
  WorkforceWorker,
  WorkforceWorkerPayload,
  WorkforceWorkerUpdatePayload,
  WorkforceTemplateOption,
} from "@/types/workforce"

async function parseApiError(response: Response, fallback: string): Promise<Error> {
  try {
    const data = await response.json()
    return new Error(data?.detail || fallback)
  } catch {
    return new Error(fallback)
  }
}

export async function listWorkforces(params?: {
  page?: number
  size?: number
  search?: string
  status?: string
}): Promise<WorkforceListResponse> {
  const searchParams = new URLSearchParams()
  if (params?.page) searchParams.set("page", String(params.page))
  if (params?.size) searchParams.set("size", String(params.size))
  if (params?.search) searchParams.set("search", params.search)
  if (params?.status) searchParams.set("status", params.status)

  const suffix = searchParams.toString() ? `?${searchParams.toString()}` : ""
  const response = await apiRequest(`${getApiUrl()}/api/workforces${suffix}`)
  if (!response.ok) {
    throw await parseApiError(response, "Failed to load workforces")
  }
  return response.json()
}

export async function getWorkforce(workforceId: number | string): Promise<WorkforceDetail> {
  const response = await apiRequest(`${getApiUrl()}/api/workforces/${workforceId}`)
  if (!response.ok) {
    throw await parseApiError(response, "Failed to load workforce")
  }
  return response.json()
}

export async function createWorkforce(payload: WorkforceCreatePayload): Promise<WorkforceDetail> {
  const response = await apiRequest(`${getApiUrl()}/api/workforces`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  })
  if (!response.ok) {
    throw await parseApiError(response, "Failed to create workforce")
  }
  return response.json()
}

export async function updateWorkforce(
  workforceId: number | string,
  payload: WorkforceUpdatePayload,
): Promise<WorkforceDetail> {
  const response = await apiRequest(`${getApiUrl()}/api/workforces/${workforceId}`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  })
  if (!response.ok) {
    throw await parseApiError(response, "Failed to update workforce")
  }
  return response.json()
}

export async function addWorkforceAgent(
  workforceId: number | string,
  payload: WorkforceWorkerPayload,
): Promise<WorkforceWorker> {
  const response = await apiRequest(`${getApiUrl()}/api/workforces/${workforceId}/agents`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  })
  if (!response.ok) {
    throw await parseApiError(response, "Failed to add workforce worker")
  }
  return response.json()
}

export async function updateWorkforceAgent(
  workforceId: number | string,
  memberId: number | string,
  payload: WorkforceWorkerUpdatePayload,
): Promise<WorkforceWorker> {
  const response = await apiRequest(
    `${getApiUrl()}/api/workforces/${workforceId}/agents/${memberId}`,
    {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    },
  )
  if (!response.ok) {
    throw await parseApiError(response, "Failed to update workforce worker")
  }
  return response.json()
}

export async function removeWorkforceAgent(
  workforceId: number | string,
  memberId: number | string,
): Promise<void> {
  const response = await apiRequest(
    `${getApiUrl()}/api/workforces/${workforceId}/agents/${memberId}`,
    {
      method: "DELETE",
    },
  )
  if (!response.ok) {
    throw await parseApiError(response, "Failed to remove workforce worker")
  }
}

export async function runWorkforce(
  workforceId: number | string,
  message: string,
): Promise<WorkforceRunResponse> {
  const response = await apiRequest(`${getApiUrl()}/api/workforces/${workforceId}/runs`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message }),
  })
  if (!response.ok) {
    throw await parseApiError(response, "Failed to run workforce")
  }
  return response.json()
}

export async function getWorkforceBuilderMessages(
  workforceId: number | string,
): Promise<WorkforceBuilderMessagesResponse> {
  const response = await apiRequest(
    `${getApiUrl()}/api/workforces/${workforceId}/builder/messages`,
  )
  if (!response.ok) {
    throw await parseApiError(response, "Failed to load builder messages")
  }
  return response.json()
}

export async function proposeWorkforceChanges(
  workforceId: number | string,
  payload: WorkforceBuilderProposePayload,
): Promise<WorkforceBuilderProposeResponse> {
  const response = await apiRequest(
    `${getApiUrl()}/api/workforces/${workforceId}/builder/propose`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    },
  )
  if (!response.ok) {
    throw await parseApiError(response, "Failed to propose workforce changes")
  }
  return response.json()
}

export async function applyWorkforceChanges(
  workforceId: number | string,
  payload: WorkforceBuilderApplyPayload,
): Promise<WorkforceBuilderApplyResponse> {
  const response = await apiRequest(
    `${getApiUrl()}/api/workforces/${workforceId}/builder/apply`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    },
  )
  if (!response.ok) {
    throw await parseApiError(response, "Failed to apply workforce changes")
  }
  return response.json()
}

export async function getWorkforceCanvas(
  workforceId: number | string,
): Promise<WorkforceCanvasResponse> {
  const response = await apiRequest(`${getApiUrl()}/api/workforces/${workforceId}/canvas`)
  if (!response.ok) {
    throw await parseApiError(response, "Failed to load workforce canvas")
  }
  return response.json()
}

export async function listAgentOptions(): Promise<WorkforceAgentOption[]> {
  const response = await apiRequest(`${getApiUrl()}/api/agents`)
  if (!response.ok) {
    throw await parseApiError(response, "Failed to load agents")
  }
  return response.json()
}

export async function listWorkforceTemplates(): Promise<WorkforceTemplateOption[]> {
  const response = await apiRequest(`${getApiUrl()}/api/workforces/templates`)
  if (!response.ok) {
    throw await parseApiError(response, "Failed to load workforce templates")
  }
  return response.json()
}
