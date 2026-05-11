export interface WorkforceAgentSummary {
  id: number
  name: string
  description: string | null
  logo_url: string | null
  status: string
}

export interface WorkforceWorker {
  id: number
  agent: WorkforceAgentSummary
  alias: string | null
  assignment_instructions: string
  source_type: string
  template_id: string | null
  enabled: boolean
  sort_order: number
  canvas_position: Record<string, unknown> | null
}

export interface WorkforceManagerListItem {
  id: number
  name: string
  logo_url: string | null
}

export interface WorkforceRunListItem {
  id: number
  task_id: number | null
  status: string
  created_at: string | null
}

export interface WorkforceListItem {
  id: number
  name: string
  description: string | null
  status: string
  manager: WorkforceManagerListItem
  worker_count: number
  last_run: WorkforceRunListItem | null
  created_at: string | null
  updated_at: string | null
}

export interface WorkforceDetail {
  id: number
  name: string
  description: string | null
  status: string
  manager: WorkforceAgentSummary
  manager_instructions: string | null
  workers: WorkforceWorker[]
  canvas_layout: Record<string, unknown> | null
  scope_type: string
  scope_id: string
  owner_user_id: number
  created_at: string | null
  updated_at: string | null
}

export interface WorkforceListResponse {
  items: WorkforceListItem[]
  total: number
  page: number
  size: number
  pages: number
}

export interface WorkforceAgentOption {
  id: number
  name: string
  description: string | null
  status: string
  logo_url: string | null
}

export interface WorkforceTemplateOption {
  id: string
  name: string
  description: string | null
}

export interface WorkforceNewAgentDraft {
  name: string
  description: string
  instructions: string
  execution_mode: string
  models?: Record<string, unknown> | null
  knowledge_bases: string[]
  skills: string[]
  tool_categories: string[]
  suggested_prompts: string[]
}

export interface WorkforceWorkerDraft {
  source_type: "existing" | "template" | "new"
  agent_id?: number
  template_id?: string
  agent?: WorkforceNewAgentDraft
  alias: string
  assignment_instructions: string
  enabled: boolean
  sort_order: number
}

export interface WorkforceCreatePayload {
  name: string
  description?: string
  manager_agent_id: number
  manager_instructions?: string
  status?: string
  workers?: Array<{
    source_type: "existing" | "template" | "new"
    agent_id?: number
    template_id?: string
    agent?: WorkforceNewAgentDraft
    alias?: string
    assignment_instructions: string
    enabled?: boolean
    sort_order?: number
  }>
}

export interface WorkforceRunResponse {
  workforce_run_id: number
  task_id: number
  status: string
  redirect_url: string
}

export interface WorkforceBuilderOperation {
  op: string
  [key: string]: unknown
}

export interface WorkforceBuilderPatch {
  summary: string
  operations: WorkforceBuilderOperation[]
  warnings: string[]
}

export interface WorkforceBuilderMessage {
  id: number
  role: string
  content: string
  status: string
  proposed_patch: WorkforceBuilderPatch | null
  created_at: string | null
}

export interface WorkforceBuilderMessagesResponse {
  items: WorkforceBuilderMessage[]
}

export interface WorkforceBuilderProposePayload {
  message: string
  context?: Record<string, unknown>
}

export interface WorkforceBuilderProposeResponse {
  message_id: number
  assistant_message: string
  proposed_patch: WorkforceBuilderPatch
  requires_confirmation: boolean
}

export interface WorkforceBuilderApplyPayload {
  message_id: number
  proposed_patch: WorkforceBuilderPatch
}

export interface WorkforceBuilderApplyResponse {
  status: string
  message_id: number
  workforce: WorkforceDetail
}

export interface WorkforceCanvasNode {
  id: string
  type: string
  agent_id?: number
  label: string
  position?: Record<string, unknown> | null
  enabled?: boolean
}

export interface WorkforceCanvasEdge {
  id: string
  source: string
  target: string
}

export interface WorkforceCanvasResponse {
  nodes: WorkforceCanvasNode[]
  edges: WorkforceCanvasEdge[]
  layout: Record<string, unknown>
}
