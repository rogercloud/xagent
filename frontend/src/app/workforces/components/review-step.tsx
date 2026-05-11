"use client"

import Link from "next/link"
import { AlertTriangle } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type {
  WorkforceAgentOption,
  WorkforceTemplateOption,
  WorkforceWorkerDraft,
} from "@/types/workforce"

interface ReviewStepProps {
  name: string
  description: string
  managerAgentId: string
  managerInstructions: string
  workers: WorkforceWorkerDraft[]
  agents: WorkforceAgentOption[]
  templates: WorkforceTemplateOption[]
}

export function ReviewStep({
  name,
  description,
  managerAgentId,
  managerInstructions,
  workers,
  agents,
  templates,
}: ReviewStepProps) {
  const manager = agents.find((agent) => String(agent.id) === managerAgentId)

  const warnings: string[] = []
  if (manager && manager.status !== "published") {
    warnings.push("Manager is not published yet.")
  }
  for (const worker of workers) {
    if (worker.source_type === "existing") {
      const agent = agents.find((item) => item.id === worker.agent_id)
      if (agent && agent.status !== "published") {
        warnings.push(`${worker.alias || agent.name} is not published yet.`)
      }
    }
    if (!worker.assignment_instructions.trim()) {
      warnings.push(`${worker.alias || "A worker"} is missing assignment instructions.`)
    }
    if (worker.source_type === "new" && worker.agent && !worker.agent.instructions.trim()) {
      warnings.push(
        `${worker.alias || worker.agent.name || "A new worker"} is missing agent instructions.`,
      )
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Review</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {warnings.length > 0 ? (
          <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-amber-900">
            <div className="flex items-center gap-2 font-medium">
              <AlertTriangle className="size-4" />
              Potential Risks
            </div>
            <div className="mt-2 space-y-1 text-sm">
              {warnings.map((warning, index) => (
                <p key={`${warning}-${index}`}>{warning}</p>
              ))}
            </div>
          </div>
        ) : null}

        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Name</div>
            <div className="mt-1 font-medium">{name || "Untitled Workforce"}</div>
          </div>
          <div>
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Manager</div>
            <div className="mt-1 flex items-center gap-2 font-medium">
              <span>{manager?.name || "Not selected"}</span>
              {manager ? <Badge variant="outline">{manager.status}</Badge> : null}
            </div>
            {manager ? (
              <Link
                href={`/build/${manager.id}`}
                target="_blank"
                className="mt-2 inline-block text-sm text-primary hover:underline"
              >
                Open Agent Editor
              </Link>
            ) : null}
          </div>
        </div>
        <div>
          <div className="text-xs uppercase tracking-wide text-muted-foreground">Description</div>
          <div className="mt-1 text-sm text-muted-foreground">{description || "No description"}</div>
        </div>
        <div>
          <div className="text-xs uppercase tracking-wide text-muted-foreground">
            Manager Instructions
          </div>
          <div className="mt-1 whitespace-pre-wrap text-sm text-muted-foreground">
            {managerInstructions || "No manager instructions"}
          </div>
        </div>
        <div>
          <div className="text-xs uppercase tracking-wide text-muted-foreground">Workers</div>
          <div className="mt-3 space-y-3">
            {workers.length === 0 ? (
              <div className="rounded-lg border border-dashed p-4 text-sm text-muted-foreground">
                No workers configured.
              </div>
            ) : (
              workers.map((worker, index) => {
                const agent = worker.agent_id
                  ? agents.find((item) => item.id === worker.agent_id)
                  : null
                const template = worker.template_id
                  ? templates.find((item) => item.id === worker.template_id)
                  : null
                const title =
                  worker.alias ||
                  agent?.name ||
                  template?.name ||
                  worker.agent?.name ||
                  `Worker ${index + 1}`

                return (
                  <div key={`${worker.source_type}-${index}`} className="rounded-lg border p-4">
                    <div className="flex flex-wrap items-center gap-2">
                      <div className="font-medium">{title}</div>
                      <Badge variant="outline">{worker.source_type}</Badge>
                      {agent ? <Badge variant="secondary">{agent.status}</Badge> : null}
                    </div>
                    <div className="mt-1 text-sm text-muted-foreground">
                      {worker.source_type === "existing"
                        ? agent?.description || "Existing agent"
                        : worker.source_type === "template"
                          ? template?.description || "Template-based worker"
                          : worker.agent?.description || "Brand new worker"}
                    </div>
                    <div className="mt-3 text-sm text-muted-foreground">
                      {worker.assignment_instructions}
                    </div>
                    {worker.source_type === "new" && worker.agent ? (
                      <div className="mt-3 rounded-lg bg-muted/30 p-3 text-sm text-muted-foreground">
                        <div>Agent name: {worker.agent.name}</div>
                        <div>Execution mode: {worker.agent.execution_mode}</div>
                      </div>
                    ) : null}
                  </div>
                )
              })
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
