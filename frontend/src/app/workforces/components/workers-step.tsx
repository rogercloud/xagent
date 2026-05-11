"use client"

import { useMemo, useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import type {
  WorkforceAgentOption,
  WorkforceNewAgentDraft,
  WorkforceTemplateOption,
  WorkforceWorkerDraft,
} from "@/types/workforce"

interface WorkersStepProps {
  managerAgentId: string
  agents: WorkforceAgentOption[]
  templates: WorkforceTemplateOption[]
  workers: WorkforceWorkerDraft[]
  onWorkersChange: (workers: WorkforceWorkerDraft[]) => void
}

const emptyNewAgentDraft = (): WorkforceNewAgentDraft => ({
  name: "",
  description: "",
  instructions: "",
  execution_mode: "balanced",
  knowledge_bases: [],
  skills: [],
  tool_categories: ["basic"],
  suggested_prompts: [],
})

export function WorkersStep({
  managerAgentId,
  agents,
  templates,
  workers,
  onWorkersChange,
}: WorkersStepProps) {
  const [mode, setMode] = useState<"existing" | "template" | "new">("existing")
  const [draftAgentId, setDraftAgentId] = useState("")
  const [draftTemplateId, setDraftTemplateId] = useState("")
  const [draftAlias, setDraftAlias] = useState("")
  const [draftInstructions, setDraftInstructions] = useState("")
  const [draftNewAgent, setDraftNewAgent] = useState<WorkforceNewAgentDraft>(emptyNewAgentDraft())

  const selectableAgents = useMemo(
    () =>
      agents.filter(
        (agent) =>
          String(agent.id) !== managerAgentId &&
          !workers.some(
            (worker) => worker.source_type === "existing" && worker.agent_id === agent.id,
          ),
      ),
    [agents, managerAgentId, workers],
  )

  const addWorker = () => {
    if (!draftInstructions.trim()) return

    if (mode === "existing") {
      if (!draftAgentId) return
      onWorkersChange([
        ...workers,
        {
          source_type: "existing",
          agent_id: Number(draftAgentId),
          alias: draftAlias.trim(),
          assignment_instructions: draftInstructions.trim(),
          enabled: true,
          sort_order: workers.length + 1,
        },
      ])
      setDraftAgentId("")
    } else if (mode === "template") {
      if (!draftTemplateId) return
      onWorkersChange([
        ...workers,
        {
          source_type: "template",
          template_id: draftTemplateId,
          alias: draftAlias.trim(),
          assignment_instructions: draftInstructions.trim(),
          enabled: true,
          sort_order: workers.length + 1,
        },
      ])
      setDraftTemplateId("")
    } else {
      if (!draftNewAgent.name.trim() || !draftNewAgent.instructions.trim()) return
      onWorkersChange([
        ...workers,
        {
          source_type: "new",
          agent: {
            ...draftNewAgent,
            name: draftNewAgent.name.trim(),
            description: draftNewAgent.description.trim(),
            instructions: draftNewAgent.instructions.trim(),
          },
          alias: draftAlias.trim(),
          assignment_instructions: draftInstructions.trim(),
          enabled: true,
          sort_order: workers.length + 1,
        },
      ])
      setDraftNewAgent(emptyNewAgentDraft())
    }

    setDraftAlias("")
    setDraftInstructions("")
  }

  const updateWorker = (index: number, nextWorker: WorkforceWorkerDraft) => {
    const next = [...workers]
    next[index] = nextWorker
    onWorkersChange(next)
  }

  const removeWorker = (index: number) => {
    const next = workers.filter((_, currentIndex) => currentIndex !== index)
    onWorkersChange(
      next.map((worker, currentIndex) => ({
        ...worker,
        sort_order: currentIndex + 1,
      })),
    )
  }

  const canAdd =
    Boolean(draftInstructions.trim()) &&
    (mode === "existing"
      ? Boolean(draftAgentId)
      : mode === "template"
        ? Boolean(draftTemplateId)
        : Boolean(draftNewAgent.name.trim() && draftNewAgent.instructions.trim()))

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Add Worker</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Tabs value={mode} onValueChange={(value) => setMode(value as typeof mode)}>
            <TabsList>
              <TabsTrigger value="existing">Existing Agent</TabsTrigger>
              <TabsTrigger value="template">Template</TabsTrigger>
              <TabsTrigger value="new">Brand New Agent</TabsTrigger>
            </TabsList>

            <TabsContent value="existing" className="space-y-4">
              <div className="space-y-2">
                <Label>Agent</Label>
                <Select
                  value={draftAgentId}
                  onValueChange={setDraftAgentId}
                  placeholder="Choose a worker agent"
                  options={selectableAgents.map((agent) => ({
                    value: String(agent.id),
                    label: agent.name,
                    description: agent.description || undefined,
                  }))}
                />
              </div>
            </TabsContent>

            <TabsContent value="template" className="space-y-4">
              <div className="space-y-2">
                <Label>Template</Label>
                <Select
                  value={draftTemplateId}
                  onValueChange={setDraftTemplateId}
                  placeholder="Choose a template"
                  options={templates.map((template) => ({
                    value: template.id,
                    label: template.name,
                    description: template.description || undefined,
                  }))}
                />
              </div>
            </TabsContent>

            <TabsContent value="new" className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <Label>Agent Name</Label>
                  <Input
                    value={draftNewAgent.name}
                    onChange={(event) =>
                      setDraftNewAgent((current) => ({ ...current, name: event.target.value }))
                    }
                    placeholder="Launch Copywriter Agent"
                  />
                </div>
                <div className="space-y-2">
                  <Label>Execution Mode</Label>
                  <Select
                    value={draftNewAgent.execution_mode}
                    onValueChange={(value) =>
                      setDraftNewAgent((current) => ({ ...current, execution_mode: value }))
                    }
                    options={[
                      { value: "flash", label: "Flash" },
                      { value: "balanced", label: "Balanced" },
                      { value: "think", label: "Think" },
                    ]}
                  />
                </div>
              </div>
              <div className="space-y-2">
                <Label>Description</Label>
                <Textarea
                  value={draftNewAgent.description}
                  onChange={(event) =>
                    setDraftNewAgent((current) => ({
                      ...current,
                      description: event.target.value,
                    }))
                  }
                  placeholder="Short summary of what this new worker agent does."
                  rows={3}
                />
              </div>
              <div className="space-y-2">
                <Label>Agent Instructions</Label>
                <Textarea
                  value={draftNewAgent.instructions}
                  onChange={(event) =>
                    setDraftNewAgent((current) => ({
                      ...current,
                      instructions: event.target.value,
                    }))
                  }
                  placeholder="System instructions for the new worker agent."
                  rows={5}
                />
              </div>
            </TabsContent>
          </Tabs>

          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label>Alias</Label>
              <Input
                value={draftAlias}
                onChange={(event) => setDraftAlias(event.target.value)}
                placeholder="Optional display name"
              />
            </div>
            <div className="space-y-2">
              <Label>Assignment Instructions</Label>
              <Textarea
                value={draftInstructions}
                onChange={(event) => setDraftInstructions(event.target.value)}
                placeholder="Describe what this worker is responsible for inside the workforce."
                rows={3}
              />
            </div>
          </div>
          <Button onClick={addWorker} disabled={!canAdd}>
            Add Worker
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Workers</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {workers.length === 0 ? (
            <div className="rounded-lg border border-dashed p-4 text-sm text-muted-foreground">
              No workers selected yet.
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
                <div
                  key={`${worker.source_type}-${worker.sort_order}-${index}`}
                  className="rounded-xl border p-4"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <div className="font-medium">{title}</div>
                      <div className="text-sm text-muted-foreground">
                        {worker.source_type === "existing"
                          ? agent?.description || "Existing agent worker"
                          : worker.source_type === "template"
                            ? template?.description || "Template-based worker"
                            : worker.agent?.description || "Brand new worker agent"}
                      </div>
                      <div className="mt-1 text-xs uppercase tracking-wide text-muted-foreground">
                        {worker.source_type}
                      </div>
                    </div>
                    <Button variant="outline" size="sm" onClick={() => removeWorker(index)}>
                      Remove
                    </Button>
                  </div>
                  <div className="mt-4 grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label>Alias</Label>
                      <Input
                        value={worker.alias}
                        onChange={(event) =>
                          updateWorker(index, { ...worker, alias: event.target.value })
                        }
                      />
                    </div>
                    <div className="flex items-center justify-between rounded-lg border px-3 py-2">
                      <div>
                        <div className="font-medium">Enabled</div>
                        <div className="text-sm text-muted-foreground">
                          Disabled workers stay in draft but won&apos;t be used at runtime.
                        </div>
                      </div>
                      <Switch
                        checked={worker.enabled}
                        onCheckedChange={(checked) =>
                          updateWorker(index, { ...worker, enabled: checked })
                        }
                      />
                    </div>
                  </div>
                  <div className="mt-4 space-y-2">
                    <Label>Assignment Instructions</Label>
                    <Textarea
                      value={worker.assignment_instructions}
                      onChange={(event) =>
                        updateWorker(index, {
                          ...worker,
                          assignment_instructions: event.target.value,
                        })
                      }
                      rows={4}
                    />
                  </div>
                  {worker.source_type === "new" && worker.agent ? (
                    <div className="mt-4 grid gap-4 md:grid-cols-2">
                      <div className="space-y-2">
                        <Label>Agent Name</Label>
                        <Input
                          value={worker.agent.name}
                          onChange={(event) =>
                            updateWorker(index, {
                              ...worker,
                              agent: { ...worker.agent!, name: event.target.value },
                            })
                          }
                        />
                      </div>
                      <div className="space-y-2">
                        <Label>Execution Mode</Label>
                        <Select
                          value={worker.agent.execution_mode}
                          onValueChange={(value) =>
                            updateWorker(index, {
                              ...worker,
                              agent: { ...worker.agent!, execution_mode: value },
                            })
                          }
                          options={[
                            { value: "flash", label: "Flash" },
                            { value: "balanced", label: "Balanced" },
                            { value: "think", label: "Think" },
                          ]}
                        />
                      </div>
                      <div className="space-y-2 md:col-span-2">
                        <Label>Agent Instructions</Label>
                        <Textarea
                          value={worker.agent.instructions}
                          onChange={(event) =>
                            updateWorker(index, {
                              ...worker,
                              agent: { ...worker.agent!, instructions: event.target.value },
                            })
                          }
                          rows={4}
                        />
                      </div>
                    </div>
                  ) : null}
                </div>
              )
            })
          )}
        </CardContent>
      </Card>
    </div>
  )
}
