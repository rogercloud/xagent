"use client"

import { useMemo, useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import type { WorkforceAgentOption, WorkforceWorkerDraft } from "@/types/workforce"

interface WorkersStepProps {
  managerAgentId: string
  agents: WorkforceAgentOption[]
  workers: WorkforceWorkerDraft[]
  onWorkersChange: (workers: WorkforceWorkerDraft[]) => void
}

export function WorkersStep({
  managerAgentId,
  agents,
  workers,
  onWorkersChange,
}: WorkersStepProps) {
  const [draftAgentId, setDraftAgentId] = useState("")
  const [draftAlias, setDraftAlias] = useState("")
  const [draftInstructions, setDraftInstructions] = useState("")

  const selectableAgents = useMemo(
    () =>
      agents.filter(
        (agent) =>
          String(agent.id) !== managerAgentId &&
          !workers.some((worker) => worker.agent_id === agent.id),
      ),
    [agents, managerAgentId, workers],
  )

  const addWorker = () => {
    if (!draftAgentId || !draftInstructions.trim()) return
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
    setDraftAlias("")
    setDraftInstructions("")
  }

  const updateWorker = (index: number, nextWorker: WorkforceWorkerDraft) => {
    const next = [...workers]
    next[index] = nextWorker
    onWorkersChange(next)
  }

  const moveWorker = (index: number, direction: -1 | 1) => {
    const targetIndex = index + direction
    if (targetIndex < 0 || targetIndex >= workers.length) return
    const next = [...workers]
    const [worker] = next.splice(index, 1)
    next.splice(targetIndex, 0, worker)
    onWorkersChange(
      next.map((item, currentIndex) => ({
        ...item,
        sort_order: currentIndex + 1,
      })),
    )
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

  const canAdd = Boolean(draftAgentId && draftInstructions.trim())

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Add Worker</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>Published Agent</Label>
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
                placeholder="Describe what this worker handles inside the workforce."
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
              const agent = agents.find((item) => item.id === worker.agent_id)
              const title = worker.alias || agent?.name || `Worker ${index + 1}`

              return (
                <div key={`${worker.agent_id}-${index}`} className="rounded-xl border p-4">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <div className="font-medium">{title}</div>
                      <div className="text-sm text-muted-foreground">
                        {agent?.description || "Published agent worker"}
                      </div>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => moveWorker(index, -1)}
                        disabled={index === 0}
                      >
                        Up
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => moveWorker(index, 1)}
                        disabled={index === workers.length - 1}
                      >
                        Down
                      </Button>
                      <Button variant="outline" size="sm" onClick={() => removeWorker(index)}>
                        Remove
                      </Button>
                    </div>
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
                          Disabled workers are kept in the Workforce but skipped at runtime.
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
                </div>
              )
            })
          )}
        </CardContent>
      </Card>
    </div>
  )
}
