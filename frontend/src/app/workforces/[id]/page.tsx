"use client"

import Link from "next/link"
import { useCallback, useEffect, useMemo, useState } from "react"
import { useParams } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import {
  addWorkforceAgent,
  getWorkforce,
  listAgentOptions,
  removeWorkforceAgent,
  updateWorkforce,
  updateWorkforceAgent,
} from "@/lib/workforces-api"
import type {
  WorkforceAgentOption,
  WorkforceDetail,
  WorkforceWorker,
} from "@/types/workforce"
import { WorkforceSummary } from "../components/workforce-summary"

interface WorkerEditState {
  alias: string
  assignment_instructions: string
  enabled: boolean
  sort_order: number
}

function workerEditState(worker: WorkforceWorker): WorkerEditState {
  return {
    alias: worker.alias || "",
    assignment_instructions: worker.assignment_instructions,
    enabled: worker.enabled,
    sort_order: worker.sort_order,
  }
}

function buildWorkerEditState(workers: WorkforceWorker[]): Record<number, WorkerEditState> {
  return workers.reduce<Record<number, WorkerEditState>>((accumulator, worker) => {
    accumulator[worker.id] = workerEditState(worker)
    return accumulator
  }, {})
}

export default function WorkforceDetailPage() {
  const params = useParams()
  const id = Array.isArray(params.id) ? params.id[0] : params.id
  const [workforce, setWorkforce] = useState<WorkforceDetail | null>(null)
  const [agents, setAgents] = useState<WorkforceAgentOption[]>([])
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)

  const [name, setName] = useState("")
  const [description, setDescription] = useState("")
  const [managerAgentId, setManagerAgentId] = useState("")
  const [managerInstructions, setManagerInstructions] = useState("")
  const [workerEdits, setWorkerEdits] = useState<Record<number, WorkerEditState>>({})
  const [newWorkerAgentId, setNewWorkerAgentId] = useState("")
  const [newWorkerAlias, setNewWorkerAlias] = useState("")
  const [newWorkerInstructions, setNewWorkerInstructions] = useState("")

  const publishedAgents = useMemo(
    () => agents.filter((agent) => agent.status === "published"),
    [agents],
  )

  const syncForm = useCallback((nextWorkforce: WorkforceDetail) => {
    setName(nextWorkforce.name)
    setDescription(nextWorkforce.description || "")
    setManagerAgentId(String(nextWorkforce.manager.id))
    setManagerInstructions(nextWorkforce.manager_instructions || "")
    setWorkerEdits(buildWorkerEditState(nextWorkforce.workers))
  }, [])

  const load = useCallback(async () => {
    if (!id) return
    try {
      setLoading(true)
      setError(null)
      const [workforceData, agentData] = await Promise.all([
        getWorkforce(id),
        listAgentOptions(),
      ])
      setWorkforce(workforceData)
      setAgents(agentData)
      syncForm(workforceData)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load workforce")
    } finally {
      setLoading(false)
    }
  }, [id, syncForm])

  useEffect(() => {
    void load()
  }, [load])

  const managerOptions = publishedAgents
    .filter((agent) => !workforce?.workers.some((worker) => worker.agent.id === agent.id))
    .map((agent) => ({
      value: String(agent.id),
      label: agent.name,
      description: agent.description || undefined,
    }))

  const workerOptions = publishedAgents
    .filter(
      (agent) =>
        String(agent.id) !== managerAgentId &&
        !workforce?.workers.some((worker) => worker.agent.id === agent.id),
    )
    .map((agent) => ({
      value: String(agent.id),
      label: agent.name,
      description: agent.description || undefined,
    }))

  const saveWorkforce = async () => {
    if (!id || !name.trim() || !managerAgentId) return
    try {
      setSaving(true)
      setError(null)
      const next = await updateWorkforce(id, {
        name: name.trim(),
        description: description.trim() || undefined,
        manager_agent_id: Number(managerAgentId),
        manager_instructions: managerInstructions.trim() || undefined,
      })
      setWorkforce(next)
      syncForm(next)
      setMessage("Workforce updated")
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to update workforce")
    } finally {
      setSaving(false)
    }
  }

  const addWorker = async () => {
    if (!id || !newWorkerAgentId || !newWorkerInstructions.trim()) return
    try {
      setSaving(true)
      setError(null)
      await addWorkforceAgent(id, {
        source_type: "existing",
        agent_id: Number(newWorkerAgentId),
        alias: newWorkerAlias.trim() || undefined,
        assignment_instructions: newWorkerInstructions.trim(),
        enabled: true,
        sort_order: (workforce?.workers.length || 0) + 1,
      })
      setNewWorkerAgentId("")
      setNewWorkerAlias("")
      setNewWorkerInstructions("")
      await load()
      setMessage("Worker added")
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to add worker")
    } finally {
      setSaving(false)
    }
  }

  const saveWorker = async (worker: WorkforceWorker) => {
    if (!id) return
    const edit = workerEdits[worker.id]
    if (!edit || !edit.assignment_instructions.trim()) return
    try {
      setSaving(true)
      setError(null)
      const updated = await updateWorkforceAgent(id, worker.id, {
        alias: edit.alias.trim() || undefined,
        assignment_instructions: edit.assignment_instructions.trim(),
        enabled: edit.enabled,
        sort_order: edit.sort_order,
      })
      setWorkforce((current) =>
        current
          ? {
              ...current,
              workers: current.workers.map((item) =>
                item.id === updated.id ? updated : item,
              ),
            }
          : current,
      )
      setMessage("Worker updated")
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to update worker")
    } finally {
      setSaving(false)
    }
  }

  const removeWorker = async (workerId: number) => {
    if (!id) return
    try {
      setSaving(true)
      setError(null)
      await removeWorkforceAgent(id, workerId)
      await load()
      setMessage("Worker removed")
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to remove worker")
    } finally {
      setSaving(false)
    }
  }

  if (loading) return <div className="h-full overflow-y-auto p-4 text-muted-foreground sm:p-8">Loading workforce...</div>
  if (error && !workforce) return <div className="h-full overflow-y-auto p-4 text-red-500 sm:p-8">{error}</div>
  if (!workforce) return <div className="h-full overflow-y-auto p-4 text-muted-foreground sm:p-8">Workforce not found.</div>

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 p-4 sm:p-8">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold">{workforce.name}</h1>
          <p className="mt-2 text-muted-foreground">
            Review and edit the current orchestration before running it.
          </p>
        </div>
        <div className="flex flex-wrap gap-3">
          <Link href={`/workforces/${workforce.id}/builder`}>
            <Button variant="outline">Builder</Button>
          </Link>
          <Link href={`/workforces/${workforce.id}/canvas`}>
            <Button variant="outline">Canvas</Button>
          </Link>
          <Link href={`/workforces/${workforce.id}/run`}>
            <Button>Run Workforce</Button>
          </Link>
        </div>
      </div>

      {error ? <div className="text-sm text-red-500">{error}</div> : null}
      {message ? <div className="text-sm text-emerald-600">{message}</div> : null}

      <div className="grid gap-6 lg:grid-cols-[1fr_0.9fr]">
        <Card>
          <CardHeader>
            <CardTitle>Edit Workforce</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Name</Label>
              <Input value={name} onChange={(event) => setName(event.target.value)} />
            </div>
            <div className="space-y-2">
              <Label>Description</Label>
              <Textarea
                value={description}
                onChange={(event) => setDescription(event.target.value)}
                rows={3}
              />
            </div>
            <div className="space-y-2">
              <Label>Manager</Label>
              <Select
                value={managerAgentId}
                onValueChange={setManagerAgentId}
                options={managerOptions}
              />
            </div>
            <div className="space-y-2">
              <Label>Manager Instructions</Label>
              <Textarea
                value={managerInstructions}
                onChange={(event) => setManagerInstructions(event.target.value)}
                rows={5}
              />
            </div>
            <Button
              onClick={saveWorkforce}
              disabled={saving || !name.trim() || !managerAgentId}
            >
              {saving ? "Saving..." : "Save Workforce"}
            </Button>
          </CardContent>
        </Card>

        <WorkforceSummary workforce={workforce} />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Add Worker</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>Published Agent</Label>
            <Select
              value={newWorkerAgentId}
              onValueChange={setNewWorkerAgentId}
              placeholder="Choose a worker agent"
              options={workerOptions}
            />
          </div>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label>Alias</Label>
              <Input
                value={newWorkerAlias}
                onChange={(event) => setNewWorkerAlias(event.target.value)}
                placeholder="Optional display name"
              />
            </div>
            <div className="space-y-2">
              <Label>Assignment Instructions</Label>
              <Textarea
                value={newWorkerInstructions}
                onChange={(event) => setNewWorkerInstructions(event.target.value)}
                rows={3}
              />
            </div>
          </div>
          <Button
            onClick={addWorker}
            disabled={saving || !newWorkerAgentId || !newWorkerInstructions.trim()}
          >
            Add Worker
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Manage Workers</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {workforce.workers.length === 0 ? (
            <div className="rounded-lg border border-dashed p-4 text-sm text-muted-foreground">
              No workers configured.
            </div>
          ) : (
            workforce.workers
              .slice()
              .sort((a, b) => a.sort_order - b.sort_order)
              .map((worker) => {
                const edit = workerEdits[worker.id] || workerEditState(worker)
                return (
                  <div key={worker.id} className="rounded-xl border p-4">
                    <div className="flex flex-wrap items-start justify-between gap-4">
                      <div>
                        <div className="font-medium">
                          {worker.alias || worker.agent.name}
                        </div>
                        <div className="text-sm text-muted-foreground">
                          {worker.agent.name} · {worker.agent.status}
                        </div>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        <Link href={`/build/${worker.agent.id}`} target="_blank">
                          <Button variant="outline" size="sm">
                            Open Agent
                          </Button>
                        </Link>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => void removeWorker(worker.id)}
                          disabled={saving}
                        >
                          Remove
                        </Button>
                      </div>
                    </div>
                    <div className="mt-4 grid gap-4 md:grid-cols-[1fr_140px_140px]">
                      <div className="space-y-2">
                        <Label>Alias</Label>
                        <Input
                          value={edit.alias}
                          onChange={(event) =>
                            setWorkerEdits((current) => ({
                              ...current,
                              [worker.id]: { ...edit, alias: event.target.value },
                            }))
                          }
                        />
                      </div>
                      <div className="space-y-2">
                        <Label>Order</Label>
                        <Input
                          type="number"
                          value={String(edit.sort_order)}
                          onChange={(event) =>
                            setWorkerEdits((current) => ({
                              ...current,
                              [worker.id]: {
                                ...edit,
                                sort_order: Number(event.target.value) || worker.sort_order,
                              },
                            }))
                          }
                        />
                      </div>
                      <div className="flex items-center justify-between rounded-lg border px-3 py-2">
                        <div className="font-medium">Enabled</div>
                        <Switch
                          checked={edit.enabled}
                          onCheckedChange={(checked) =>
                            setWorkerEdits((current) => ({
                              ...current,
                              [worker.id]: { ...edit, enabled: checked },
                            }))
                          }
                        />
                      </div>
                    </div>
                    <div className="mt-4 space-y-2">
                      <Label>Assignment Instructions</Label>
                      <Textarea
                        value={edit.assignment_instructions}
                        onChange={(event) =>
                          setWorkerEdits((current) => ({
                            ...current,
                            [worker.id]: {
                              ...edit,
                              assignment_instructions: event.target.value,
                            },
                          }))
                        }
                        rows={4}
                      />
                    </div>
                    <Button
                      className="mt-4"
                      variant="outline"
                      onClick={() => void saveWorker(worker)}
                      disabled={saving || !edit.assignment_instructions.trim()}
                    >
                      Save Worker
                    </Button>
                  </div>
                )
              })
          )}
        </CardContent>
      </Card>
      </div>
    </div>
  )
}
