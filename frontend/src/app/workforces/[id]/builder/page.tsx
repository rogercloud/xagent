"use client"

import { useEffect, useMemo, useState } from "react"
import { useParams } from "next/navigation"
import { toast } from "sonner"
import {
  applyWorkforceChanges,
  getWorkforce,
  getWorkforceBuilderMessages,
  proposeWorkforceChanges,
} from "@/lib/workforces-api"
import type {
  WorkforceBuilderMessage,
  WorkforceBuilderPatch,
  WorkforceDetail,
} from "@/types/workforce"
import { WorkforceBuilderChat } from "../../components/workforce-builder-chat"
import { ProposedPatchCard } from "../../components/proposed-patch-card"
import { WorkforceSummary } from "../../components/workforce-summary"
import { WorkforceTestPanel } from "../../components/workforce-test-panel"

function latestProposedAssistantMessage(
  messages: WorkforceBuilderMessage[],
): WorkforceBuilderMessage | null {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const item = messages[index]
    if (item.role === "assistant" && item.proposed_patch) {
      return item
    }
  }
  return null
}

export default function WorkforceBuilderPage() {
  const params = useParams()
  const id = Array.isArray(params.id) ? params.id[0] : params.id
  const [workforce, setWorkforce] = useState<WorkforceDetail | null>(null)
  const [messages, setMessages] = useState<WorkforceBuilderMessage[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [applying, setApplying] = useState(false)

  const activeProposal = useMemo(() => latestProposedAssistantMessage(messages), [messages])

  useEffect(() => {
    if (!id) return

    const load = async () => {
      try {
        setLoading(true)
        setError(null)
        const [workforceData, historyData] = await Promise.all([
          getWorkforce(id),
          getWorkforceBuilderMessages(id),
        ])
        setWorkforce(workforceData)
        setMessages(historyData.items)
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load builder")
      } finally {
        setLoading(false)
      }
    }

    void load()
  }, [id])

  const handleSubmit = async (message: string) => {
    if (!id) return
    try {
      setSubmitting(true)
      const result = await proposeWorkforceChanges(id, { message })
      const history = await getWorkforceBuilderMessages(id)
      setMessages(history.items)
      toast.success(result.assistant_message || "Builder proposal created")
    } catch (err) {
      const nextError = err instanceof Error ? err.message : "Failed to propose changes"
      toast.error(nextError)
    } finally {
      setSubmitting(false)
    }
  }

  const handleApply = async (messageId: number, patch: WorkforceBuilderPatch) => {
    if (!id) return
    try {
      setApplying(true)
      const result = await applyWorkforceChanges(id, {
        message_id: messageId,
        proposed_patch: patch,
      })
      setWorkforce(result.workforce)
      const history = await getWorkforceBuilderMessages(id)
      setMessages(history.items)
      toast.success("Workforce changes applied")
    } catch (err) {
      const nextError = err instanceof Error ? err.message : "Failed to apply changes"
      toast.error(nextError)
    } finally {
      setApplying(false)
    }
  }

  if (loading) return <div className="h-full overflow-y-auto p-4 text-muted-foreground sm:p-8">Loading builder...</div>
  if (error) return <div className="h-full overflow-y-auto p-4 text-red-500 sm:p-8">{error}</div>
  if (!workforce) return <div className="h-full overflow-y-auto p-4 text-muted-foreground sm:p-8">Workforce not found.</div>

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto grid w-full max-w-[1600px] gap-6 p-4 sm:p-8 xl:grid-cols-[0.95fr_1.05fr_0.8fr]">
        <div className="min-h-[640px]">
          <WorkforceBuilderChat
            messages={messages}
            loading={loading}
            submitting={submitting}
            onSubmit={handleSubmit}
          />
        </div>
        <div className="space-y-6">
          <WorkforceSummary workforce={workforce} />
          <ProposedPatchCard
            patch={activeProposal?.proposed_patch ?? null}
            messageId={activeProposal?.id ?? null}
            status={activeProposal?.status ?? null}
            applying={applying}
            onApply={handleApply}
          />
        </div>
        <div>
          <WorkforceTestPanel workforceId={workforce.id} />
        </div>
      </div>
    </div>
  )
}
