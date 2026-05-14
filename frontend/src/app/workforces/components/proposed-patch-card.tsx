"use client"

import { AlertTriangle, CheckCircle2, Loader2, Sparkles } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import type { WorkforceBuilderPatch } from "@/types/workforce"

interface ProposedPatchCardProps {
  patch: WorkforceBuilderPatch | null
  messageId: number | null
  status?: string | null
  applying?: boolean
  onApply: (messageId: number, patch: WorkforceBuilderPatch) => Promise<void> | void
}

function formatOperationTitle(op: string) {
  return op
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ")
}

function operationSubtitle(operation: Record<string, unknown>) {
  if (operation.op === "add_existing_worker") {
    return `agent_id=${String(operation.agent_id ?? "")}`
  }
  if (operation.op === "add_worker_from_template") {
    return `template_id=${String(operation.template_id ?? "")}`
  }
  if (operation.op === "create_worker_agent") {
    const agent = operation.agent as Record<string, unknown> | undefined
    return `agent=${String(agent?.name ?? "")}`
  }
  if (operation.op === "update_worker" || operation.op === "remove_worker") {
    return `member_id=${String(operation.member_id ?? "")}`
  }
  return null
}

export function ProposedPatchCard({
  patch,
  messageId,
  status,
  applying = false,
  onApply,
}: ProposedPatchCardProps) {
  const alreadyApplied = status === "applied"
  const canApply = Boolean(
    patch && messageId && patch.operations.length > 0 && !applying && !alreadyApplied,
  )

  return (
    <Card>
      <CardHeader>
        <CardTitle>Proposed Patch</CardTitle>
        <CardDescription>
          Review the generated workforce changes before applying them.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {!patch ? (
          <div className="rounded-lg border border-dashed p-4 text-sm text-muted-foreground">
            No proposal yet. Send a request in Builder Chat to generate a patch.
          </div>
        ) : (
          <>
            <div className="rounded-xl border bg-muted/20 p-4">
              <div className="flex items-start gap-3">
                <Sparkles className="mt-0.5 size-4 text-primary" />
                <div>
                  <div className="font-medium">Summary</div>
                  <p className="mt-1 text-sm text-muted-foreground">{patch.summary}</p>
                </div>
              </div>
            </div>

            {patch.clarification ? (
              <Alert>
                <AlertTriangle className="size-4" />
                <AlertTitle>Clarification needed</AlertTitle>
                <AlertDescription>{patch.clarification}</AlertDescription>
              </Alert>
            ) : null}

            {patch.warnings.length > 0 ? (
              <Alert className="border-amber-200 bg-amber-50 text-amber-900">
                <AlertTriangle className="size-4" />
                <AlertTitle>Warnings</AlertTitle>
                <AlertDescription>
                  <div className="space-y-1">
                    {patch.warnings.map((warning, index) => (
                      <p key={`${warning}-${index}`}>{warning}</p>
                    ))}
                  </div>
                </AlertDescription>
              </Alert>
            ) : (
              <Alert>
                <CheckCircle2 className="size-4 text-emerald-600" />
                <AlertTitle>Ready to apply</AlertTitle>
                <AlertDescription>No destructive warning was detected in this patch.</AlertDescription>
              </Alert>
            )}

            <div className="space-y-3">
              <div className="flex items-center justify-between gap-3">
                <div className="font-medium">Operations</div>
                <Badge variant="outline">{patch.operations.length} change(s)</Badge>
              </div>
              {patch.operations.length === 0 ? (
                <div className="rounded-lg border border-dashed p-4 text-sm text-muted-foreground">
                  This proposal does not contain any executable operation yet.
                </div>
              ) : (
                patch.operations.map((operation, index) => (
                  <div key={`${operation.op}-${index}`} className="rounded-lg border bg-background p-4">
                    <div className="mb-3 flex items-center justify-between gap-3">
                      <div>
                        <div className="font-medium">{formatOperationTitle(operation.op)}</div>
                        {operationSubtitle(operation) ? (
                          <div className="mt-1 text-xs text-muted-foreground">
                            {operationSubtitle(operation)}
                          </div>
                        ) : null}
                      </div>
                      <Badge variant="secondary">#{index + 1}</Badge>
                    </div>
                    <pre className="overflow-x-auto whitespace-pre-wrap break-words text-xs leading-6 text-muted-foreground">
                      {JSON.stringify(operation, null, 2)}
                    </pre>
                  </div>
                ))
              )}
            </div>

            <Button
              className="w-full"
              disabled={!canApply}
              onClick={() => {
                if (patch && messageId) {
                  void onApply(messageId, patch)
                }
              }}
            >
              {applying ? (
                <>
                  <Loader2 className="size-4 animate-spin" />
                  Applying Changes...
                </>
              ) : alreadyApplied ? (
                "Already Applied"
              ) : (
                "Apply Changes"
              )}
            </Button>
          </>
        )}
      </CardContent>
    </Card>
  )
}
