"use client"

import Link from "next/link"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { WorkforceDetail } from "@/types/workforce"

interface WorkforceSummaryProps {
  workforce: WorkforceDetail
}

function statusVariant(status: string) {
  if (status === "active") return "default"
  if (status === "archived") return "secondary"
  return "outline"
}

export function WorkforceSummary({ workforce }: WorkforceSummaryProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between gap-4">
          <span>{workforce.name}</span>
          <Badge variant={statusVariant(workforce.status)}>{workforce.status}</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-4 md:grid-cols-3">
          <div>
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Manager</div>
            <div className="mt-1 font-medium">{workforce.manager.name}</div>
            <div className="text-sm text-muted-foreground">
              {workforce.manager.description || "No description"}
            </div>
            <Link
              href={`/build/${workforce.manager.id}`}
              target="_blank"
              className="mt-2 inline-block text-sm text-primary hover:underline"
            >
              Open Agent Editor
            </Link>
          </div>
          <div>
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Workers</div>
            <div className="mt-1 font-medium">{workforce.workers.length}</div>
            <div className="text-sm text-muted-foreground">
              {workforce.workers.filter((item) => item.enabled).length} enabled
            </div>
          </div>
          <div>
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Updated</div>
            <div className="mt-1 font-medium">
              {workforce.updated_at ? new Date(workforce.updated_at).toLocaleString() : "N/A"}
            </div>
          </div>
        </div>

        {workforce.description && (
          <div>
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Description</div>
            <p className="mt-1 text-sm text-muted-foreground">{workforce.description}</p>
          </div>
        )}

        {workforce.manager_instructions && (
          <div>
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              Manager Instructions
            </div>
            <p className="mt-1 whitespace-pre-wrap text-sm text-muted-foreground">
              {workforce.manager_instructions}
            </p>
          </div>
        )}

        <div>
          <div className="text-xs uppercase tracking-wide text-muted-foreground">Workers</div>
          <div className="mt-3 grid gap-3">
            {workforce.workers.length === 0 ? (
              <div className="rounded-lg border border-dashed p-4 text-sm text-muted-foreground">
                No workers yet.
              </div>
            ) : (
              workforce.workers.map((worker) => (
                <div key={worker.id} className="rounded-lg border bg-background/40 p-4">
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <div className="font-medium">{worker.alias || worker.agent.name}</div>
                      <div className="text-sm text-muted-foreground">
                        {worker.agent.description || "No description"}
                      </div>
                      <div className="mt-2 flex flex-wrap items-center gap-2">
                        <Badge variant="outline">{worker.source_type}</Badge>
                        {worker.template_id ? (
                          <Badge variant="outline">{worker.template_id}</Badge>
                        ) : null}
                        <Link
                          href={`/build/${worker.agent.id}`}
                          target="_blank"
                          className="text-sm text-primary hover:underline"
                        >
                          Edit Agent
                        </Link>
                      </div>
                    </div>
                    <Badge variant={worker.enabled ? "default" : "secondary"}>
                      {worker.enabled ? "enabled" : "disabled"}
                    </Badge>
                  </div>
                  <p className="mt-3 text-sm text-muted-foreground">
                    {worker.assignment_instructions}
                  </p>
                </div>
              ))
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
