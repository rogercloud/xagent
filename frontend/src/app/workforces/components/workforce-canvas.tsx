"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useI18n } from "@/contexts/i18n-context"
import type { WorkforceCanvasResponse } from "@/types/workforce"

interface WorkforceCanvasProps {
  canvas: WorkforceCanvasResponse
}

export function WorkforceCanvas({ canvas }: WorkforceCanvasProps) {
  const { t } = useI18n()

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t("workforces.actions.canvas")}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid gap-3 md:grid-cols-3">
          {canvas.nodes.map((node) => (
            <div key={node.id} className="rounded-xl border bg-background/40 p-4">
              <div className="text-xs uppercase tracking-wide text-muted-foreground">
                {node.type}
              </div>
              <div className="mt-2 font-medium">{node.label}</div>
              {node.enabled === false ? (
                <div className="mt-2 text-xs text-muted-foreground">{t("workforces.status.disabled")}</div>
              ) : null}
            </div>
          ))}
        </div>
        <div>
          <div className="mb-2 text-xs uppercase tracking-wide text-muted-foreground">
            {t("workforces.canvas.connections")}
          </div>
          <div className="space-y-2">
            {canvas.edges.map((edge) => (
              <div key={edge.id} className="rounded-lg border px-3 py-2 text-sm text-muted-foreground">
                {edge.source} → {edge.target}
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
