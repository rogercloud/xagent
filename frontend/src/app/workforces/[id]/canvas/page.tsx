"use client"

import { useEffect, useState } from "react"
import { useParams } from "next/navigation"
import { getWorkforceCanvas } from "@/lib/workforces-api"
import type { WorkforceCanvasResponse } from "@/types/workforce"
import { WorkforceCanvas } from "../../components/workforce-canvas"

export default function WorkforceCanvasPage() {
  const params = useParams()
  const [canvas, setCanvas] = useState<WorkforceCanvasResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true)
        setError(null)
        const id = Array.isArray(params.id) ? params.id[0] : params.id
        if (!id) {
          setCanvas(null)
          return
        }
        const data = await getWorkforceCanvas(id)
        setCanvas(data)
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load workforce canvas")
      } finally {
        setLoading(false)
      }
    }
    void load()
  }, [params.id])

  if (loading) return <div className="h-full overflow-y-auto p-4 text-muted-foreground sm:p-8">Loading canvas...</div>
  if (error) return <div className="h-full overflow-y-auto p-4 text-red-500 sm:p-8">{error}</div>
  if (!canvas) return <div className="h-full overflow-y-auto p-4 text-muted-foreground sm:p-8">Canvas unavailable.</div>

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto w-full max-w-7xl p-4 sm:p-8">
        <WorkforceCanvas canvas={canvas} />
      </div>
    </div>
  )
}
