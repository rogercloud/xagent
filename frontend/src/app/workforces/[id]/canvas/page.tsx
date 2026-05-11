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

  if (loading) return <div className="p-8 text-muted-foreground">Loading canvas...</div>
  if (error) return <div className="p-8 text-red-500">{error}</div>
  if (!canvas) return <div className="p-8 text-muted-foreground">Canvas unavailable.</div>

  return (
    <div className="mx-auto w-full max-w-7xl p-8">
      <WorkforceCanvas canvas={canvas} />
    </div>
  )
}
