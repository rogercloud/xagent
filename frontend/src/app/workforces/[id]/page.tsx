"use client"

import Link from "next/link"
import { useEffect, useState } from "react"
import { useParams } from "next/navigation"
import { Button } from "@/components/ui/button"
import { getWorkforce } from "@/lib/workforces-api"
import type { WorkforceDetail } from "@/types/workforce"
import { WorkforceSummary } from "../components/workforce-summary"

export default function WorkforceDetailPage() {
  const params = useParams()
  const [workforce, setWorkforce] = useState<WorkforceDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true)
        setError(null)
        const id = Array.isArray(params.id) ? params.id[0] : params.id
        if (!id) {
          setWorkforce(null)
          return
        }
        const data = await getWorkforce(id)
        setWorkforce(data)
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load workforce")
      } finally {
        setLoading(false)
      }
    }
    void load()
  }, [params.id])

  if (loading) return <div className="p-8 text-muted-foreground">Loading workforce...</div>
  if (error) return <div className="p-8 text-red-500">{error}</div>
  if (!workforce) return <div className="p-8 text-muted-foreground">Workforce not found.</div>

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 p-8">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold">{workforce.name}</h1>
          <p className="mt-2 text-muted-foreground">
            Review the current orchestration, then move into Builder, Canvas, or Run.
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
      <WorkforceSummary workforce={workforce} />
    </div>
  )
}
