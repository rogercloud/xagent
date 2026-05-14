"use client"

import { useEffect, useState } from "react"
import { useParams } from "next/navigation"
import { getWorkforce } from "@/lib/workforces-api"
import type { WorkforceDetail } from "@/types/workforce"
import { WorkforceSummary } from "../../components/workforce-summary"
import { WorkforceTestPanel } from "../../components/workforce-test-panel"

export default function WorkforceRunPage() {
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

  if (loading) return <div className="h-full overflow-y-auto p-4 text-muted-foreground sm:p-8">Loading run view...</div>
  if (error) return <div className="h-full overflow-y-auto p-4 text-red-500 sm:p-8">{error}</div>
  if (!workforce) return <div className="h-full overflow-y-auto p-4 text-muted-foreground sm:p-8">Workforce not found.</div>

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto grid w-full max-w-7xl gap-6 p-4 sm:p-8 lg:grid-cols-[1.1fr_0.9fr]">
        <WorkforceSummary workforce={workforce} />
        <WorkforceTestPanel workforceId={workforce.id} />
      </div>
    </div>
  )
}
