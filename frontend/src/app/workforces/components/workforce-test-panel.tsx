"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { runWorkforce } from "@/lib/workforces-api"

interface WorkforceTestPanelProps {
  workforceId: number
}

export function WorkforceTestPanel({ workforceId }: WorkforceTestPanelProps) {
  const router = useRouter()
  const [message, setMessage] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleRun = async () => {
    if (!message.trim()) return
    setLoading(true)
    setError(null)
    try {
      const result = await runWorkforce(workforceId, message.trim())
      router.push(result.redirect_url)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to run workforce")
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Test Workforce</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <Textarea
          placeholder="Describe the task you want the manager to coordinate."
          value={message}
          onChange={(event) => setMessage(event.target.value)}
          rows={8}
        />
        {error ? <div className="text-sm text-red-500">{error}</div> : null}
        <Button onClick={handleRun} disabled={loading || !message.trim()} className="w-full">
          {loading ? "Starting..." : "Run Workforce"}
        </Button>
      </CardContent>
    </Card>
  )
}
