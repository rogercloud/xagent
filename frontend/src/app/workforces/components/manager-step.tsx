"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Select } from "@/components/ui/select"
import type { WorkforceAgentOption } from "@/types/workforce"

interface ManagerStepProps {
  managerAgentId: string
  onManagerAgentIdChange: (value: string) => void
  agents: WorkforceAgentOption[]
}

export function ManagerStep({
  managerAgentId,
  onManagerAgentIdChange,
  agents,
}: ManagerStepProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Manager</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <Label>Select the manager agent</Label>
        <Select
          value={managerAgentId}
          onValueChange={onManagerAgentIdChange}
          placeholder="Choose an agent"
          options={agents.map((agent) => ({
            value: String(agent.id),
            label: agent.name,
            description: agent.description || undefined,
          }))}
        />
      </CardContent>
    </Card>
  )
}
