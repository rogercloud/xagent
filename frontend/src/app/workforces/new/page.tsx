"use client"

import { useState } from "react"
import { WorkforcePromptCreator } from "../components/workforce-prompt-creator"
import { WorkforceWizard } from "../components/workforce-wizard"

export default function NewWorkforcePage() {
  const [mode, setMode] = useState<"prompt" | "manual">("prompt")

  if (mode === "manual") {
    return <WorkforceWizard onPromptSetup={() => setMode("prompt")} />
  }

  return <WorkforcePromptCreator onManualSetup={() => setMode("manual")} />
}
