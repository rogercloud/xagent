"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import { Bot, Loader2, User } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Textarea } from "@/components/ui/textarea"
import { cn, formatDate } from "@/lib/utils"
import type { WorkforceBuilderMessage } from "@/types/workforce"

interface WorkforceBuilderChatProps {
  messages: WorkforceBuilderMessage[]
  loading?: boolean
  submitting?: boolean
  onSubmit: (message: string) => Promise<void> | void
}

function roleIcon(role: string) {
  return role === "assistant" ? Bot : User
}

function roleLabel(role: string) {
  return role === "assistant" ? "Builder" : "You"
}

export function WorkforceBuilderChat({
  messages,
  loading = false,
  submitting = false,
  onSubmit,
}: WorkforceBuilderChatProps) {
  const [message, setMessage] = useState("")
  const containerRef = useRef<HTMLDivElement | null>(null)

  const canSubmit = useMemo(() => message.trim().length > 0 && !submitting, [message, submitting])

  useEffect(() => {
    const viewport = containerRef.current?.querySelector(
      "[data-radix-scroll-area-viewport]",
    ) as HTMLDivElement | null
    if (viewport) {
      viewport.scrollTop = viewport.scrollHeight
    }
  }, [messages, submitting])

  const handleSubmit = async () => {
    const value = message.trim()
    if (!value || submitting) return
    setMessage("")
    await onSubmit(value)
  }

  return (
    <Card className="h-full min-h-[640px]">
      <CardHeader>
        <CardTitle>Builder Chat</CardTitle>
        <CardDescription>
          Describe the workforce change you want. The builder will turn it into a reviewable patch.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex h-full flex-col gap-4">
        <div ref={containerRef}>
          <ScrollArea className="h-[440px] rounded-lg border bg-muted/20">
            <div className="space-y-4 p-4">
              {loading ? (
                <div className="text-sm text-muted-foreground">Loading builder history...</div>
              ) : messages.length === 0 ? (
                <div className="rounded-lg border border-dashed bg-background p-4 text-sm text-muted-foreground">
                  Start with something like: add worker Research Agent to handle competitor
                  research.
                </div>
              ) : (
                messages.map((item) => {
                  const Icon = roleIcon(item.role)
                  return (
                    <div
                      key={item.id}
                      className={cn(
                        "flex gap-3",
                        item.role === "user" ? "justify-end" : "justify-start",
                      )}
                    >
                      {item.role !== "user" ? (
                        <div className="mt-1 flex size-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary">
                          <Icon className="size-4" />
                        </div>
                      ) : null}
                      <div
                        className={cn(
                          "max-w-[85%] rounded-2xl border px-4 py-3",
                          item.role === "user"
                            ? "border-primary/20 bg-primary text-primary-foreground"
                            : "bg-background",
                        )}
                      >
                        <div
                          className={cn(
                            "mb-2 flex items-center gap-2 text-xs",
                            item.role === "user"
                              ? "text-primary-foreground/80"
                              : "text-muted-foreground",
                          )}
                        >
                          <span className="font-medium">{roleLabel(item.role)}</span>
                          {item.created_at ? <span>{formatDate(item.created_at)}</span> : null}
                        </div>
                        <div className="whitespace-pre-wrap text-sm leading-6">{item.content}</div>
                      </div>
                      {item.role === "user" ? (
                        <div className="mt-1 flex size-8 shrink-0 items-center justify-center rounded-full bg-secondary text-secondary-foreground">
                          <Icon className="size-4" />
                        </div>
                      ) : null}
                    </div>
                  )
                })
              )}
              {submitting ? (
                <div className="flex items-center gap-3 rounded-lg border bg-background px-4 py-3 text-sm text-muted-foreground">
                  <Loader2 className="size-4 animate-spin" />
                  <span>Preparing a proposed patch...</span>
                </div>
              ) : null}
            </div>
          </ScrollArea>
        </div>

        <div className="space-y-3">
          <Textarea
            placeholder='Rename this workforce to "Launch Crew" and make Writer focus on launch email copy.'
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            rows={6}
            onKeyDown={(event) => {
              if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
                event.preventDefault()
                void handleSubmit()
              }
            }}
          />
          <div className="flex items-center justify-between gap-3">
            <div className="text-xs text-muted-foreground">Press Ctrl/Cmd + Enter to send.</div>
            <Button onClick={() => void handleSubmit()} disabled={!canSubmit}>
              {submitting ? "Proposing..." : "Propose Changes"}
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
