"use client"

import Link from "next/link"
import { useCallback, useEffect, useState } from "react"
import { ChevronLeft, ChevronRight, Layers, Play, Plus } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { SearchInput } from "@/components/ui/search-input"
import { listWorkforces } from "@/lib/workforces-api"
import type { WorkforceListItem } from "@/types/workforce"

export default function WorkforcesPage() {
  const [items, setItems] = useState<WorkforceListItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState("")
  const [page, setPage] = useState(1)
  const [pages, setPages] = useState(1)
  const [total, setTotal] = useState(0)
  const pageSize = 10

  const load = useCallback(async (nextPage: number, nextSearch: string) => {
    try {
      setLoading(true)
      setError(null)
      const data = await listWorkforces({ page: nextPage, size: pageSize, search: nextSearch })
      setItems(data.items)
      setPages(data.pages)
      setTotal(data.total)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load workforces")
      setItems([])
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load(page, search)
  }, [load, page, search])

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 p-4 sm:p-8">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <div className="inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs uppercase tracking-[0.2em] text-muted-foreground">
            <Layers className="h-3.5 w-3.5" />
            Workforce
          </div>
          <h1 className="mt-4 text-3xl font-bold">Workforces</h1>
          <p className="mt-2 max-w-2xl text-muted-foreground">
            Create manager-led multi-agent work groups, keep worker roles explicit, and
            run them from a single entry point.
          </p>
        </div>
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
          <SearchInput
            placeholder="Search workforces..."
            value={search}
            onChange={(value) => {
              setSearch(value)
              setPage(1)
            }}
            containerClassName="w-full sm:w-80"
          />
          <Link href="/workforces/new">
            <Button>
              <Plus className="mr-2 h-4 w-4" />
              New Workforce
            </Button>
          </Link>
        </div>
      </div>

      {loading ? <div className="p-8 text-muted-foreground">Loading workforces...</div> : null}
      {error ? <div className="p-8 text-red-500">{error}</div> : null}

      {!loading && !error ? (
        items.length === 0 ? (
          <Card className="border-dashed">
            <CardContent className="flex flex-col items-center gap-4 p-12 text-center">
              <div className="text-lg font-medium">No workforces yet</div>
              <p className="max-w-xl text-sm text-muted-foreground">
                Start with a manager agent, add a few workers, and return here to launch
                coordinated runs.
              </p>
              <Link href="/workforces/new">
                <Button>Create your first workforce</Button>
              </Link>
            </CardContent>
          </Card>
        ) : (
          <>
            <div className="grid gap-4">
              {items.map((item) => (
                <Card key={item.id} className="overflow-hidden">
                  <CardContent className="p-0">
                    <div className="grid gap-6 p-6 lg:grid-cols-[1.4fr_0.6fr] lg:items-center">
                      <div className="space-y-4">
                        <div className="flex flex-wrap items-center gap-3">
                          <Link
                            href={`/workforces/${item.id}`}
                            className="text-xl font-semibold hover:underline"
                          >
                            {item.name}
                          </Link>
                          <span className="rounded-full border px-2.5 py-1 text-xs capitalize text-muted-foreground">
                            {item.status}
                          </span>
                        </div>
                        <p className="text-sm text-muted-foreground">
                          {item.description || "No description"}
                        </p>
                        <div className="flex flex-wrap gap-6 text-sm text-muted-foreground">
                          <span>Manager: {item.manager.name}</span>
                          <span>Workers: {item.worker_count}</span>
                          <span>
                            Last update:{" "}
                            {item.updated_at ? new Date(item.updated_at).toLocaleString() : "N/A"}
                          </span>
                        </div>
                      </div>
                      <div className="flex flex-col gap-3 lg:items-end">
                        <Link href={`/workforces/${item.id}/run`}>
                          <Button className="w-full lg:w-auto">
                            <Play className="mr-2 h-4 w-4" />
                            Run
                          </Button>
                        </Link>
                        <div className="flex gap-3">
                          <Link href={`/workforces/${item.id}`}>
                            <Button variant="outline">Details</Button>
                          </Link>
                          <Link href={`/workforces/${item.id}/builder`}>
                            <Button variant="outline">Builder</Button>
                          </Link>
                          <Link href={`/workforces/${item.id}/canvas`}>
                            <Button variant="outline">Canvas</Button>
                          </Link>
                        </div>
                        {item.last_run ? (
                          <div className="text-xs text-muted-foreground">
                            Last run #{item.last_run.id}{item.last_run.task_id != null ? ` · task #${item.last_run.task_id}` : ""} · {item.last_run.status}
                          </div>
                        ) : (
                          <div className="text-xs text-muted-foreground">No runs yet</div>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            {pages > 1 ? (
              <div className="flex items-center justify-between">
                <div className="text-sm text-muted-foreground">
                  Showing {(page - 1) * pageSize + 1}–{Math.min(page * pageSize, total)} of {total}
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setPage((current) => current - 1)}
                    disabled={page <= 1}
                  >
                    <ChevronLeft className="mr-1 h-4 w-4" />
                    Prev
                  </Button>
                  <span className="text-sm text-muted-foreground">
                    Page {page} of {pages}
                  </span>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setPage((current) => current + 1)}
                    disabled={page >= pages}
                  >
                    Next
                    <ChevronRight className="ml-1 h-4 w-4" />
                  </Button>
                </div>
              </div>
            ) : null}
          </>
        )
      ) : null}
      </div>
    </div>
  )
}
