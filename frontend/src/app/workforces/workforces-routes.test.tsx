/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const getWorkforceMock = vi.hoisted(() => vi.fn())
const listWorkforcesMock = vi.hoisted(() => vi.fn())
const runWorkforceMock = vi.hoisted(() => vi.fn())
const routerPushMock = vi.hoisted(() => vi.fn())
const paramsMock = vi.hoisted(() => ({ id: "42" as string | string[] | undefined }))
const translateMock = vi.hoisted(
  () => (key: string, vars?: Record<string, string | number>) => {
    if (!vars) return key
    return Object.entries(vars).reduce(
      (value, [name, replacement]) =>
        value.replace(`{${name}}`, String(replacement)),
      key,
    )
  },
)

vi.mock("next/navigation", () => ({
  useParams: () => paramsMock,
  useRouter: () => ({ push: routerPushMock }),
}))

vi.mock("next/link", () => ({
  default: ({
    children,
    href,
    ...props
  }: React.AnchorHTMLAttributes<HTMLAnchorElement> & { href: string }) => (
    <a href={href} {...props}>
      {children}
    </a>
  ),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    locale: "en-US",
    t: translateMock,
  }),
}))

vi.mock("@/lib/workforces-api", () => ({
  getWorkforce: getWorkforceMock,
  listWorkforces: listWorkforcesMock,
  runWorkforce: runWorkforceMock,
}))

import WorkforcesPage from "./page"
import WorkforceRunPage from "./[id]/run/page"
import { getNavigationGroupsForUser } from "@/components/layout/sidebar"
import type { WorkforceDetail, WorkforceListResponse } from "@/types/workforce"

const workforceDetail: WorkforceDetail = {
  id: 42,
  name: "Launch Workforce",
  description: null,
  status: "active",
  manager: {
    id: 7,
    name: "Manager Agent",
    description: null,
    logo_url: null,
    status: "published",
  },
  manager_instructions: null,
  workers: [],
  canvas_layout: null,
  scope_type: "user",
  scope_id: "1",
  owner_user_id: 1,
  created_at: null,
  updated_at: null,
}

const listResponse: WorkforceListResponse = {
  items: [
    {
      id: 42,
      name: "Launch Workforce",
      description: "Coordinate launch work",
      status: "active",
      manager: {
        id: 7,
        name: "Manager Agent",
        logo_url: null,
      },
      worker_count: 2,
      last_run: {
        id: 9,
        task_id: 99,
        status: "completed",
        created_at: null,
      },
      created_at: null,
      updated_at: "2026-05-27T00:00:00Z",
    },
  ],
  total: 1,
  page: 1,
  size: 10,
  pages: 1,
}

describe("workforce route entry points", () => {
  beforeEach(() => {
    getWorkforceMock.mockReset()
    listWorkforcesMock.mockReset()
    runWorkforceMock.mockReset()
    routerPushMock.mockReset()
    paramsMock.id = "42"
  })

  afterEach(() => {
    cleanup()
  })

  it("adds the visible sidebar entry for workforces", () => {
    const agentDevelopment = getNavigationGroupsForUser(null)[0]

    expect(agentDevelopment.items).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          href: "/workforces",
          nameKey: "nav.workforces",
        }),
      ]),
    )
  })

  it("renders the workforce list with PR7 route links", async () => {
    listWorkforcesMock.mockResolvedValueOnce(listResponse)

    render(<WorkforcesPage />)

    expect(await screen.findByText("Launch Workforce")).toBeInTheDocument()
    expect(screen.getByRole("link", { name: /workforces.actions.new/ })).toHaveAttribute(
      "href",
      "/workforces/new",
    )
    expect(screen.getByRole("link", { name: /Launch Workforce/ })).toHaveAttribute(
      "href",
      "/workforces/42",
    )
    expect(screen.getByRole("link", { name: /workforces.actions.run/ })).toHaveAttribute(
      "href",
      "/workforces/42/run",
    )
    expect(screen.getByRole("link", { name: /workforces.actions.builder/ })).toHaveAttribute(
      "href",
      "/workforces/42/builder",
    )
    expect(screen.getByRole("link", { name: /workforces.actions.canvas/ })).toHaveAttribute(
      "href",
      "/workforces/42/canvas",
    )
  })

  it("runs a workforce and redirects to the created task", async () => {
    getWorkforceMock.mockResolvedValueOnce(workforceDetail)
    runWorkforceMock.mockResolvedValueOnce({
      workforce_run_id: 5,
      task_id: 99,
      status: "running",
      redirect_url: "/task/99",
    })

    render(<WorkforceRunPage />)

    expect(await screen.findByText("Launch Workforce")).toBeInTheDocument()

    fireEvent.change(screen.getByPlaceholderText("workforces.run.placeholder"), {
      target: { value: " Draft launch plan " },
    })
    fireEvent.click(screen.getByText("workforces.actions.runWorkforce"))

    await waitFor(() => {
      expect(runWorkforceMock).toHaveBeenCalledWith(42, {
        message: "Draft launch plan",
      })
    })
    expect(routerPushMock).toHaveBeenCalledWith("/task/99")
  })
})
