import { describe, expect, it } from "vitest"

import { getTaskBackHref } from "./task-navigation"

describe("getTaskBackHref", () => {
  it("prefers workforce run pages for workforce tasks", () => {
    expect(getTaskBackHref({ agentId: 10, workforceId: 20 })).toBe("/workforces/20/run")
  })

  it("falls back to the agent page for regular agent tasks", () => {
    expect(getTaskBackHref({ agentId: 10 })).toBe("/agent/10")
  })

  it("falls back to the task list without a task source", () => {
    expect(getTaskBackHref(null)).toBe("/task")
  })
})
