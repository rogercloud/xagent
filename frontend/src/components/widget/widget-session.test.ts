import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { pathToFileURL } from "node:url"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const widgetScriptPath = resolve(process.cwd(), "public/widget.js")
const widgetScript = readFileSync(widgetScriptPath, "utf8")
const widgetScriptUrl = pathToFileURL(widgetScriptPath).href

export const HOST = "https://chat.example"
export const GRANT = "eyJhbGciOiJIUzI1NiJ9.grant-one.sig"
const EXCHANGE_URL = `${HOST}/v1/external/chat/sessions`
const RECONNECT_URL = `${HOST}/v1/external/chat/sessions/reconnect`

const fetchMock = vi.fn()

function runWidget(attributes: Record<string, string>) {
  const script = document.createElement("script")
  script.src = `${HOST}/widget.js`
  for (const [name, value] of Object.entries(attributes)) {
    script.setAttribute(name, value)
  }
  document.body.appendChild(script)
  Object.defineProperty(document, "currentScript", { configurable: true, value: script })
  window.eval(`${widgetScript}\n//# sourceURL=${widgetScriptUrl}`)
  return script
}

function iframeEl(): HTMLIFrameElement | null {
  return document.querySelector<HTMLIFrameElement>(".xagent-widget-iframe")
}

function spyOnIframePostMessage() {
  const frame = iframeEl()
  if (!frame?.contentWindow) throw new Error("iframe not mounted")
  return vi.spyOn(frame.contentWindow, "postMessage")
}

function fromIframe(type: string, extra: Record<string, unknown> = {}) {
  const frame = iframeEl()
  window.dispatchEvent(new MessageEvent("message", {
    data: { xagent: true, v: 1, type, ...extra },
    origin: HOST,
    source: frame?.contentWindow as Window,
  }))
}

function jsonResponse(status: number, body: unknown, headers: Record<string, string> = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...headers },
  })
}

function errorResponse(status: number, code: string) {
  return jsonResponse(status, { error: { code, message: "nope" } })
}

// vi.waitFor's condition here (fetchMock call count) goes true the instant
// fetch() is invoked, which happens synchronously during runWidget(). That
// resolves the waitFor promise before the mocked Response's real .json()
// body-read (an inherently async, multi-microtask-tick operation) has settled.
// A macrotask flush lets every already-scheduled microtask (the fetch/json/
// applySession chain) drain before we simulate the iframe's "ready" signal.
function flushAsync() {
  return new Promise<void>((resolve) => setTimeout(resolve, 0))
}

function exchangeBody(overrides: Record<string, unknown> = {}) {
  return {
    session_token: "st_first",
    session_token_expires_at: new Date(Date.now() + 15 * 60_000).toISOString(),
    reconnect_token: "rt_first",
    session: {
      absolute_expires_at: new Date(Date.now() + 8 * 3_600_000).toISOString(),
      agent: {
        id: 42,
        name: "Care Assistant",
        description: "Rosters and shifts",
        logo_url: `${HOST}/logo.png`,
        suggested_prompts: ["What is my next shift?"],
      },
    },
    ...overrides,
  }
}

describe("widget session mode", () => {
  let currentScriptDescriptor: PropertyDescriptor | undefined

  beforeEach(() => {
    currentScriptDescriptor = Object.getOwnPropertyDescriptor(document, "currentScript")
    document.head.innerHTML = ""
    document.body.innerHTML = ""
    localStorage.clear()
    // The grant dedupe registry lives on window and survives between tests in a file.
    Reflect.deleteProperty(window as unknown as Record<string, unknown>, "__xagentWidgetGrants")
    vi.stubGlobal("fetch", fetchMock)
    fetchMock.mockReset()
  })

  afterEach(() => {
    if (currentScriptDescriptor) {
      Object.defineProperty(document, "currentScript", currentScriptDescriptor)
    } else {
      Reflect.deleteProperty(document, "currentScript")
    }
    vi.useRealTimers()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it("harness boots the guest path unchanged", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(200, { ticket: "t", agent_id: 17 }))
    runWidget({ "data-widget-key": "widget-secret" })
    await vi.waitFor(() => {
      expect(iframeEl()?.src).toContain("/widget/chat/default")
    })
  })

  it("navigates the iframe to the session URL and exchanges the grant immediately", () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(200, exchangeBody()))

    runWidget({ "data-encrypted-context": GRANT })

    expect(iframeEl()?.src).toBe(`${HOST}/widget/chat/session`)
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock).toHaveBeenCalledWith(EXCHANGE_URL, expect.objectContaining({
      method: "POST",
      credentials: "omit",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ encrypted_context: GRANT }),
    }))
  })

  it("fails closed on an empty grant attribute with no DOM and no network", () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)

    runWidget({ "data-encrypted-context": "   " })

    expect(fetchMock).not.toHaveBeenCalled()
    expect(document.querySelector(".xagent-widget-container")).toBeNull()
    expect(document.head.querySelector("style")).toBeNull()
    expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining("[grant_malformed]"))
  })

  it.each(["data-widget-key", "data-token"])(
    "fails closed when %s coexists with the grant",
    (legacyAttribute) => {
      const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)

      runWidget({ "data-encrypted-context": GRANT, [legacyAttribute]: "legacy" })

      expect(fetchMock).not.toHaveBeenCalled()
      expect(document.querySelector(".xagent-widget-container")).toBeNull()
      expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining("[attribute_conflict]"))
    },
  )

  it("keeps cosmetic attributes working in session mode", () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(200, exchangeBody()))

    runWidget({ "data-encrypted-context": GRANT, "data-button-color": "rgb(1, 2, 3)" })

    expect(document.head.querySelector("style")?.innerHTML).toContain("rgb(1, 2, 3)")
  })

  it("never writes a guest id in session mode", () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(200, exchangeBody()))

    runWidget({ "data-encrypted-context": GRANT })

    expect(localStorage.getItem("xagent_guest_id")).toBeNull()
  })

  it("removes the grant attribute from the DOM once it is read", () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(200, exchangeBody()))

    const script = runWidget({ "data-encrypted-context": GRANT })

    expect(script.hasAttribute("data-encrypted-context")).toBe(false)
  })

  it("ignores a repeat injection of the same grant but allows a different one", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined)
    fetchMock.mockResolvedValue(jsonResponse(200, exchangeBody()))

    runWidget({ "data-encrypted-context": GRANT })
    runWidget({ "data-encrypted-context": GRANT })

    expect(document.querySelectorAll(".xagent-widget-container")).toHaveLength(1)
    expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining("[duplicate_init]"))

    runWidget({ "data-encrypted-context": `${GRANT}-other` })

    expect(document.querySelectorAll(".xagent-widget-container")).toHaveLength(2)
  })

  it("pushes session_update to the iframe once it announces ready", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(200, exchangeBody()))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()

    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    await flushAsync()
    fromIframe("ready")

    expect(post).toHaveBeenCalledTimes(1)
    const [message, targetOrigin] = post.mock.calls[0]
    expect(targetOrigin).toBe(HOST)
    expect(message).toMatchObject({
      xagent: true,
      v: 1,
      type: "session_update",
      session_token: "st_first",
      agent: { id: 42, name: "Care Assistant" },
    })
    expect(message).not.toHaveProperty("reconnect_token")
  })

  it("holds only the latest state until ready arrives", async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody()))
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody({
        session_token: "st_second",
        reconnect_token: "rt_second",
      })))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()

    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    await flushAsync()
    fromIframe("reconnect_request", { reason: "ws_closed" })
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    await flushAsync()

    expect(post).not.toHaveBeenCalled()
    fromIframe("ready")

    expect(post).toHaveBeenCalledTimes(1)
    expect(post.mock.calls[0][0]).toMatchObject({ session_token: "st_second" })
  })

  it("re-sends the current state on every ready", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(200, exchangeBody()))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()

    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    await flushAsync()
    fromIframe("ready")
    fromIframe("ready")

    expect(post).toHaveBeenCalledTimes(2)
    expect(post.mock.calls[1][0]).toMatchObject({ type: "session_update", session_token: "st_first" })
  })

  it("ignores messages from a foreign origin, a foreign source, or a foreign shape", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(200, exchangeBody()))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))

    window.dispatchEvent(new MessageEvent("message", {
      data: { xagent: true, v: 1, type: "ready" },
      origin: "https://evil.example",
      source: iframeEl()?.contentWindow as Window,
    }))
    window.dispatchEvent(new MessageEvent("message", {
      data: { xagent: true, v: 1, type: "ready" },
      origin: HOST,
      source: window,
    }))
    fromIframe("ready", { v: 2 })
    window.dispatchEvent(new MessageEvent("message", {
      data: { type: "ready" },
      origin: HOST,
      source: iframeEl()?.contentWindow as Window,
    }))

    expect(post).not.toHaveBeenCalled()
  })
})
