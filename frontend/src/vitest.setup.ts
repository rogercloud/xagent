import "@testing-library/jest-dom/vitest"
import { beforeEach, vi } from "vitest"

// Vitest copies jsdom's window properties onto the global through accessors
// whose setter keeps the last assigned value for good. React 18's dev build
// assigns window.event while it dispatches a React event handler, so the first
// fireEvent.click otherwise stays in window.event for the rest of a test file.
// React reads window.event to prioritize updates made outside its own event
// dispatch, and would then treat every later promise callback or timer update
// as a discrete click: a sync render that jumps ahead of default-priority
// updates and hides missing act()/waitFor() in tests. Read the live jsdom value
// instead, and let an assignment replace the property the way the browser's
// [Replaceable] window.event does, so React's restore of the saved descriptor
// undoes it.
const jsdomWindow = (globalThis as { jsdom?: { window: Window } }).jsdom?.window
if (jsdomWindow) {
  Object.defineProperty(globalThis, "event", {
    configurable: true,
    get: () => jsdomWindow.event,
    set: (value: unknown) => {
      Object.defineProperty(globalThis, "event", { configurable: true, writable: true, value })
    },
  })
}

// jsdom does not implement blob URL APIs; InlineFilePreview authenticated image
// previews rely on them in tests that assert blob: src values or spy on revoke.
URL.createObjectURL = vi.fn(
  (blob: Blob) => `blob:mock-${blob.size}`
) as typeof URL.createObjectURL
URL.revokeObjectURL = vi.fn() as typeof URL.revokeObjectURL

class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}

globalThis.ResizeObserver = ResizeObserverMock

// jsdom does not implement DOMMatrixReadOnly; @xyflow/react reads the scale
// off it when recomputing node internals (e.g. after a node's type changes).
class DOMMatrixReadOnlyMock {
  m22: number

  constructor(transform: string) {
    const scale = transform?.match(/scale\(([0-9.]+)\)/)?.[1]
    this.m22 = scale !== undefined ? Number(scale) : 1
  }
}

// @ts-expect-error partial DOMMatrixReadOnly polyfill, sufficient for @xyflow/react in tests
globalThis.DOMMatrixReadOnly = DOMMatrixReadOnlyMock

class LocalStorageMock implements Storage {
  private store = new Map<string, string>()

  get length() {
    return this.store.size
  }

  clear() {
    this.store.clear()
  }

  getItem(key: string) {
    return this.store.get(key) ?? null
  }

  key(index: number) {
    return Array.from(this.store.keys())[index] ?? null
  }

  removeItem(key: string) {
    this.store.delete(key)
  }

  setItem(key: string, value: string) {
    this.store.set(key, String(value))
  }
}

Object.defineProperty(globalThis, "localStorage", {
  configurable: true,
  value: new LocalStorageMock(),
})

Object.defineProperty(window, "localStorage", {
  configurable: true,
  value: globalThis.localStorage,
})

beforeEach(() => {
  localStorage.clear()
})
