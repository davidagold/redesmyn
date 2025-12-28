const HARNESS_COMMAND_KEY = "rn.agentCommand"

export function getStoredHarnessCommand() {
  if (typeof window === "undefined") {
    return "codex"
  }
  return window.localStorage.getItem(HARNESS_COMMAND_KEY) ?? "codex"
}

export function storeHarnessCommand(value: string) {
  if (typeof window === "undefined") {
    return
  }
  window.localStorage.setItem(HARNESS_COMMAND_KEY, value)
}
