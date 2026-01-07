export type AgentKind = "generic" | "codex" | "claude_code"
export type AgentKindSelection = "auto" | AgentKind
export type AgentInterfaceMode = "interactive" | "structured"

export function labelForAgentKind(kind: AgentKind): string {
  switch (kind) {
    case "codex":
      return "Codex"
    case "claude_code":
      return "Claude Code"
    case "generic":
      return "Generic"
  }
}

export function labelForAgentKindSelection(
  selection: AgentKindSelection,
): string {
  switch (selection) {
    case "auto":
      return "Auto"
    default:
      return labelForAgentKind(selection)
  }
}

function _name(token: string): string {
  const normalized = token.replace(/\\/g, "/")
  return (normalized.split("/").pop() ?? normalized).toLowerCase()
}

function _directKindFromName(name: string): AgentKind | null {
  if (name === "codex") {
    return "codex"
  }
  if (name === "claude" || name === "claude-code") {
    return "claude_code"
  }
  if (name.startsWith("claude-code@")) {
    return "claude_code"
  }
  if (name.endsWith("@anthropic-ai/claude-code")) {
    return "claude_code"
  }
  return null
}

function _unwrapLauncherArgv(argv: string[]): string[] {
  if (argv.length === 0) {
    return []
  }
  const name0 = _name(argv[0])
  if (name0 === "uv") {
    const runIndex = argv.indexOf("run")
    return runIndex >= 0 ? argv.slice(runIndex + 1) : argv.slice(1)
  }
  if (name0 === "npx" || name0 === "bunx") {
    return argv.slice(1)
  }
  if (name0 === "npm") {
    if (argv[1] === "exec" || argv[1] === "x") {
      return argv.slice(2)
    }
  }
  if (name0 === "pnpm") {
    if (argv[1] === "dlx" || argv[1] === "exec") {
      return argv.slice(2)
    }
  }
  if (name0 === "yarn") {
    if (argv[1] === "dlx" || argv[1] === "exec" || argv[1] === "run") {
      return argv.slice(2)
    }
  }
  return argv
}

function _agentInvocationArgv(argv: string[], kind: AgentKind): string[] {
  const unwrapped = _unwrapLauncherArgv(argv)
  if (kind === "codex") {
    const idx = unwrapped.findIndex((token) => _name(token) === "codex")
    return idx >= 0 ? unwrapped.slice(idx) : unwrapped
  }
  if (kind === "claude_code") {
    const idx = unwrapped.findIndex((token) => {
      const name = _name(token)
      if (name === "claude" || name === "claude-code") {
        return true
      }
      if (name.startsWith("claude-code@")) {
        return true
      }
      const normalized = token.replace(/\\/g, "/").toLowerCase()
      return (
        normalized.endsWith("@anthropic-ai/claude-code") ||
        normalized.startsWith("@anthropic-ai/claude-code@")
      )
    })
    return idx >= 0 ? unwrapped.slice(idx) : unwrapped
  }
  return unwrapped
}

function _hasFlagValue(argv: string[], flag: string, value: string): boolean {
  for (let idx = 0; idx < argv.length; idx += 1) {
    const token = argv[idx] ?? ""
    if (token === flag && argv[idx + 1] === value) {
      return true
    }
    if (token.startsWith(`${flag}=`) && token.split("=", 2)[1] === value) {
      return true
    }
  }
  return false
}

function _isCodexStructuredArgv(argv: string[]): boolean {
  const idx = argv.findIndex((token) => _name(token) === "codex")
  if (idx < 0) {
    return false
  }
  const execIndex = argv.indexOf("exec", idx + 1)
  if (execIndex < 0) {
    return false
  }
  return argv.slice(execIndex + 1).includes("--json")
}

export function inferAgentKindFromCommand(command: string): AgentKind {
  const raw = command.trim()
  if (!raw) {
    return "generic"
  }

  const argv = raw.split(/\s+/).filter(Boolean)
  if (argv.length === 0) {
    return "generic"
  }

  const name0 = _name(argv[0])
  const direct = _directKindFromName(name0)
  if (direct) {
    return direct
  }

  let tokens: string[] = []
  if (name0 === "uv") {
    const runIndex = argv.indexOf("run")
    tokens = runIndex >= 0 ? argv.slice(runIndex + 1) : argv.slice(1)
  } else if (name0 === "npx" || name0 === "bunx") {
    tokens = argv.slice(1)
  } else if (name0 === "npm") {
    if (argv[1] === "exec" || argv[1] === "x") {
      tokens = argv.slice(2)
    }
  } else if (name0 === "pnpm") {
    if (argv[1] === "dlx" || argv[1] === "exec") {
      tokens = argv.slice(2)
    }
  } else if (name0 === "yarn") {
    if (argv[1] === "dlx" || argv[1] === "exec" || argv[1] === "run") {
      tokens = argv.slice(2)
    }
  }

  for (const token of tokens) {
    if (!token || token.startsWith("-")) {
      continue
    }
    const kind = _directKindFromName(_name(token))
    if (kind) {
      return kind
    }
  }

  return "generic"
}

export function inferStructuredAgentFromCommand(
  command: string,
): AgentKind | null {
  const raw = command.trim()
  if (!raw) {
    return null
  }
  const argv = raw.split(/\s+/).filter(Boolean)
  if (argv.length === 0) {
    return null
  }

  const kind = inferAgentKindFromCommand(raw)
  if (kind === "generic") {
    return null
  }

  const agentArgv = _agentInvocationArgv(argv, kind)
  if (kind === "codex") {
    return _isCodexStructuredArgv(agentArgv) ? "codex" : null
  }
  if (kind === "claude_code") {
    return _hasFlagValue(agentArgv, "--output-format", "stream-json")
      ? "claude_code"
      : null
  }
  return null
}
