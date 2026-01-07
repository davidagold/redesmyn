export type AgentKind = "generic" | "codex" | "claude_code"
export type AgentKindSelection = "auto" | AgentKind

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
  if (name.endsWith("@anthropic-ai/claude-code")) {
    return "claude_code"
  }
  return null
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
