import { useCallback, useEffect, useMemo, useState } from "react"
import {
  fetchEpicGraph,
  fetchEpics,
  fetchStatus,
  type ApiStatus,
  type Epic,
  type EpicGraph,
} from "@/api"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Markdown } from "@/components/markdown"
import {
  applyThemePreference,
  getStoredThemePreference,
  setStoredThemePreference,
  type ThemePreference,
} from "@/lib/theme"
import { ChevronDown, Monitor, Moon, Sun } from "lucide-react"

type GraphNode = EpicGraph["nodes"][number]

function App() {
  const [theme, setTheme] = useState<ThemePreference>(
    () => getStoredThemePreference() ?? "dark",
  )
  const [status, setStatus] = useState<ApiStatus | null>(null)
  const [epics, setEpics] = useState<Epic[]>([])
  const [selectedEpicId, setSelectedEpicId] = useState<number | null>(null)
  const [graph, setGraph] = useState<EpicGraph | null>(null)
  const [selectedNodeId, setSelectedNodeId] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    applyThemePreference(theme)
    setStoredThemePreference(theme)
  }, [theme])

  useEffect(() => {
    if (theme !== "system") {
      return
    }
    const media = window.matchMedia("(prefers-color-scheme: dark)")
    const handler = () => applyThemePreference(theme)
    media.addEventListener?.("change", handler)
    media.addListener?.(handler)
    return () => {
      media.removeEventListener?.("change", handler)
      media.removeListener?.(handler)
    }
  }, [theme])

  const refresh = useCallback(async (epicIdOverride?: number | null) => {
    setLoading(true)
    setError(null)
    try {
      const [statusResponse, epicResponses] = await Promise.all([
        fetchStatus(),
        fetchEpics(),
      ])
      setStatus(statusResponse)
      setEpics(epicResponses)

      let epicId: number | null = epicIdOverride ?? epicResponses[0]?.id ?? null
      if (epicId !== null && !epicResponses.some((e) => e.id === epicId)) {
        epicId = epicResponses[0]?.id ?? null
      }
      setSelectedEpicId(epicId)
      setGraph(epicId === null ? null : await fetchEpicGraph(epicId))
      setSelectedNodeId(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh(null)
  }, [refresh])

  const tasksById = useMemo(() => {
    const map = new Map<number, EpicGraph["tasks"][number]>()
    for (const task of graph?.tasks ?? []) {
      map.set(task.id, task)
    }
    return map
  }, [graph])

  const agentsById = useMemo(() => {
    const map = new Map<number, EpicGraph["agents"][number]>()
    for (const agent of graph?.agents ?? []) {
      map.set(agent.id, agent)
    }
    return map
  }, [graph])

  const childrenByParent = useMemo(() => {
    const map = new Map<number | null, GraphNode[]>()
    for (const node of graph?.nodes ?? []) {
      const key = node.parentNodeId ?? null
      map.set(key, [...(map.get(key) ?? []), node])
    }
    for (const [key, value] of map.entries()) {
      value.sort((a, b) => a.id - b.id)
      map.set(key, value)
    }
    return map
  }, [graph])

  const rootNodes = childrenByParent.get(null) ?? []

  const nodesById = useMemo(() => {
    const map = new Map<number, GraphNode>()
    for (const node of graph?.nodes ?? []) {
      map.set(node.id, node)
    }
    return map
  }, [graph])

  const selectedEpic = useMemo(
    () =>
      selectedEpicId === null
        ? null
        : (epics.find((e) => e.id === selectedEpicId) ?? null),
    [epics, selectedEpicId],
  )

  const selectedNode = useMemo(
    () =>
      selectedNodeId === null ? null : (nodesById.get(selectedNodeId) ?? null),
    [nodesById, selectedNodeId],
  )

  const selectedTask = useMemo(() => {
    if (!selectedNode || selectedNode.primaryTaskId === null) {
      return null
    }
    return tasksById.get(selectedNode.primaryTaskId) ?? null
  }, [selectedNode, tasksById])

  const selectedAgent = useMemo(() => {
    if (!selectedNode || selectedNode.agentId === null) {
      return null
    }
    return agentsById.get(selectedNode.agentId) ?? null
  }, [selectedNode, agentsById])

  function renderNode(node: GraphNode, depth: number) {
    const task =
      node.primaryTaskId !== null
        ? tasksById.get(node.primaryTaskId)
        : undefined
    const agent =
      node.agentId !== null ? agentsById.get(node.agentId) : undefined

    return (
      <div key={node.id} style={{ paddingLeft: depth * 16 }}>
        <Card
          className={`cursor-pointer transition-colors hover:bg-accent/40 ${
            selectedNodeId === node.id ? "ring-2 ring-ring" : ""
          }`}
          onClick={() => setSelectedNodeId(node.id)}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault()
              setSelectedNodeId(node.id)
            }
          }}
        >
          <CardContent className="grid gap-1 p-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="font-mono text-sm">{node.branchName}</div>
              <div className="text-xs text-muted-foreground">
                node {node.id}
                {agent ? ` · agent: ${agent.displayName}` : ""}
              </div>
            </div>
            <div className="text-sm text-muted-foreground">
              {task?.title ?? "—"}
            </div>
          </CardContent>
        </Card>

        <div className="mt-2 grid gap-2">
          {(childrenByParent.get(node.id) ?? []).map((child) =>
            renderNode(child, depth + 1),
          )}
        </div>
      </div>
    )
  }

  const themeIcon =
    theme === "dark" ? <Moon /> : theme === "light" ? <Sun /> : <Monitor />

  const breadcrumbs = useMemo(() => {
    const items: string[] = []
    if (selectedEpic) {
      items.push(`Epic: ${selectedEpic.slug}`)
    }
    if (selectedNode) {
      items.push(`Branch: ${selectedNode.branchName}`)
    }
    if (selectedTask) {
      items.push(`Task: ${selectedTask.title}`)
    }
    return items
  }, [selectedEpic, selectedNode, selectedTask])

  return (
    <div className="h-screen w-screen bg-background text-foreground">
      <div className="grid h-full grid-cols-[minmax(14rem,18rem)_1fr_minmax(18rem,24rem)]">
        <aside className="flex min-w-0 flex-col border-r bg-sidebar text-sidebar-foreground">
          <div className="p-4">
            <div className="text-xs font-medium text-muted-foreground">
              Project
            </div>
            {epics.length ? (
              <div className="relative mt-2">
                <select
                  className="h-9 w-full appearance-none rounded-md border bg-background px-3 pr-10 text-sm text-foreground shadow-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  value={selectedEpicId ?? ""}
                  onChange={(e) => void refresh(Number(e.target.value))}
                  disabled={loading}
                >
                  {epics.map((epic) => (
                    <option key={epic.id} value={epic.id}>
                      {epic.slug}
                    </option>
                  ))}
                </select>
                <ChevronDown className="pointer-events-none absolute right-3 top-2.5 h-4 w-4 text-muted-foreground" />
              </div>
            ) : (
              <div className="mt-2 text-sm text-muted-foreground">
                No projects.
              </div>
            )}
          </div>

          <nav className="flex-1" />

          <div className="flex items-center justify-between border-t p-3">
            <Button
              variant="ghost"
              size="icon"
              title={`Theme: ${theme}`}
              aria-label={`Theme: ${theme}`}
              onClick={() =>
                setTheme(
                  theme === "dark"
                    ? "light"
                    : theme === "light"
                      ? "system"
                      : "dark",
                )
              }
            >
              {themeIcon}
            </Button>
          </div>
        </aside>

        <main className="flex min-w-0 flex-col">
          <header className="flex items-center justify-between gap-3 border-b bg-background/80 px-4 py-2 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <div className="min-w-0 text-sm">
              {breadcrumbs.length ? (
                <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1 text-muted-foreground">
                  {breadcrumbs.map((item, idx) => (
                    <div key={item} className="flex min-w-0 items-center gap-2">
                      <span className="truncate text-foreground">{item}</span>
                      {idx < breadcrumbs.length - 1 ? (
                        <span className="text-muted-foreground/60">/</span>
                      ) : null}
                    </div>
                  ))}
                </div>
              ) : (
                <span className="text-muted-foreground">No selection</span>
              )}
              {error ? (
                <div className="mt-1 text-xs text-destructive">{error}</div>
              ) : null}
            </div>
            <Button
              variant="outline"
              onClick={() => void refresh(selectedEpicId)}
              disabled={loading}
            >
              {loading ? "Refreshing…" : "Refresh"}
            </Button>
          </header>

          <div className="relative flex-1 overflow-hidden">
            <div className="pointer-events-none absolute inset-0 opacity-40 [background-image:radial-gradient(var(--border)_1px,transparent_1px)] [background-size:24px_24px]" />
            <div className="relative h-full overflow-auto p-6">
              {graph ? (
                <div className="grid gap-3">
                  {rootNodes.length ? (
                    rootNodes.map((node) => renderNode(node, 0))
                  ) : (
                    <div className="text-sm text-muted-foreground">
                      No nodes.
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-sm text-muted-foreground">
                  Select a project to view its graph.
                </div>
              )}
            </div>
          </div>
        </main>

        <aside className="flex min-w-0 flex-col border-l">
          <header className="border-b px-4 py-2 text-sm font-medium">
            Details
          </header>
          <div className="flex-1 overflow-auto p-4">
            {selectedNode ? (
              <div className="grid gap-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Branch</CardTitle>
                  </CardHeader>
                  <CardContent className="grid gap-3">
                    <dl className="grid grid-cols-1 gap-3 text-sm">
                      <div className="grid gap-1">
                        <dt className="text-xs text-muted-foreground">Name</dt>
                        <dd className="break-all font-mono">
                          {selectedNode.branchName}
                        </dd>
                      </div>
                      <div className="grid gap-1">
                        <dt className="text-xs text-muted-foreground">Node</dt>
                        <dd className="font-mono">{selectedNode.id}</dd>
                      </div>
                      {selectedAgent ? (
                        <div className="grid gap-1">
                          <dt className="text-xs text-muted-foreground">
                            Agent
                          </dt>
                          <dd className="font-mono">
                            {selectedAgent.displayName}
                          </dd>
                        </div>
                      ) : null}
                    </dl>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Task</CardTitle>
                  </CardHeader>
                  <CardContent className="grid gap-3">
                    {selectedTask ? (
                      <>
                        <div className="grid gap-1">
                          <div className="text-sm font-medium">
                            {selectedTask.title}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {selectedTask.state} · {selectedTask.source} ·{" "}
                            {selectedTask.authority}
                          </div>
                        </div>
                        {selectedTask.readme ? (
                          <div className="max-h-[36rem] overflow-auto rounded-md border bg-muted p-3">
                            <Markdown content={selectedTask.readme} />
                          </div>
                        ) : (
                          <div className="text-sm text-muted-foreground">
                            No README.
                          </div>
                        )}
                      </>
                    ) : (
                      <div className="text-sm text-muted-foreground">
                        No task attached.
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            ) : (
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Daemon Status</CardTitle>
                </CardHeader>
                <CardContent className="grid gap-4">
                  <dl className="grid grid-cols-1 gap-4">
                    <div className="grid gap-1">
                      <dt className="text-xs text-muted-foreground">Repo</dt>
                      <dd className="break-all font-mono text-sm">
                        {status?.repoRoot ?? "—"}
                      </dd>
                    </div>
                    <div className="grid gap-1">
                      <dt className="text-xs text-muted-foreground">DB</dt>
                      <dd className="break-all font-mono text-sm">
                        {status?.dbPath ?? "—"}
                      </dd>
                    </div>
                    <div className="grid gap-1">
                      <dt className="text-xs text-muted-foreground">
                        Default Branch
                      </dt>
                      <dd className="font-mono text-sm">
                        {status?.defaultBranch ?? "—"}
                      </dd>
                    </div>
                    <div className="grid gap-1">
                      <dt className="text-xs text-muted-foreground">Block</dt>
                      <dd className="font-mono text-sm">
                        {status?.block
                          ? `${status.block.mode} (${status.block.reason ?? "n/a"})`
                          : "none"}
                      </dd>
                    </div>
                  </dl>
                </CardContent>
              </Card>
            )}
          </div>
        </aside>
      </div>
    </div>
  )
}

export default App
