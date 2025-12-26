import { useCallback, useEffect, useMemo, useState } from "react"
import { fetchEpicGraph, fetchEpics, type Epic, type EpicGraph } from "@/api"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Markdown } from "@/components/markdown"
import {
  applyThemePreference,
  getStoredThemePreference,
  setStoredThemePreference,
  type ThemePreference,
} from "@/lib/theme"
import { ChevronDown, ChevronRight, Monitor, Moon, Sun, X } from "lucide-react"

type GraphNode = EpicGraph["nodes"][number]

function formatBranchName(branchName: string, epicSlug?: string | null) {
  if (!epicSlug) {
    return branchName
  }
  const prefix = `rn/${epicSlug}/`
  return branchName.startsWith(prefix) ? branchName.slice(prefix.length) : branchName
}

function App() {
  const [theme, setTheme] = useState<ThemePreference>(
    () => getStoredThemePreference() ?? "dark",
  )
  const [epics, setEpics] = useState<Epic[]>([])
  const [selectedEpicId, setSelectedEpicId] = useState<number | null>(null)
  const [graph, setGraph] = useState<EpicGraph | null>(null)
  const [selectedNodeId, setSelectedNodeId] = useState<number | null>(null)
  const [projectMenuOpen, setProjectMenuOpen] = useState(false)
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
      const epicResponses = await fetchEpics()
      setEpics(epicResponses)

      let epicId: number | null = epicIdOverride ?? epicResponses[0]?.id ?? null
      if (epicId !== null && !epicResponses.some((e) => e.id === epicId)) {
        epicId = epicResponses[0]?.id ?? null
      }
      setSelectedEpicId(epicId)
      setGraph(epicId === null ? null : await fetchEpicGraph(epicId))
      setSelectedNodeId(null)
      setProjectMenuOpen(false)
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

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key !== "Escape") {
        return
      }
      if (projectMenuOpen) {
        setProjectMenuOpen(false)
        return
      }
      if (selectedNodeId !== null) {
        setSelectedNodeId(null)
      }
    }

    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [projectMenuOpen, selectedNodeId])

  function renderNode(node: GraphNode, depth: number) {
    const task =
      node.primaryTaskId !== null
        ? tasksById.get(node.primaryTaskId)
        : undefined
    const agent =
      node.agentId !== null ? agentsById.get(node.agentId) : undefined
    const branchLabel = formatBranchName(node.branchName, selectedEpic?.slug)

    return (
      <div key={node.id} style={{ paddingLeft: depth * 16 }}>
        <Card
          data-node-card
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
              <div className="font-mono text-sm" title={node.branchName}>
                {branchLabel}
              </div>
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

  const selectedLabel =
    selectedTask?.title ??
    (selectedNode
      ? formatBranchName(selectedNode.branchName, selectedEpic?.slug)
      : null)

  return (
    <div className="h-screen w-screen bg-background text-foreground">
      <div className="flex h-full gap-2 p-2">
        <aside className="flex w-56 shrink-0 flex-col gap-4">
          <div className="relative flex h-9 items-center">
            <Button
              variant="ghost"
              className="w-fit gap-2"
              onClick={() => setProjectMenuOpen((open) => !open)}
              aria-expanded={projectMenuOpen}
              disabled={!epics.length}
            >
              <span className="truncate">
                {selectedEpic?.slug ??
                  (epics.length ? "Project" : "No projects")}
              </span>
              <ChevronDown className="h-4 w-4 text-muted-foreground" />
            </Button>

            {projectMenuOpen ? (
              <>
                <button
                  type="button"
                  aria-label="Close project menu"
                  className="fixed inset-0 z-10 cursor-default bg-transparent"
                  onClick={() => setProjectMenuOpen(false)}
                />
                <div className="absolute left-0 top-full z-20 mt-2 w-64 rounded-lg border bg-popover p-2 shadow">
                  <div className="grid gap-1">
                    {epics.map((epic) => (
                      <Button
                        key={epic.id}
                        variant={
                          epic.id === selectedEpicId ? "secondary" : "ghost"
                        }
                        className="w-fit justify-start"
                        onClick={() => void refresh(epic.id)}
                        disabled={loading}
                      >
                        {epic.slug}
                      </Button>
                    ))}
                  </div>
                </div>
              </>
            ) : null}
          </div>

          <nav className="flex-1" />

          <div className="flex items-center gap-2">
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

        <div className="flex min-w-0 flex-1 flex-col overflow-hidden rounded-lg border border-border/60 bg-card shadow-sm">
          <header className="flex h-9 items-center justify-between gap-3 border-b bg-card/80 px-4 py-0 backdrop-blur supports-[backdrop-filter]:bg-card/60">
            <div className="flex min-w-0 items-center gap-2 text-sm">
              {selectedEpic ? (
                <Button
                  variant="ghost"
                  className="w-fit px-2"
                  onClick={() => setSelectedNodeId(null)}
                >
                  {selectedEpic.slug}
                </Button>
              ) : (
                <span className="text-muted-foreground">Select project</span>
              )}
              {selectedLabel ? (
                <>
                  <ChevronRight
                    className="h-4 w-4 shrink-0 text-muted-foreground/60"
                    aria-hidden="true"
                  />
                  <span className="truncate">{selectedLabel}</span>
                </>
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

          {error ? (
            <div className="border-b px-4 py-2 text-sm text-destructive">
              {error}
            </div>
          ) : null}

          <div className="relative flex min-h-0 flex-1">
            <main className="relative min-w-0 flex-1 overflow-hidden">
              <div className="pointer-events-none absolute inset-0 opacity-[0.06] [background-image:radial-gradient(var(--border)_1px,transparent_1px)] [background-size:32px_32px]" />
              <div
                className="relative h-full overflow-auto p-6"
                onClick={(e) => {
                  if (!(e.target instanceof Node)) {
                    return
                  }
                  const el =
                    e.target instanceof Element
                      ? e.target
                      : e.target.parentElement
                  if (el?.closest("[data-node-card]")) {
                    return
                  }
                  setSelectedNodeId(null)
                }}
              >
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
            </main>

            <aside
              className={`absolute inset-y-0 right-0 z-20 w-[32rem] border-l bg-card shadow-lg transition-transform duration-200 ease-out ${
                selectedNode ? "translate-x-0" : "translate-x-full"
              } ${selectedNode ? "" : "pointer-events-none"}`}
            >
              <Button
                variant="ghost"
                size="icon"
                aria-label="Close details"
                className="absolute right-3 top-3"
                onClick={() => setSelectedNodeId(null)}
              >
                <X className="h-4 w-4" />
              </Button>

              <div
                className={`h-full overflow-auto p-4 pt-12 transition-opacity duration-150 ease-out ${
                  selectedNode ? "opacity-100" : "opacity-0"
                }`}
              >
                {selectedTask?.readme ? (
                  <Markdown
                    content={selectedTask.readme}
                    omitFirstHeading
                    omitMetadataSection
                  />
                ) : (
                  <div className="text-sm text-muted-foreground">
                    No README.
                  </div>
                )}
              </div>
            </aside>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
