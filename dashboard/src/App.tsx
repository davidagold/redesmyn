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

type GraphNode = EpicGraph["nodes"][number]

function App() {
  const [status, setStatus] = useState<ApiStatus | null>(null)
  const [epics, setEpics] = useState<Epic[]>([])
  const [selectedEpicId, setSelectedEpicId] = useState<number | null>(null)
  const [graph, setGraph] = useState<EpicGraph | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

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

      let epicId: number | null =
        epicIdOverride ?? epicResponses[0]?.id ?? null
      if (epicId !== null && !epicResponses.some((e) => e.id === epicId)) {
        epicId = epicResponses[0]?.id ?? null
      }
      setSelectedEpicId(epicId)
      setGraph(epicId === null ? null : await fetchEpicGraph(epicId))
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

  function renderNode(node: GraphNode, depth: number) {
    const task =
      node.primaryTaskId !== null ? tasksById.get(node.primaryTaskId) : undefined
    const agent = node.agentId !== null ? agentsById.get(node.agentId) : undefined

    return (
      <div key={node.id} style={{ paddingLeft: depth * 16 }}>
        <Card>
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

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto flex max-w-5xl flex-col gap-6 p-6">
        <header className="flex items-center justify-between gap-3">
          <h1 className="text-2xl font-semibold tracking-tight">Redesmyn</h1>
          <Button onClick={() => void refresh(selectedEpicId)} disabled={loading}>
            {loading ? "Refreshing…" : "Refresh"}
          </Button>
        </header>

        {error ? <div className="text-sm text-destructive">{error}</div> : null}

        <Card>
          <CardHeader>
            <CardTitle>Daemon Status</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-4">
            <dl className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="grid gap-1">
                <dt className="text-sm text-muted-foreground">Repo</dt>
                <dd className="break-all font-mono text-sm">
                  {status?.repoRoot ?? "—"}
                </dd>
              </div>
              <div className="grid gap-1">
                <dt className="text-sm text-muted-foreground">DB</dt>
                <dd className="break-all font-mono text-sm">
                  {status?.dbPath ?? "—"}
                </dd>
              </div>
              <div className="grid gap-1">
                <dt className="text-sm text-muted-foreground">
                  Default Branch
                </dt>
                <dd className="font-mono text-sm">
                  {status?.defaultBranch ?? "—"}
                </dd>
              </div>
              <div className="grid gap-1">
                <dt className="text-sm text-muted-foreground">Block</dt>
                <dd className="font-mono text-sm">
                  {status?.block
                    ? `${status.block.mode} (${status.block.reason ?? "n/a"})`
                    : "none"}
                </dd>
              </div>
            </dl>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Epic Graph</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-4">
            {epics.length ? (
              <div className="flex flex-wrap items-center gap-3">
                <div className="text-sm text-muted-foreground">Epic</div>
                <select
                  className="h-9 rounded-md border bg-background px-3 text-sm"
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
              </div>
            ) : (
              <div className="text-sm text-muted-foreground">No epics.</div>
            )}

            {graph ? (
              <div className="grid gap-3">
                {rootNodes.length ? (
                  rootNodes.map((node) => renderNode(node, 0))
                ) : (
                  <div className="text-sm text-muted-foreground">No nodes.</div>
                )}
              </div>
            ) : null}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export default App
