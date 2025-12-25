import { useCallback, useEffect, useState } from "react"
import { fetchStatus, type ApiStatus } from "@/api"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

function App() {
  const [status, setStatus] = useState<ApiStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setStatus(await fetchStatus())
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto flex max-w-3xl flex-col gap-6 p-6">
        <header className="flex items-center justify-between gap-3">
          <h1 className="text-2xl font-semibold tracking-tight">Redesmyn</h1>
          <Button onClick={() => void refresh()} disabled={loading}>
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
                <dt className="text-sm text-muted-foreground">Pause</dt>
                <dd className="font-mono text-sm">
                  {status?.pause
                    ? `${status.pause.mode} (${status.pause.reason ?? "n/a"})`
                    : "none"}
                </dd>
              </div>
            </dl>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export default App
