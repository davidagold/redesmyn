import { useNavigate } from "@tanstack/react-router"
import { useEffect } from "react"
import { ContentPanel, ContentPanelHeader } from "@/components/ui/content-panel"
import { EpicSelector } from "@/components/layout/EpicSelector"
import { Button } from "@/components/ui/button"
import { useEpics } from "@/hooks/useEpics"

export function IndexView() {
  const navigate = useNavigate()
  const { epics, loading, error, refresh } = useEpics()

  useEffect(() => {
    if (epics.length > 0) {
      void navigate({ to: "/$epicSlug", params: { epicSlug: epics[0].slug } })
    }
  }, [epics, navigate])

  return (
    <ContentPanel>
      <ContentPanelHeader>
        <div className="flex min-w-0 items-center gap-2 text-sm">
          <EpicSelector
            epics={epics}
            selectedEpic={null}
            selectedEpicId={null}
            loading={loading}
            menuOpen={false}
            onMenuToggle={() => {}}
            onMenuClose={() => {}}
            onSelectEpic={(epicId) => {
              const epic = epics.find((e) => e.id === epicId)
              if (epic) {
                void navigate({
                  to: "/$epicSlug",
                  params: { epicSlug: epic.slug },
                })
              }
            }}
          />
        </div>

        <Button
          variant="outline"
          onClick={() => void refresh()}
          disabled={loading}
        >
          {loading ? "Refreshing..." : "Refresh"}
        </Button>
      </ContentPanelHeader>

      {error ? (
        <div className="border-b px-4 py-2 text-sm text-destructive">
          {error}
        </div>
      ) : null}

      <div className="flex-1 p-6 text-sm text-muted-foreground">
        {loading ? "Loading epics..." : "Select an epic to view its graph."}
      </div>
    </ContentPanel>
  )
}
