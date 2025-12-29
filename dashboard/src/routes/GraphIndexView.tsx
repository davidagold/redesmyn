import { useNavigate } from "@tanstack/react-router"
import { useState } from "react"
import { ContentPanel, ContentPanelHeader } from "@/components/ui/content-panel"
import { EpicSelector } from "@/components/layout/EpicSelector"
import { Button } from "@/components/ui/button"
import { useEpics } from "@/hooks/useEpics"

export function GraphIndexView() {
  const navigate = useNavigate()
  const { epics, loading, error, refresh } = useEpics()
  const [menuOpen, setMenuOpen] = useState(false)

  function handleSelectEpic(epicId: number) {
    const epic = epics.find((e) => e.id === epicId)
    if (epic) {
      void navigate({
        to: "/graph/$epicSlug",
        params: { epicSlug: epic.slug },
      })
    }
    setMenuOpen(false)
  }

  return (
    <ContentPanel>
      <ContentPanelHeader>
        <div className="flex min-w-0 items-center gap-2 text-sm">
          <EpicSelector
            epics={epics}
            selectedEpic={null}
            selectedEpicId={null}
            loading={loading}
            menuOpen={menuOpen}
            onMenuToggle={() => setMenuOpen((open) => !open)}
            onMenuClose={() => setMenuOpen(false)}
            onSelectEpic={handleSelectEpic}
          />
        </div>

        <Button
          variant="outline"
          onClick={() => void refresh()}
          disabledReason={loading ? "Refreshing…" : null}
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
