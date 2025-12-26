import { Button } from "@/components/ui/button"
import { EpicSelector } from "./EpicSelector"
import type { Epic } from "@/api"
import { ChevronRight } from "lucide-react"

interface HeaderProps {
  epics: Epic[]
  selectedEpic: Epic | null
  selectedEpicId: number | null
  selectedLabel: string | null
  loading: boolean
  error: string | null
  epicMenuOpen: boolean
  onEpicMenuToggle: () => void
  onEpicMenuClose: () => void
  onSelectEpic: (epicId: number) => void
  onClearSelection: () => void
  onRefresh: () => void
}

export function Header({
  epics,
  selectedEpic,
  selectedEpicId,
  selectedLabel,
  loading,
  error,
  epicMenuOpen,
  onEpicMenuToggle,
  onEpicMenuClose,
  onSelectEpic,
  onClearSelection,
  onRefresh,
}: HeaderProps) {
  return (
    <>
      <header className="flex h-10 items-center justify-between gap-3 border-b px-4 py-0">
        <div className="flex min-w-0 items-center gap-2 text-sm">
          <EpicSelector
            epics={epics}
            selectedEpic={selectedEpic}
            selectedEpicId={selectedEpicId}
            loading={loading}
            menuOpen={epicMenuOpen}
            onMenuToggle={onEpicMenuToggle}
            onMenuClose={onEpicMenuClose}
            onSelectEpic={onSelectEpic}
          />

          {selectedLabel ? (
            <>
              <ChevronRight
                className="h-4 w-4 shrink-0 text-muted-foreground/60"
                aria-hidden="true"
              />
              <Button
                variant="ghost"
                className="w-fit"
                onClick={onClearSelection}
              >
                <span className="truncate">{selectedLabel}</span>
              </Button>
            </>
          ) : null}
        </div>

        <Button variant="outline" onClick={onRefresh} disabled={loading}>
          {loading ? "Refreshing..." : "Refresh"}
        </Button>
      </header>

      {error ? (
        <div className="border-b px-4 py-2 text-sm text-destructive">
          {error}
        </div>
      ) : null}
    </>
  )
}
