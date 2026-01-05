import { Button } from "@/components/ui/button"
import type { Epic } from "@/api"
import { ChevronDown } from "lucide-react"

interface EpicSelectorProps {
  epics: Epic[]
  selectedEpic: Epic | null
  selectedEpicId: number | null
  loading: boolean
  menuOpen: boolean
  onMenuToggle: () => void
  onMenuClose: () => void
  onSelectEpic: (epicId: number) => void
}

export function EpicSelector({
  epics,
  selectedEpic,
  selectedEpicId,
  loading,
  menuOpen,
  onMenuToggle,
  onMenuClose,
  onSelectEpic,
}: EpicSelectorProps) {
  return (
    <div className="relative">
      <Button
        variant="ghost"
        className="w-fit gap-1"
        onClick={onMenuToggle}
        aria-expanded={menuOpen}
        disabledReason={!epics.length ? "No epics available" : null}
      >
        <span className="truncate">
          {selectedEpic?.slug ?? (epics.length ? "Select epic" : "No epics")}
        </span>
        <ChevronDown className="h-3 w-3 text-muted-foreground" />
      </Button>

      {menuOpen ? (
        <>
          <button
            type="button"
            aria-label="Close epic menu"
            className="fixed inset-0 z-10 cursor-default bg-transparent"
            onClick={onMenuClose}
          />
          <div className="absolute left-0 top-full z-20 mt-2 w-64 rounded-lg border bg-popover p-2 shadow">
            <div className="grid gap-1">
              {epics.map((epic) => (
                <Button
                  key={epic.id}
                  variant={epic.id === selectedEpicId ? "secondary" : "ghost"}
                  className="w-fit justify-start"
                  onClick={() => onSelectEpic(epic.id)}
                  disabledReason={loading ? "Refreshing…" : null}
                >
                  {epic.slug}
                </Button>
              ))}
            </div>
          </div>
        </>
      ) : null}
    </div>
  )
}
