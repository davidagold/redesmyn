import { Button } from "@/components/ui/button"
import type { Epic } from "@/api"
import { ChevronRight } from "lucide-react"

interface HeaderProps {
  selectedEpic: Epic | null
  selectedLabel: string | null
  loading: boolean
  error: string | null
  onClearSelection: () => void
  onRefresh: () => void
}

export function Header({
  selectedEpic,
  selectedLabel,
  loading,
  error,
  onClearSelection,
  onRefresh,
}: HeaderProps) {
  return (
    <>
      <header className="flex h-10 items-center justify-between gap-3 border-b px-4 py-0">
        <div className="flex min-w-0 items-center gap-2 text-sm">
          {selectedEpic ? (
            <Button
              variant="ghost"
              className="w-fit"
              onClick={onClearSelection}
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
