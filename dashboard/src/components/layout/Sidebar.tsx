import { Button } from "@/components/ui/button"
import type { ThemePreference } from "@/lib/theme"
import type { Epic } from "@/api"
import { ChevronDown, Monitor, Moon, Sun } from "lucide-react"

interface SidebarProps {
  epics: Epic[]
  selectedEpic: Epic | null
  selectedEpicId: number | null
  loading: boolean
  projectMenuOpen: boolean
  onProjectMenuToggle: () => void
  onProjectMenuClose: () => void
  onSelectEpic: (epicId: number) => void
  theme: ThemePreference
  onCycleTheme: () => void
}

export function Sidebar({
  epics,
  selectedEpic,
  selectedEpicId,
  loading,
  projectMenuOpen,
  onProjectMenuToggle,
  onProjectMenuClose,
  onSelectEpic,
  theme,
  onCycleTheme,
}: SidebarProps) {
  const themeIcon =
    theme === "dark" ? <Moon /> : theme === "light" ? <Sun /> : <Monitor />

  return (
    <aside className="flex w-56 shrink-0 flex-col gap-4">
      <div className="relative flex h-10 items-center">
        <Button
          variant="ghost"
          className="w-fit gap-2"
          onClick={onProjectMenuToggle}
          aria-expanded={projectMenuOpen}
          disabled={!epics.length}
        >
          <span className="truncate">
            {selectedEpic?.slug ?? (epics.length ? "Project" : "No projects")}
          </span>
          <ChevronDown className="h-4 w-4 text-muted-foreground" />
        </Button>

        {projectMenuOpen ? (
          <>
            <button
              type="button"
              aria-label="Close project menu"
              className="fixed inset-0 z-10 cursor-default bg-transparent"
              onClick={onProjectMenuClose}
            />
            <div className="absolute left-0 top-full z-20 mt-2 w-64 rounded-lg border bg-popover p-2 shadow">
              <div className="grid gap-1">
                {epics.map((epic) => (
                  <Button
                    key={epic.id}
                    variant={epic.id === selectedEpicId ? "secondary" : "ghost"}
                    className="w-fit justify-start"
                    onClick={() => onSelectEpic(epic.id)}
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
          onClick={onCycleTheme}
        >
          {themeIcon}
        </Button>
      </div>
    </aside>
  )
}
