import { Button } from "@/components/ui/button"
import { WorkspaceButton } from "./WorkspaceButton"
import type { ThemePreference } from "@/lib/theme"
import { Monitor, Moon, Sun } from "lucide-react"

interface SidebarProps {
  theme: ThemePreference
  onCycleTheme: () => void
}

export function Sidebar({ theme, onCycleTheme }: SidebarProps) {
  const themeIcon =
    theme === "dark" ? <Moon /> : theme === "light" ? <Sun /> : <Monitor />

  return (
    <aside className="flex w-56 shrink-0 flex-col gap-4">
      <div className="relative flex h-10 items-center">
        <WorkspaceButton />
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
