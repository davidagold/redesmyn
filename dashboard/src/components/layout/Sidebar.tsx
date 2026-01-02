import { Link, useLocation } from "@tanstack/react-router"
import { Button, buttonVariants } from "@/components/ui/button"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { WorkspaceButton } from "./WorkspaceButton"
import type { ThemePreference } from "@/lib/theme"
import {
  Bot,
  GitBranch,
  GitGraph,
  History,
  Monitor,
  Moon,
  Sun,
} from "lucide-react"
import { cn } from "@/lib/utils"

interface SidebarProps {
  theme: ThemePreference
  onCycleTheme: () => void
}

const navItems = [
  { label: "Epics", icon: GitGraph, path: "/graph" },
  { label: "Agents", icon: Bot, path: null },
  { label: "Worktrees", icon: GitBranch, path: null },
  { label: "Timeline", icon: History, path: null },
] as const

export function Sidebar({ theme, onCycleTheme }: SidebarProps) {
  const location = useLocation()
  const themeIcon =
    theme === "dark" ? <Moon /> : theme === "light" ? <Sun /> : <Monitor />

  return (
    <aside className="flex w-56 shrink-0 flex-col gap-4">
      <div className="relative flex h-10 items-center">
        <WorkspaceButton />
      </div>

      <nav className="flex flex-1 flex-col gap-0.5">
        {navItems.map(({ label, icon: Icon, path }) => {
          const isActive = path ? location.pathname.startsWith(path) : false

          if (path) {
            return (
              <Link
                key={label}
                to={path}
                className={cn(
                  buttonVariants({ variant: "ghost" }),
                  "h-8 justify-start gap-2 px-2 text-sm font-normal",
                  isActive
                    ? "text-foreground"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                <Icon className="h-4 w-4" />
                {label}
              </Link>
            )
          }

          return (
            <Button
              key={label}
              variant="ghost"
              className="h-8 justify-start gap-2 px-2 text-sm font-normal text-muted-foreground hover:text-foreground"
              disabledReason="Coming soon"
            >
              <Icon className="h-4 w-4" />
              {label}
            </Button>
          )
        })}
      </nav>

      <div className="flex items-center gap-2">
        <Tooltip>
          <TooltipTrigger
            render={(triggerProps) => (
              <Button
                {...triggerProps}
                variant="ghost"
                size="icon"
                aria-label={`Theme: ${theme}`}
                onClick={onCycleTheme}
              >
                {themeIcon}
              </Button>
            )}
          />
          <TooltipContent side="bottom" sideOffset={10}>
            Theme: {theme}
          </TooltipContent>
        </Tooltip>
      </div>
    </aside>
  )
}
