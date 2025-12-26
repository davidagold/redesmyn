import { Outlet } from "@tanstack/react-router"
import { Sidebar } from "./Sidebar"
import { useTheme } from "@/hooks/useTheme"

export function RootLayout() {
  const { theme, cycleTheme } = useTheme()

  return (
    <div className="h-screen w-screen bg-surface text-foreground">
      <div className="flex h-full gap-2 p-2">
        <Sidebar theme={theme} onCycleTheme={cycleTheme} />
        <Outlet />
      </div>
    </div>
  )
}
