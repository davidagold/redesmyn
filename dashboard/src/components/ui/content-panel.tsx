import { cn } from "@/lib/utils"

interface ContentPanelProps {
  children: React.ReactNode
  className?: string
}

export function ContentPanel({ children, className }: ContentPanelProps) {
  return (
    <div
      className={cn(
        "flex min-w-0 flex-1 flex-col overflow-hidden rounded-sm rounded-br-xl border border-border/60 bg-background shadow-sm",
        className,
      )}
    >
      {children}
    </div>
  )
}

interface ContentPanelHeaderProps {
  children: React.ReactNode
  className?: string
}

export function ContentPanelHeader({
  children,
  className,
}: ContentPanelHeaderProps) {
  return (
    <header
      className={cn(
        "flex h-10 items-center justify-between gap-3 border-b px-4 py-0",
        className,
      )}
    >
      {children}
    </header>
  )
}
