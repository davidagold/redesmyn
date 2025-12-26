import { cn } from "@/lib/utils"

interface SlidePanelProps {
  open: boolean
  children: React.ReactNode
  className?: string
}

export function SlidePanel({ open, children, className }: SlidePanelProps) {
  return (
    <aside
      className={cn(
        "absolute inset-y-0 right-0 z-20 w-[32rem] border-l bg-background shadow-lg transition-transform duration-200 ease-out",
        open ? "translate-x-0" : "pointer-events-none translate-x-full",
        className,
      )}
    >
      <div
        className={cn(
          "h-full overflow-auto transition-opacity duration-150 ease-out",
          open ? "opacity-100" : "opacity-0",
        )}
      >
        {children}
      </div>
    </aside>
  )
}
