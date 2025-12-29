import { cn } from "@/lib/utils"

interface SlidePanelProps {
  open: boolean
  children: React.ReactNode
  className?: string
  side?: "left" | "right"
}

export function SlidePanel({
  open,
  children,
  className,
  side = "right",
}: SlidePanelProps) {
  const isLeft = side === "left"
  return (
    <aside
      className={cn(
        "absolute inset-y-0 z-20 w-[32rem] bg-background shadow-lg transition-transform duration-200 ease-out",
        isLeft ? "left-0 border-r" : "right-0 border-l",
        open
          ? "translate-x-0"
          : isLeft
            ? "pointer-events-none -translate-x-full"
            : "pointer-events-none translate-x-full",
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
