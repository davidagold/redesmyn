import type { ReactNode } from "react"
import { cn } from "@/lib/utils"

export function FloatingActions(props: {
  children: ReactNode
  className?: string
  containerClassName?: string
  topClassName?: string
}) {
  const { children, className, containerClassName, topClassName } = props

  return (
    <div
      className={cn(
        "sticky z-20 h-0 pointer-events-none",
        topClassName ?? "top-4",
        containerClassName,
      )}
    >
      <div className="flex justify-end">
        <div
          className={cn(
            "inline-flex h-6 w-fit overflow-hidden rounded-md border border-border/60 bg-background/40 shadow-sm backdrop-blur pointer-events-auto",
            className,
          )}
        >
          {children}
        </div>
      </div>
    </div>
  )
}
