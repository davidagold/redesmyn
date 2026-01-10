import type { CSSProperties, ElementType } from "react"
import { cn } from "@/lib/utils"

export type ShimmerProps = {
  children: string
  as?: ElementType
  className?: string
  duration?: number
  spread?: number
}

export function Shimmer({
  children,
  as: Component = "span",
  className,
  duration = 1.75,
  spread = 28,
}: ShimmerProps) {
  return (
    <Component
      className={cn(
        "relative inline-block bg-[length:250%_100%,auto] bg-clip-text text-transparent",
        "[--bg:linear-gradient(90deg,#0000_calc(50%_-_var(--spread)),var(--color-foreground),#0000_calc(50%_+_var(--spread)))] [background-repeat:no-repeat,padding-box]",
        "[animation:rn-shimmer_var(--duration)_linear_infinite]",
        className,
      )}
      style={
        {
          "--spread": `${spread}px`,
          "--duration": `${duration}s`,
          backgroundImage:
            "var(--bg), linear-gradient(var(--color-muted-foreground), var(--color-muted-foreground))",
        } as CSSProperties
      }
    >
      {children}
    </Component>
  )
}
