import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex items-center gap-1 whitespace-nowrap rounded-full border px-2 py-0.5 text-xs font-medium",
  {
    variants: {
      variant: {
        default: "border-border/60 bg-background/30 text-muted-foreground",
        muted: "border-border/60 bg-background/10 text-muted-foreground",
        amber: "border-amber-400/25 bg-amber-400/10 text-amber-200/90",
        emerald: "border-emerald-400/25 bg-emerald-400/10 text-emerald-100",
        destructive: "border-destructive/25 bg-destructive/10 text-destructive",
      },
      size: {
        sm: "",
        xs: "px-1.5 text-[0.625rem]",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "sm",
    },
  },
)

export type BadgeProps = React.HTMLAttributes<HTMLSpanElement> & VariantProps<typeof badgeVariants>

export function Badge({ className, variant, size, ...props }: BadgeProps) {
  return (
    <span
      className={cn(badgeVariants({ variant, size }), className)}
      {...props}
    />
  )
}
