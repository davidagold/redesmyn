import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const resourceBadgeVariants = cva(
  "inline-flex items-center overflow-hidden whitespace-nowrap rounded-sm border text-xs font-mono",
  {
    variants: {
      variant: {
        default: "border-border/60 text-foreground/80",
        muted: "border-border/60 text-muted-foreground",
      },
      size: {
        sm: "",
        xs: "text-[0.625rem]",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "sm",
    },
  },
)

export type ResourceBadgeProps = Omit<React.HTMLAttributes<HTMLSpanElement>, "children"> & VariantProps<typeof resourceBadgeVariants> & {
  label: React.ReactNode
  value?: React.ReactNode
  extendBackground?: boolean
}

export function ResourceBadge({
  className,
  variant,
  size,
  label,
  value,
  extendBackground = false,
  ...props
}: ResourceBadgeProps) {
  const displayValue =
    typeof value === "string" && value === "Generic" ? "Shell" : value
  const hasValue =
    displayValue !== null && displayValue !== undefined && displayValue !== ""
  const segmentClasses = cn(
    "inline-flex items-center px-2 py-0.5 transition-colors",
    size === "xs" ? "text-[0.625rem]" : null,
  )

  return (
    <span
      className={cn(resourceBadgeVariants({ variant, size }), className)}
      {...props}
    >
      <span
        className={cn(
          segmentClasses,
          "bg-foreground/5 group-hover:bg-foreground/10 group-focus-within:bg-foreground/10",
        )}
      >
        {label}
      </span>
      {hasValue ? (
        <span
          className={cn(
            segmentClasses,
            "border-l border-border/40",
            extendBackground
              ? "bg-muted/40 group-hover:bg-muted/55 group-focus-within:bg-muted/55"
              : null,
          )}
        >
          {displayValue}
        </span>
      ) : null}
    </span>
  )
}
