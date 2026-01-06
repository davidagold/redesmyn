import { cva, type VariantProps } from "class-variance-authority"
import { forwardRef, type HTMLAttributes } from "react"

import { cn } from "@/lib/utils"

const alertVariants = cva(
  "grid gap-2 rounded-md border px-3 py-2 shadow-sm ring-1 backdrop-blur",
  {
    variants: {
      variant: {
        default: "border-border/60 bg-background/30 ring-border/10",
        destructive:
          "border-destructive/25 bg-destructive/10 ring-destructive/10",
        amber: "border-amber-400/25 bg-amber-400/10 ring-amber-400/10",
        emerald: "border-emerald-400/25 bg-emerald-400/10 ring-emerald-400/10",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
)

type AlertProps = HTMLAttributes<HTMLDivElement> & VariantProps<typeof alertVariants>

const Alert = forwardRef<HTMLDivElement, AlertProps>(function Alert(
  { className, variant = "default", ...props },
  ref,
) {
  return (
    <div
      ref={ref}
      role="alert"
      className={cn(alertVariants({ variant }), className)}
      {...props}
    />
  )
})

Alert.displayName = "Alert"

const AlertTitle =
  forwardRef<HTMLHeadingElement, HTMLAttributes<HTMLHeadingElement>>(
    function AlertTitle({ className, ...props }, ref) {
      return (
        <h5
          ref={ref}
          className={cn("text-xs font-medium text-foreground", className)}
          {...props}
        />
      )
    },
  )

AlertTitle.displayName = "AlertTitle"

const AlertDescription =
  forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
    function AlertDescription({ className, ...props }, ref) {
      return (
        <div
          ref={ref}
          className={cn("text-xs text-foreground/80", className)}
          {...props}
        />
      )
    },
  )

AlertDescription.displayName = "AlertDescription"

export { Alert, AlertTitle, AlertDescription, alertVariants }
