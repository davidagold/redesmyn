import { forwardRef, type ComponentPropsWithoutRef } from "react"
import { cn } from "@/lib/utils"

type TextareaProps = ComponentPropsWithoutRef<"textarea">

const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(function Textarea(
  { className, ...props },
  ref,
) {
  return (
    <textarea
      ref={ref}
      data-slot="textarea"
      className={cn(
        "resize-y rounded-md border bg-background/40 px-2 py-2 text-xs text-foreground shadow-sm outline-none focus-visible:ring-2 focus-visible:ring-ring/30 disabled:cursor-not-allowed disabled:opacity-50",
        className,
      )}
      {...props}
    />
  )
})

Textarea.displayName = "Textarea"

export { Textarea }
