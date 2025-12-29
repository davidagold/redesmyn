import { Switch as SwitchPrimitive } from "@base-ui/react/switch"

import { cn } from "@/lib/utils"
import { Tooltip, TooltipContent, TooltipTrigger } from "./tooltip"

function Switch({
  className,
  disabledReason = null,
  ...props
}: Omit<SwitchPrimitive.Root.Props, "disabled"> & {
  disabledReason?: string | null
}) {
  const disabled = Boolean(disabledReason)
  const switchControl = (
    <SwitchPrimitive.Root
      disabled={disabled}
      className={cn(
        "inline-flex h-5 w-9 items-center rounded-full border border-border/60 bg-background/30 p-0.5 shadow-sm outline-none transition-colors focus-visible:ring-2 focus-visible:ring-ring/30 data-[checked]:border-primary/40 data-[checked]:bg-primary data-[disabled]:cursor-not-allowed data-[disabled]:opacity-50",
        className,
      )}
      {...props}
    >
      <SwitchPrimitive.Thumb className="size-4 rounded-full bg-foreground/80 shadow-sm transition-transform data-[checked]:translate-x-4 data-[unchecked]:translate-x-0 data-[disabled]:bg-foreground/40" />
    </SwitchPrimitive.Root>
  )

  if (!disabledReason) {
    return switchControl
  }

  return (
    <Tooltip>
      <TooltipTrigger
        render={(triggerProps) => (
          <span
            {...triggerProps}
            className={cn("inline-flex", triggerProps.className)}
            tabIndex={triggerProps.tabIndex ?? 0}
            onClick={(event) => {
              triggerProps.onClick?.(event)
              event.preventDefault()
              event.stopPropagation()
            }}
            onKeyDown={(event) => {
              triggerProps.onKeyDown?.(event)
              event.stopPropagation()
            }}
          >
            {switchControl}
          </span>
        )}
      />
      <TooltipContent side="bottom" align="center">
        {disabledReason}
      </TooltipContent>
    </Tooltip>
  )
}

export { Switch }
