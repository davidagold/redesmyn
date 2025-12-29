import { buttonVariants } from "@/components/ui/button"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

interface CopyRunCommandButtonProps {
  disabledReason: string | null
  onCopy: () => void
}

export function CopyRunCommandButton({
  disabledReason,
  onCopy,
}: CopyRunCommandButtonProps) {
  return (
    <Tooltip>
      <TooltipTrigger
        type="button"
        className={cn(
          buttonVariants({ variant: "outline", size: "sm" }),
          disabledReason
            ? "cursor-not-allowed opacity-50 hover:bg-transparent hover:text-muted-foreground"
            : null,
        )}
        aria-disabled={disabledReason ? "true" : undefined}
        onClick={(event) => {
          if (disabledReason) {
            event.preventDefault()
            return
          }
          onCopy()
        }}
      >
        Copy rn run …
      </TooltipTrigger>
      {disabledReason ? (
        <TooltipContent side="bottom" align="end">
          {disabledReason}
        </TooltipContent>
      ) : null}
    </Tooltip>
  )
}
