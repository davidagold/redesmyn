import { Button } from "@/components/ui/button"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"

export type AgentMessageConfirmMode = "structured" | "interactive"

export function AgentMessageConfirmDialog({
  open,
  mode,
  canInterrupt,
  pendingReason,
  onOpenChange,
  onInterruptAndSend,
  onSendAnyway,
}: {
  open: boolean
  mode: AgentMessageConfirmMode
  canInterrupt: boolean
  pendingReason?: string | null
  onOpenChange: (open: boolean) => void
  onInterruptAndSend: () => void
  onSendAnyway: () => void
}) {
  const title =
    mode === "structured" ? "Interrupt agent turn?" : "Agent is busy"

  const description =
    mode === "structured"
      ? "A structured agent turn is currently in progress. Interrupt it and send your message?"
      : canInterrupt
        ? "The agent looks busy. Interrupt it before sending your message?"
        : "The agent looks busy. Interrupt is unavailable; send anyway?"

  return (
    <AlertDialog
      open={open}
      onOpenChange={(nextOpen) => {
        if (pendingReason && !nextOpen) {
          return
        }
        onOpenChange(nextOpen)
      }}
    >
      <AlertDialogContent size="sm">
        <AlertDialogHeader>
          <AlertDialogTitle>{title}</AlertDialogTitle>
          <AlertDialogDescription>{description}</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <Button
            variant="outline"
            disabledReason={pendingReason ?? null}
            onClick={(event) => {
              event.preventDefault()
              onOpenChange(false)
            }}
          >
            Cancel
          </Button>

          {mode === "interactive" ? (
            <Button
              variant="secondary"
              disabledReason={pendingReason ?? null}
              onClick={(event) => {
                event.preventDefault()
                onSendAnyway()
              }}
            >
              Send anyway
            </Button>
          ) : null}

          {canInterrupt ? (
            <AlertDialogAction
              disabledReason={pendingReason ?? null}
              onClick={(event) => {
                event.preventDefault()
                onInterruptAndSend()
              }}
            >
              Interrupt and send
            </AlertDialogAction>
          ) : null}
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
