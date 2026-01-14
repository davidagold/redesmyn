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

export type AgentMessageConfirmKind = "structured_turn_in_progress" | "structured_session_conflict" | "interactive_busy"

export function AgentMessageConfirmDialog({
  open,
  kind,
  canInterrupt,
  pendingReason,
  onOpenChange,
  onInterruptAndSend,
  onStopAndSend,
  onSendAnyway,
}: {
  open: boolean
  kind: AgentMessageConfirmKind
  canInterrupt: boolean
  pendingReason?: string | null
  onOpenChange: (open: boolean) => void
  onInterruptAndSend: () => void
  onStopAndSend: () => void
  onSendAnyway: () => void
}) {
  const title = (() => {
    if (kind === "structured_session_conflict") {
      return "Stop agent session?"
    }
    if (kind === "structured_turn_in_progress") {
      return "Interrupt agent turn?"
    }
    return "Agent is busy"
  })()

  const description = (() => {
    if (kind === "structured_session_conflict") {
      return "Another agent session is already running for this task. Stop it and send your message?"
    }
    if (kind === "structured_turn_in_progress") {
      return "A structured agent turn is currently in progress. Interrupt it and send your message?"
    }
    if (canInterrupt) {
      return "The agent looks busy. Interrupt it before sending your message?"
    }
    return "The agent looks busy. Interrupt is unavailable; send anyway?"
  })()

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

          {kind === "interactive_busy" ? (
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

          {kind === "structured_session_conflict" ? (
            <AlertDialogAction
              disabledReason={pendingReason ?? null}
              onClick={(event) => {
                event.preventDefault()
                onStopAndSend()
              }}
            >
              Stop and send
            </AlertDialogAction>
          ) : null}

          {kind === "structured_turn_in_progress" ||
          (kind === "interactive_busy" && canInterrupt) ? (
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
