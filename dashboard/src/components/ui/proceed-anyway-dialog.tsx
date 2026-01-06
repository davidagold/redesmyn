import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Button } from "@/components/ui/button"
import { Loader2 } from "lucide-react"

export function ProceedAnywayDialog({
  open,
  pending,
  actionLabel,
  detail,
  onOpenChange,
  onProceed,
}: {
  open: boolean
  pending: boolean
  actionLabel: string
  detail?: string | null
  onOpenChange: (open: boolean) => void
  onProceed: () => void
}) {
  return (
    <AlertDialog
      open={open}
      onOpenChange={(nextOpen) => {
        if (pending && !nextOpen) {
          return
        }
        onOpenChange(nextOpen)
      }}
    >
      <AlertDialogContent size="sm">
        <AlertDialogHeader>
          <AlertDialogTitle>Proceed anyway?</AlertDialogTitle>
          <AlertDialogDescription>
            This {actionLabel} affects running tasks/agents.
            {detail ? (
              <div className="mt-2 whitespace-pre-line text-xs text-muted-foreground">
                {detail}
              </div>
            ) : null}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <Button
            variant="outline"
            disabledReason={pending ? "Action in progress" : null}
            onClick={(e) => {
              e.preventDefault()
              onOpenChange(false)
            }}
          >
            Cancel
          </Button>
          <AlertDialogAction
            disabledReason={pending ? "Action in progress" : null}
            onClick={(e) => {
              e.preventDefault()
              onProceed()
            }}
          >
            {pending ? (
              <>
                <Loader2 className="size-4 animate-spin" />
                Proceeding…
              </>
            ) : (
              "Proceed"
            )}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
