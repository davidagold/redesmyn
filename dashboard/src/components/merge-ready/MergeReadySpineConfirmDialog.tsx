import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"

export function MergeReadySpineConfirmDialog({
  open,
  taskCount,
  dontAskAgain,
  pendingReason,
  onOpenChange,
  onDontAskAgainChange,
  onConfirm,
}: {
  open: boolean
  taskCount: number
  dontAskAgain: boolean
  pendingReason?: string | null
  onOpenChange: (open: boolean) => void
  onDontAskAgainChange: (next: boolean) => void
  onConfirm: () => void
}) {
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
          <AlertDialogTitle>Mark merge-ready spine?</AlertDialogTitle>
          <AlertDialogDescription>
            This will mark{" "}
            <span className="font-medium text-foreground/90">{taskCount}</span>{" "}
            task{taskCount === 1 ? "" : "s"} ready to merge.
            <div className="mt-2 flex items-center justify-between gap-3 rounded-md bg-muted/30 px-3 py-2">
              <div className="text-xs text-muted-foreground">
                Don&apos;t ask again (this session)
              </div>
              <Switch
                checked={dontAskAgain}
                onCheckedChange={onDontAskAgainChange}
              />
            </div>
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <Button
            variant="outline"
            disabledReason={pendingReason ?? null}
            onClick={(e) => {
              e.preventDefault()
              onOpenChange(false)
            }}
          >
            Cancel
          </Button>
          <AlertDialogAction
            disabledReason={pendingReason ?? null}
            onClick={(e) => {
              e.preventDefault()
              onConfirm()
            }}
          >
            Mark ready
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
