import { useEffect, useRef, useState } from "react"
import type { Task } from "@/lib/graph-utils"
import { previewMergeReadySpineMarkCount } from "@/lib/graph-utils"
import {
  recordMergeReadySpineConfirmResult,
  shouldShowMergeReadySpineConfirm,
} from "@/lib/session-flags"

export function useMergeReadySpineConfirm(options?: { resetKey?: unknown }) {
  const [open, setOpen] = useState(false)
  const [taskCount, setTaskCount] = useState(0)
  const [dontAskAgain, setDontAskAgain] = useState(false)
  const confirmActionRef = useRef<(() => void) | null>(null)

  useEffect(() => {
    setOpen(false)
    setTaskCount(0)
    setDontAskAgain(false)
    confirmActionRef.current = null
  }, [options?.resetKey])

  function confirm() {
    recordMergeReadySpineConfirmResult({ suppressFuture: dontAskAgain })
    setOpen(false)
    const action = confirmActionRef.current
    confirmActionRef.current = null
    action?.()
  }

  async function requestMergeReadySpine(options: {
    tasksById: Map<number, Task>
    leafTaskId: number
    onWarn: (warning: string) => void
    onProceed: (scope: "task" | "spine") => void | Promise<void>
  }) {
    const preview = previewMergeReadySpineMarkCount(
      options.tasksById,
      options.leafTaskId,
    )
    if (preview.warnings.length > 0) {
      options.onWarn(preview.warnings[0] ?? "Could not resolve merge spine.")
      await options.onProceed("task")
      return
    }

    if (shouldShowMergeReadySpineConfirm(preview.count)) {
      setTaskCount(preview.count)
      confirmActionRef.current = () => void options.onProceed("spine")
      setOpen(true)
      return
    }

    await options.onProceed("spine")
  }

  return {
    dialog: {
      open,
      taskCount,
      dontAskAgain,
      setOpen,
      setDontAskAgain,
      confirm,
    },
    requestMergeReadySpine,
  }
}
