let mergeReadySpineConfirmSuppressed = false
let mergeReadySpineConfirmShown = false

export function shouldShowMergeReadySpineConfirm(taskCount: number): boolean {
  if (mergeReadySpineConfirmSuppressed) {
    return false
  }
  if (mergeReadySpineConfirmShown) {
    return false
  }
  return taskCount > 1
}

export function recordMergeReadySpineConfirmResult(options: {
  suppressFuture: boolean
}): void {
  mergeReadySpineConfirmShown = true
  if (options.suppressFuture) {
    mergeReadySpineConfirmSuppressed = true
  }
}

export function suppressMergeReadySpineConfirmForSession(): void {
  mergeReadySpineConfirmSuppressed = true
}
