import type { MergeRun } from "@/lib/graph-utils"

export type RebaseRemediation = {
  attachCommand: string | null
  message: string | null
  branchName: string | null
  worktreePath: string | null
}

export function getRebaseRemediation(
  mergeRun: MergeRun | null | undefined,
  fallbackBranchName: string | null,
): RebaseRemediation | null {
  if (
    !mergeRun ||
    mergeRun.status !== "blocked" ||
    mergeRun.blockedStepKind !== "rebase"
  ) {
    return null
  }

  const operation = mergeRun.operation ?? "merge"
  const worktreePath = mergeRun.blockedWorktreePath ?? null
  const branchName = mergeRun.blockedBranchName ?? fallbackBranchName
  const attachCommand =
    mergeRun.blockedTaskId !== null && mergeRun.blockedTaskId !== undefined
      ? `rn agent attach --task ${mergeRun.blockedTaskId}`
      : null

  const message =
    worktreePath && branchName
      ? [
          `We hit a rebase conflict while ${
            operation === "restack" ? "restacking" : "merging"
          } the stack.`,
          "",
          `Branch: ${branchName}`,
          `Worktree: ${worktreePath}`,
          "",
          "Please:",
          "- Review and resolve the conflict(s).",
          `- \`cd ${worktreePath}\``,
          "- `git status` (to see conflicted files)",
          "- `git add -A`",
          "- `git rebase --continue`",
          "- Repeat until the rebase completes (resolving any further conflicts).",
          "",
          `Once the rebase finishes and the worktree is clean, let me know so I can resume the ${operation} run.`,
        ].join("\n")
      : null

  return {
    attachCommand,
    message,
    branchName,
    worktreePath,
  }
}
