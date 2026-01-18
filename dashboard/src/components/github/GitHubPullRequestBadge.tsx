import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { useGitHubStatus } from "@/hooks/useGitHubStatus"
import {
  gitHubPullRequestUrl,
  useGitHubPullRequest,
} from "@/hooks/useGitHubPullRequest"
import { cn } from "@/lib/utils"
import type { Task } from "@/lib/graph-utils"
import { GitPullRequest } from "lucide-react"

export type GitHubPullRequestBadgeProps = {
  prId: string
  taskState: Task["state"]
  activeBackground?: boolean
}

function prBorderClass(options: {
  taskState: Task["state"]
  pr: {
    merged: boolean
    draft: boolean
    state: "open" | "closed"
  } | null
}) {
  const { taskState, pr } = options

  if (pr) {
    if (pr.merged) {
      return "border-emerald-400/70"
    }
    if (pr.draft) {
      return "border-amber-300/70"
    }
    return pr.state === "open"
      ? "border-sky-400/70"
      : "border-muted-foreground/40"
  }

  return taskState === "done"
    ? "border-emerald-400/40"
    : "border-muted-foreground/40"
}

function prStateLabel(
  pr: { merged: boolean draft: boolean state: "open" | "closed" } | null,
) {
  if (!pr) {
    return null
  }
  if (pr.merged) {
    return "Merged"
  }
  if (pr.draft) {
    return "Draft"
  }
  return pr.state === "open" ? "Open" : "Closed"
}

export function GitHubPullRequestBadge({
  prId,
  taskState,
  activeBackground = false,
}: GitHubPullRequestBadgeProps) {
  const { status: githubStatus } = useGitHubStatus()
  const canFetch = githubStatus?.connected ?? false

  const { ref, pullRequest } = useGitHubPullRequest(prId, {
    enabled: canFetch,
  })
  if (!ref) {
    return null
  }

  const prUrl = pullRequest?.url ?? gitHubPullRequestUrl(ref)
  const stateLabel = prStateLabel(pullRequest)
  const borderClass = prBorderClass({
    taskState,
    pr: pullRequest
      ? {
          merged: pullRequest.merged,
          draft: pullRequest.draft,
          state: pullRequest.state,
        }
      : null,
  })

  return (
    <Tooltip>
      <TooltipTrigger
        render={(tooltipTriggerProps) => (
          <button
            {...tooltipTriggerProps}
            type="button"
            className={cn(
              "group/github inline-flex h-6 items-center overflow-hidden whitespace-nowrap rounded-full border-2 shadow-sm backdrop-blur focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/30",
              "transition-colors duration-200",
              activeBackground
                ? "bg-accent/40"
                : "bg-transparent group-hover:bg-accent/40 group-focus-within:bg-accent/40",
              borderClass,
              tooltipTriggerProps.className,
            )}
            aria-label={`GitHub PR #${ref.number}`}
            onClick={() => window.open(prUrl, "_blank", "noreferrer")}
          >
            <span className="relative inline-flex size-6 shrink-0 items-center justify-center">
              <GitPullRequest className="size-3.5 text-muted-foreground" />
            </span>
            <span className="min-w-0 truncate pr-2 font-mono text-[0.625rem] leading-none text-muted-foreground">
              #{ref.number}
            </span>
          </button>
        )}
      />
      <TooltipContent side="bottom" sideOffset={10}>
        <div className="space-y-1">
          <div className="text-xs">
            GitHub PR #{ref.number}
            {stateLabel ? `: ${stateLabel}` : null}
          </div>
          {pullRequest?.title ? (
            <div className="text-xs text-muted-foreground">
              {pullRequest.title}
            </div>
          ) : null}
          {!canFetch ? (
            <div className="text-xs text-muted-foreground">
              Connect GitHub to show PR state
            </div>
          ) : null}
        </div>
      </TooltipContent>
    </Tooltip>
  )
}
