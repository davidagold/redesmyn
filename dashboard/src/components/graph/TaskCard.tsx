import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type ComponentProps,
} from "react"
import { useIsFetching } from "@tanstack/react-query"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ResourceBadge } from "@/components/ui/resource-badge"
import { Textarea } from "@/components/ui/textarea"
import { ApiHttpError, type TaskAgentMessageConflictAction } from "@/api"
import { queryKeys } from "@/api/queryKeys"
import {
  useCancelMergeRunMutation,
  useOpenTaskGithubPullRequestMutation,
  useMergeTaskMutation,
  useRestartTaskAgentMutation,
  useRestackTaskMutation,
  useResumeMergeRunMutation,
  useSetTaskMergeReadyMutation,
  useStartTaskAgentMutation,
  useStopTaskAgentMutation,
  useTaskAgentMessageMutation,
} from "@/api/mutations"
import { copyToClipboard } from "@/lib/clipboard"
import { cn } from "@/lib/utils"
import {
  type AgentSession,
  type GraphNode,
  type MergeRun,
  type Task,
} from "@/lib/graph-utils"
import { getRebaseRemediation } from "@/lib/merge-remediation"
import {
  isRunningAgentsConflict,
  runningAgentsSummary,
} from "@/lib/runningAgentsConflict"
import { AgentStatusIcon } from "@/components/agents/AgentStatusIcon"
import { GithubIcon } from "@/components/github/GithubIcon"
import { LinearIcon } from "@/components/linear/LinearIcon"
import { Markdown, MarkdownInline } from "@/components/markdown"
import { ProceedAnywayDialog } from "@/components/ui/proceed-anyway-dialog"
import {
  inferStructuredAgentFromCommand,
  labelForAgentKind,
} from "@/lib/agent-kind"
import { Shimmer } from "@/components/ui/shimmer"
import {
  AgentMessageConfirmDialog,
  type AgentMessageConfirmKind,
} from "@/components/agents/AgentMessageConfirmDialog"
import {
  ActionErrorCallout,
  BlockedRebaseCallout,
  MergeReadySpineWarningCallout,
  ResumableMergeCallout,
  SuccessToastCallout,
  type TaskCardActionError,
} from "@/components/graph/TaskCardCallouts"
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { MergeReadySpineConfirmDialog } from "@/components/merge-ready/MergeReadySpineConfirmDialog"
import { useMergeReadySpineConfirm } from "@/hooks/useMergeReadySpineConfirm"
import { useGitHubStatus } from "@/hooks/useGitHubStatus"
import {
  ArrowUp,
  EllipsisVertical,
  AlertTriangle,
  GitBranch,
  GitMerge,
  Ghost,
  Layers,
  MessageSquareText,
  Play,
  RotateCcw,
  Square,
  Terminal,
} from "lucide-react"

interface TaskCardProps {
  node: GraphNode
  task?: Task
  tasksById: Map<number, Task>
  agentSession?: AgentSession
  mergeRun?: MergeRun
  blockingMergeRun?: MergeRun
  branchLabel: string
  branchLabelProvisional?: boolean
  gitMutationsDisabledReason?: string | null
  stackProjectionsFresh: boolean
  harnessCommand: string
  detach: boolean
  isSelected: boolean
  isHighlighted?: boolean
  onSelect: (options: { additive: boolean }) => void
}

interface MergeAllowRunningPrompt {
  kind: "merge"
  cascade: boolean
  restackMode: "strict" | "merge_then_restack"
  detail?: string | null
}

interface ResumeAllowRunningPrompt {
  kind: "resume"
  runId: string
  operation: "merge" | "restack"
  detail?: string | null
}

interface RestackAllowRunningPrompt {
  kind: "restack"
  scope: "descendants" | "spine"
  detail?: string | null
}

type AllowRunningPrompt = MergeAllowRunningPrompt | ResumeAllowRunningPrompt | RestackAllowRunningPrompt

type MarkdownLeadingBoldSplit = {
  title: string | null
  body: string
}

function splitAgentPreviewHeading(preview: string): MarkdownLeadingBoldSplit {
  const trimmed = preview.trim()
  if (!trimmed) {
    return { title: null, body: "" }
  }

  const match =
    /^(?:[-*]\s*)?(?:[|:—–-]\s*)?(?:\*\*|__)([^*_][\s\S]*?)(?:\*\*|__)\s*([\s\S]*)$/.exec(
      trimmed,
    )
  if (!match) {
    return { title: null, body: trimmed }
  }

  const title = (match[1] ?? "").trim() || null
  let body = (match[2] ?? "").trimStart()

  body = body.replace(/^(?:[|:—–-])\s+/, "")
  body = body.replace(/^\|\s*/, "")

  return { title, body: body.trim() }
}

function linearStateTypeFromTaskState(state: Task["state"]) {
  switch (state) {
    case "todo":
      return "unstarted"
    case "in_progress":
      return "started"
    case "blocked":
      return "blocked"
    case "done":
      return "completed"
  }
}

function linearStateLabel(stateName: string | null, stateType: string | null) {
  const name = stateName?.trim()
  if (name) {
    return name
  }

  const normalized = stateType?.trim().toLowerCase() ?? ""
  if (normalized === "unstarted") {
    return "Todo"
  }
  if (normalized === "started") {
    return "In progress"
  }
  if (normalized === "blocked") {
    return "Blocked"
  }
  if (normalized === "completed") {
    return "Done"
  }
  return "Unknown"
}

function formatObservedAt(value: string | null) {
  if (!value) {
    return null
  }
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return null
  }
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "numeric",
    minute: "2-digit",
  }).format(date)
}

function statusSummary(
  task: Task | undefined,
  agentSession: AgentSession | undefined,
) {
  if (task?.state === "blocked") {
    return "blocked"
  }
  if (task?.state === "done") {
    return "done"
  }
  if (!agentSession) {
    return "not started"
  }
  return agentSession.status
}

function agentStatusTooltip(
  task: Task | undefined,
  agentSession: AgentSession | undefined,
) {
  if (!agentSession) {
    return "Agent not started"
  }

  if (agentSession.status === "running") {
    return "Agent running"
  }
  if (agentSession.status === "blocked") {
    return "Agent blocked"
  }
  if (agentSession.status === "error") {
    return "Agent error"
  }
  if (agentSession.status === "stopped") {
    return "Agent stopped"
  }

  const summary = statusSummary(task, agentSession)
  return summary === "not started" ? "Agent not started" : `Agent ${summary}`
}

const STRUCTURED_TURN_CONFLICT_PREFIX =
  "[task_agent_message_conflict:structured_turn_in_progress]"
const STRUCTURED_SESSION_CONFLICT_PREFIX =
  "[task_agent_message_conflict:structured_session_conflict]"

function stripAgentMessageConflictPrefix(detail: string): string {
  const trimmed = detail.trim()
  if (trimmed.startsWith(STRUCTURED_TURN_CONFLICT_PREFIX)) {
    return trimmed.slice(STRUCTURED_TURN_CONFLICT_PREFIX.length).trimStart()
  }
  if (trimmed.startsWith(STRUCTURED_SESSION_CONFLICT_PREFIX)) {
    return trimmed.slice(STRUCTURED_SESSION_CONFLICT_PREFIX.length).trimStart()
  }
  return trimmed
}

function hasAgentMessageConflictPrefix(
  detail: string | null | undefined,
): boolean {
  const trimmed = detail?.trim() ?? ""
  return (
    trimmed.startsWith(STRUCTURED_TURN_CONFLICT_PREFIX) ||
    trimmed.startsWith(STRUCTURED_SESSION_CONFLICT_PREFIX)
  )
}

function agentMessageConfirmKindForApiError(
  error: ApiHttpError,
  mode: "structured" | "interactive",
): AgentMessageConfirmKind | null {
  if (error.code === "structured_turn_in_progress") {
    return "structured_turn_in_progress"
  }
  if (error.code === "structured_session_conflict") {
    return "structured_session_conflict"
  }
  if (hasAgentMessageConflictPrefix(error.detail)) {
    return agentMessageConfirmKindForConflictDetail(error.detail, mode)
  }
  return null
}

function agentMessageConfirmKindForConflictDetail(
  detail: string | null | undefined,
  mode: "structured" | "interactive",
): AgentMessageConfirmKind {
  const trimmed = detail?.trim() ?? ""
  if (trimmed.startsWith(STRUCTURED_TURN_CONFLICT_PREFIX)) {
    return "structured_turn_in_progress"
  }
  if (trimmed.startsWith(STRUCTURED_SESSION_CONFLICT_PREFIX)) {
    return "structured_session_conflict"
  }
  if (mode !== "structured") {
    return "interactive_busy"
  }
  return "structured_turn_in_progress"
}

export function TaskCard({
  node,
  task,
  tasksById,
  mergeRun,
  blockingMergeRun,
  agentSession,
  branchLabel,
  branchLabelProvisional = false,
  gitMutationsDisabledReason,
  stackProjectionsFresh,
  harnessCommand,
  detach,
  isSelected,
  isHighlighted = false,
  onSelect,
}: TaskCardProps) {
  const startAgent = useStartTaskAgentMutation()
  const stopAgent = useStopTaskAgentMutation()
  const restartAgent = useRestartTaskAgentMutation()
  const setMergeReadyMutation = useSetTaskMergeReadyMutation()
  const mergeTaskMutation = useMergeTaskMutation()
  const restackTaskMutation = useRestackTaskMutation()
  const resumeMergeRunMutation = useResumeMergeRunMutation()
  const cancelMergeRunMutation = useCancelMergeRunMutation()

  const now = Date.now()
  const stackInSync = node.stackInSync ?? null
  const outOfSync = stackInSync === false
  const mergeRunStatus = mergeRun?.status ?? null
  const mergeRunOperation = mergeRun?.operation ?? "merge"
  const mergeConflictAssist = mergeRun?.conflictAssist ?? null
  const mergeRunBlocked = mergeRunStatus === "blocked"
  const rebaseRemediation = getRebaseRemediation(mergeRun, node.branchName)
  const mergeRunBlockedRebase = rebaseRemediation !== null
  const blockingRebaseRemediation = getRebaseRemediation(
    blockingMergeRun,
    node.branchName,
  )
  const blockingMergeRunStatus = blockingMergeRun?.status ?? null
  const blockingMergeRunBlockedRebase =
    blockingMergeRun?.blockedStepKind === "rebase" &&
    (blockingMergeRunStatus === "blocked" ||
      blockingMergeRunStatus === "resumable")
  const blockingConflictAssist = blockingMergeRun?.conflictAssist ?? null
  const gitDisabledReason = gitMutationsDisabledReason ?? null
  const agentKindLabel = agentSession
    ? labelForAgentKind(agentSession.agentKind)
    : null
  const agentMessagePreview = (() => {
    const raw = agentSession?.agentPreview?.lastAssistantMessagePreview ?? null
    const trimmed = raw?.trim() ?? ""
    return trimmed ? trimmed : null
  })()
  const agentLastMessageSource =
    agentSession?.agentPreview?.lastAssistantMessageSource ?? null
  const agentMessageSplit = agentMessagePreview
    ? splitAgentPreviewHeading(agentMessagePreview)
    : null
  const agentPreviewTitle = agentMessageSplit?.title ?? null
  const agentPreviewBody = agentMessageSplit?.body ?? null
  const agentLastMessageText = (() => {
    const raw = agentSession?.agentPreview?.lastAssistantMessageText ?? null
    const trimmed = raw?.trim() ?? ""
    return trimmed ? trimmed : null
  })()
  const capturedFinalOutputText =
    agentLastMessageSource === "last_message_file" ? agentLastMessageText : null
  const agentTurnState = agentSession?.agentSemanticStatus.turnState ?? null
  const showAgentPreviewShimmer =
    agentSession?.status === "running" && agentTurnState === "busy"
  const agentPreviewDimmed =
    agentSession?.status === "stopped" || agentSession?.status === "error"
  const linearIssueId = task?.linearIssueId ?? null
  const linearIdentifier = task?.linearIdentifier ?? null
  const linearPillLabel = linearIdentifier ?? "Linear"
  const linearStateType = task?.linearStateType ?? null
  const linearStateName = task?.linearStateName ?? null
  const linearStateObservedAt = task?.linearStateObservedAt ?? null
  const linearObservedAgeMs = linearStateObservedAt
    ? now - new Date(linearStateObservedAt).getTime()
    : null
  const linearObservedOld =
    linearObservedAgeMs !== null &&
    Number.isFinite(linearObservedAgeMs) &&
    linearObservedAgeMs > 30 * 60 * 1000
  const linearObservedAtLabel = formatObservedAt(linearStateObservedAt)

  const [pendingAction, setPendingAction] =
    useState<"start" | "stop" | "restart" | "attach" | null>(null)
  const [pendingMerge, setPendingMerge] =
    useState<"ready" | "merge" | "mergeStack" | "restack" | "resume" | "cancel" | "githubPr" | null>(
      null,
    )
  const [actionsMenuOpen, setActionsMenuOpen] = useState(false)
  const [agentComposerExpanded, setAgentComposerExpanded] = useState(false)
  const [agentComposerDraft, setAgentComposerDraft] = useState("")
  const [agentComposerError, setAgentComposerError] = useState<string | null>(
    null,
  )
  const [agentComposerConfirmOpen, setAgentComposerConfirmOpen] =
    useState(false)
  const [agentComposerConfirmKind, setAgentComposerConfirmKind] =
    useState<AgentMessageConfirmKind>("interactive_busy")
  const agentComposerTextareaRef = useRef<HTMLTextAreaElement | null>(null)
  const taskAgentMessageMutation = useTaskAgentMessageMutation()
  const mergeReady = Boolean(task?.mergeReadyAt)
  const expectedLinearStateType =
    task === undefined
      ? null
      : task.state === "done"
        ? "completed"
        : mergeReady
          ? "started"
          : linearStateTypeFromTaskState(task.state)
  const linearStateMatches =
    task !== undefined &&
    linearIssueId !== null &&
    linearStateType !== null &&
    expectedLinearStateType !== null &&
    linearStateType.toLowerCase() === expectedLinearStateType.toLowerCase()
  const linearStateMismatch =
    task !== undefined &&
    linearIssueId !== null &&
    linearStateType !== null &&
    expectedLinearStateType !== null &&
    linearStateType.toLowerCase() !== expectedLinearStateType.toLowerCase()
  const linearPillBorderClass = (() => {
    if (task?.state === "done") {
      return "border-green-950/80"
    }
    if (mergeReady && linearStateMatches) {
      return "border-emerald-400/70"
    }

    const normalized = linearStateType?.trim().toLowerCase() ?? ""
    if (normalized === "unstarted") {
      return "border-foreground/10 border-dashed"
    }
    if (normalized === "started") {
      return "border-foreground/10"
    }
    if (normalized === "blocked") {
      return "border-amber-400/80"
    }
    if (normalized === "completed") {
      return "border-green-950/70"
    }
    return "border-border/60"
  })()
  const [actionError, setActionError] = useState<TaskCardActionError | null>(
    null,
  )
  const [allowRunningPrompt, setAllowRunningPrompt] =
    useState<AllowRunningPrompt | null>(null)
  const [allowRunningConfirming, setAllowRunningConfirming] = useState(false)
  const [blockedRebaseCompletionNotice, setBlockedRebaseCompletionNotice] =
    useState<string | null>(null)
  const [mergeReadySpineNotice, setMergeReadySpineNotice] =
    useState<string | null>(null)
  const [githubPrNotice, setGithubPrNotice] = useState<string | null>(null)
  const [gitActionInProgress, setGitActionInProgress] = useState<{
    kind: "merge" | "mergeStack" | "restack" | "resume" | "ready" | "githubPr"
    startedAt: number
    initialMergeRunId: string | null
    initialMergeRunStatus: string | null
    initialMergeReadyAt: string | null
  } | null>(null)
  const mergeReadySpineConfirm = useMergeReadySpineConfirm()
  const githubStatus = useGitHubStatus()
  const openGithubPr = useOpenTaskGithubPullRequestMutation()
  const lastBlockedRebaseToastRunIdRef = useRef<string | null>(null)
  const prevBlockedRebaseRef = useRef<{
    runId: string | null
    blockedRebase: boolean
    hadAssist: boolean
    assistState: string | null
  }>({ runId: null, blockedRebase: false, hadAssist: false, assistState: null })

  const epicGraphIsFetching =
    useIsFetching({ queryKey: queryKeys.epicGraph(node.epicId) }) > 0

  function beginGitAction(
    kind: "merge" | "mergeStack" | "restack" | "resume" | "ready" | "githubPr",
  ) {
    setGitActionInProgress({
      kind,
      startedAt: Date.now(),
      initialMergeRunId: mergeRun?.runId ?? null,
      initialMergeRunStatus: mergeRun?.status ?? null,
      initialMergeReadyAt: task?.mergeReadyAt ?? null,
    })
  }

  useEffect(() => {
    if (!gitActionInProgress) {
      return
    }

    const elapsedMs = Date.now() - gitActionInProgress.startedAt
    if (elapsedMs > 15_000) {
      setGitActionInProgress(null)
      return
    }

    if (gitActionInProgress.kind === "githubPr") {
      if (pendingMerge !== "githubPr") {
        setGitActionInProgress(null)
      }
      return
    }

    if (gitActionInProgress.kind === "ready") {
      const currentReadyAt = task?.mergeReadyAt ?? null
      if (currentReadyAt !== gitActionInProgress.initialMergeReadyAt) {
        setGitActionInProgress(null)
      }
      return
    }

    const currentRunId = mergeRun?.runId ?? null
    const currentStatus = mergeRun?.status ?? null
    if (
      currentRunId !== gitActionInProgress.initialMergeRunId ||
      currentStatus !== gitActionInProgress.initialMergeRunStatus
    ) {
      setGitActionInProgress(null)
    }
  }, [
    gitActionInProgress,
    mergeRun?.runId,
    mergeRun?.status,
    pendingMerge,
    task?.mergeReadyAt,
  ])

  useEffect(() => {
    if (!gitActionInProgress) {
      return
    }
    const id = window.setTimeout(() => setGitActionInProgress(null), 15_000)
    return () => window.clearTimeout(id)
  }, [gitActionInProgress])

  const mergeAutoAssistActive =
    mergeRunStatus === "resumable" &&
    mergeConflictAssist?.active === true &&
    mergeConflictAssist.state !== "timed_out" &&
    mergeConflictAssist.state !== "unsupported" &&
    ((mergeRun?.blockedTaskId ?? null) === null ||
      (mergeRun?.blockedTaskId ?? null) === node.id)

  const blockingAutoAssistActive =
    blockingMergeRunBlockedRebase &&
    blockingConflictAssist?.active === true &&
    blockingConflictAssist.state !== "timed_out" &&
    blockingConflictAssist.state !== "unsupported"

  const showGitActionGlow =
    gitActionInProgress !== null ||
    mergeAutoAssistActive ||
    blockingAutoAssistActive

  useEffect(() => {
    if (!mergeReadySpineNotice) {
      return
    }
    const id = window.setTimeout(() => setMergeReadySpineNotice(null), 2_000)
    return () => window.clearTimeout(id)
  }, [mergeReadySpineNotice])

  useEffect(() => {
    if (!blockedRebaseCompletionNotice) {
      return
    }
    const id = window.setTimeout(
      () => setBlockedRebaseCompletionNotice(null),
      4_000,
    )
    return () => window.clearTimeout(id)
  }, [blockedRebaseCompletionNotice])

  useEffect(() => {
    const prev = prevBlockedRebaseRef.current
    const currentRunId = blockingMergeRun?.runId ?? null
    const currentAssistState = blockingConflictAssist?.state ?? null
    const currentHasAssist = currentAssistState !== null

    const blockedRebaseCleared =
      prev.blockedRebase &&
      !blockingMergeRunBlockedRebase &&
      prev.runId !== null &&
      (currentRunId === null || currentRunId !== prev.runId)

    if (
      blockedRebaseCleared &&
      lastBlockedRebaseToastRunIdRef.current !== prev.runId &&
      prev.hadAssist &&
      prev.assistState !== "timed_out" &&
      prev.assistState !== "unsupported"
    ) {
      setBlockedRebaseCompletionNotice("Auto-resolve complete.")
      lastBlockedRebaseToastRunIdRef.current = prev.runId
    }

    prevBlockedRebaseRef.current = {
      runId: currentRunId,
      blockedRebase: blockingMergeRunBlockedRebase,
      hadAssist: currentHasAssist,
      assistState: currentAssistState,
    }
  }, [
    blockingConflictAssist?.state,
    blockingMergeRun?.runId,
    blockingMergeRunBlockedRebase,
  ])

  useEffect(() => {
    if (!isSelected) {
      setAgentComposerExpanded(false)
      setAgentComposerConfirmOpen(false)
      setAgentComposerError(null)
    }
  }, [isSelected])

  function clearActionError() {
    setActionError(null)
  }

  function setActionErrorFromException(
    actionLabel: string,
    exception: unknown,
  ) {
    if (exception instanceof ApiHttpError) {
      const summary = exception.detail?.trim() || exception.message
      setActionError({
        title: `${actionLabel} failed`,
        summary,
        raw: exception.message,
      })
      return
    }

    const raw =
      exception instanceof Error ? exception.message : String(exception)
    const httpDetailMatch = raw.match(/failed \\(\\d+\\): (.+)$/)
    const summary = httpDetailMatch?.[1]?.trim() || raw
    setActionError({
      title: `${actionLabel} failed`,
      summary,
      raw,
    })
  }

  const agentStatus = agentSession?.status ?? null
  const agentContinuable =
    agentStatus === "stopped" &&
    agentSession !== undefined &&
    agentSession.agentCapabilities.canResumeById &&
    agentSession.externalSessionRef.type !== "none"
  const taskId = task?.id ?? null
  const canResumeMerge = mergeRunStatus === "resumable"
  const canCancelMergeRun =
    mergeRunStatus === "running" ||
    mergeRunStatus === "blocked" ||
    mergeRunStatus === "resumable"
  const canCancelMergeRunAbortGit =
    mergeRunStatus === "blocked" && Boolean(mergeRun?.blockedWorktreePath)

  const showResumableMergeCallout = useMemo(() => {
    if (!canResumeMerge || !mergeRun) {
      return false
    }

    const blockedTaskId = mergeRun.blockedTaskId ?? null
    return (
      (blockedTaskId === null || blockedTaskId === node.id) &&
      !blockingMergeRunBlockedRebase
    )
  }, [blockingMergeRunBlockedRebase, canResumeMerge, mergeRun, node.id])

  const quickActionsEnabled =
    taskId !== null && task?.state !== "blocked" && task?.state !== "done"

  const isRunning = agentStatus === "running" || agentStatus === "blocked"
  const canStart =
    (!agentSession || agentStatus === "stopped") && harnessCommand.trim()
  const canRestart = isRunning || agentStatus === "error"
  const canMerge =
    taskId !== null && task?.state !== "blocked" && task?.state !== "done"

  const githubPrDisabledReason =
    pendingMerge !== null
      ? "Action in progress"
      : canResumeMerge
        ? "Resolve the blocked merge/restack run first."
        : (gitDisabledReason ??
          (task?.branchName
            ? null
            : "Task has no branch backing (sync branches first).") ??
          (githubStatus.status?.connected === false
            ? "GitHub is not connected"
            : null))

  const agentComposerDraftTrimmed = agentComposerDraft.trim()
  const harnessStructuredKind = inferStructuredAgentFromCommand(harnessCommand)
  const composerInterfaceMode = harnessStructuredKind
    ? "structured"
    : "interactive"
  const agentLooksBusy =
    agentSession?.agentSemanticStatus.turnState === "busy" &&
    (agentStatus === "running" || agentStatus === "blocked")
  const agentCanInterrupt =
    agentSession?.agentCapabilities.canInterrupt ?? false
  const agentComposerPendingReason = taskAgentMessageMutation.isPending
    ? "Sending…"
    : null
  const agentComposerSendDisabledReason =
    agentComposerPendingReason ??
    (taskId === null
      ? "Task unavailable."
      : !harnessCommand.trim()
        ? "Set a harness command to send messages."
        : !agentComposerDraftTrimmed
          ? "Write a message."
          : null)

  async function handleAttach() {
    if (taskId === null) {
      return
    }
    setPendingAction("attach")
    clearActionError()
    try {
      await copyToClipboard(`rn agent attach --task ${taskId}`)
    } catch (e) {
      setActionErrorFromException("Copy attach command", e)
    } finally {
      setPendingAction(null)
    }
  }

  async function handleOpenGithubPullRequest() {
    if (taskId === null) {
      return
    }
    if (!task?.branchName) {
      setActionError({
        title: "Open PR failed",
        summary: "Task has no branch backing.",
        raw: "Task has no branch backing.",
      })
      return
    }

    setPendingMerge("githubPr")
    beginGitAction("githubPr")
    clearActionError()
    setGithubPrNotice(null)

    const popup = window.open("about:blank", "_blank", "noreferrer")
    const popupBlocked = !popup

    if (popup) {
      try {
        popup.document.title = `Opening PR… (task ${taskId})`
      } catch {
        // Best-effort only.
      }
    }

    try {
      const result = await openGithubPr.mutateAsync({
        epicId: node.epicId,
        taskId,
      })
      if (popup) {
        popup.location.assign(result.url)
        return
      }

      window.open(result.url, "_blank", "noreferrer")
      if (popupBlocked) {
        void copyToClipboard(result.url)
          .then(() => setGithubPrNotice("Popups blocked; PR URL copied."))
          .catch(() =>
            setGithubPrNotice("Popups blocked; PR URL ready to copy."),
          )
      }
    } catch (e) {
      popup?.close()
      setActionErrorFromException("Open PR", e)
    } finally {
      setPendingMerge(null)
    }
  }

  async function handleStart() {
    if (taskId === null) {
      return
    }
    if (!harnessCommand.trim()) {
      return
    }
    setPendingAction("start")
    clearActionError()
    try {
      await startAgent.mutateAsync({
        epicId: node.epicId,
        taskId,
        request: { harness: harnessCommand, detach },
      })
    } catch (e) {
      setActionErrorFromException("Start agent", e)
    } finally {
      setPendingAction(null)
    }
  }

  async function submitAgentMessage(options?: {
    onConflict?: TaskAgentMessageConflictAction
  }) {
    if (taskId === null) {
      return
    }
    if (!agentComposerDraftTrimmed) {
      return
    }
    setAgentComposerError(null)

    try {
      await taskAgentMessageMutation.mutateAsync({
        epicId: node.epicId,
        taskId,
        request: {
          message: agentComposerDraftTrimmed,
          onConflict: options?.onConflict ?? "fail",
          preferredInterfaceMode: "auto",
        },
      })
      setAgentComposerDraft("")
      requestAnimationFrame(() => agentComposerTextareaRef.current?.focus())
    } catch (e) {
      if (e instanceof ApiHttpError) {
        if (e.status === 409) {
          const confirmKind = agentMessageConfirmKindForApiError(
            e,
            composerInterfaceMode,
          )
          if (confirmKind) {
            setAgentComposerConfirmKind(confirmKind)
            setAgentComposerConfirmOpen(true)
          }
        }
        setAgentComposerError(
          e.detail ? stripAgentMessageConflictPrefix(e.detail) : e.message,
        )
        return
      }
      const raw = e instanceof Error ? e.message : String(e)
      setAgentComposerError(raw)
    }
  }

  async function handleStop() {
    if (taskId === null) {
      return
    }
    setPendingAction("stop")
    clearActionError()
    try {
      await stopAgent.mutateAsync({ epicId: node.epicId, taskId })
    } catch (e) {
      setActionErrorFromException("Stop agent", e)
    } finally {
      setPendingAction(null)
    }
  }

  async function handleRestart() {
    if (taskId === null) {
      return
    }
    setPendingAction("restart")
    clearActionError()
    try {
      await restartAgent.mutateAsync({
        epicId: node.epicId,
        taskId,
        request: { detach: true },
      })
    } catch (e) {
      setActionErrorFromException("Restart agent", e)
    } finally {
      setPendingAction(null)
    }
  }

  async function commitMergeReady(next: boolean, scope?: "task" | "spine") {
    if (taskId === null) {
      return
    }
    if (next && !task?.branchName) {
      return
    }
    if (setMergeReadyMutation.isPending) {
      return
    }
    beginGitAction("ready")
    setPendingMerge("ready")
    clearActionError()
    try {
      await setMergeReadyMutation.mutateAsync({ taskId, ready: next, scope })
    } catch (e) {
      setGitActionInProgress(null)
      setActionErrorFromException("Set merge readiness", e)
    } finally {
      setPendingMerge(null)
    }
  }

  async function handleToggleMergeReady(next: boolean) {
    if (!next) {
      await commitMergeReady(false, "task")
      return
    }
    if (taskId === null) {
      return
    }
    await mergeReadySpineConfirm.requestMergeReadySpine({
      tasksById,
      leafTaskId: taskId,
      onWarn: setMergeReadySpineNotice,
      onProceed: (scope) => commitMergeReady(true, scope),
    })
  }

  async function handleMerge({ cascade }: { cascade: boolean }) {
    if (taskId === null) {
      return
    }
    beginGitAction(cascade ? "mergeStack" : "merge")
    setPendingMerge(cascade ? "mergeStack" : "merge")
    clearActionError()

    const restackMode: "strict" | "merge_then_restack" = "strict"
    const actionLabel = cascade ? "Merge and Restack" : "Merge"
    try {
      await mergeTaskMutation.mutateAsync({
        epicId: node.epicId,
        taskId,
        request: { cascade, restackMode },
      })
    } catch (e) {
      setGitActionInProgress(null)
      if (e instanceof ApiHttpError && e.status === 409) {
        if (isRunningAgentsConflict(e)) {
          setAllowRunningPrompt({
            kind: "merge",
            cascade,
            restackMode,
            detail: runningAgentsSummary(e),
          })
          return
        }
      }
      setActionErrorFromException(actionLabel, e)
      return
    } finally {
      setPendingMerge(null)
    }
  }

  async function handleMergeThenRestack() {
    if (taskId === null) {
      return
    }
    beginGitAction("mergeStack")
    setPendingMerge("mergeStack")
    clearActionError()
    const actionLabel = "Merge then Restack"
    const restackMode: "strict" | "merge_then_restack" = "merge_then_restack"
    try {
      await mergeTaskMutation.mutateAsync({
        epicId: node.epicId,
        taskId,
        request: { cascade: true, restackMode },
      })
    } catch (e) {
      setGitActionInProgress(null)
      if (e instanceof ApiHttpError && e.status === 409) {
        if (isRunningAgentsConflict(e)) {
          setAllowRunningPrompt({
            kind: "merge",
            cascade: true,
            restackMode,
            detail: runningAgentsSummary(e),
          })
          return
        }
      }
      setActionErrorFromException(actionLabel, e)
      return
    } finally {
      setPendingMerge(null)
    }
  }

  async function handleResumeMerge() {
    if (!mergeRun?.runId) {
      return
    }
    beginGitAction("resume")
    setPendingMerge("resume")
    clearActionError()
    try {
      await resumeMergeRunMutation.mutateAsync({
        epicId: node.epicId,
        runId: mergeRun.runId,
        request: {},
      })
    } catch (e) {
      setGitActionInProgress(null)
      if (e instanceof ApiHttpError && e.status === 409) {
        if (isRunningAgentsConflict(e)) {
          setAllowRunningPrompt({
            kind: "resume",
            runId: mergeRun.runId,
            operation: mergeRunOperation,
            detail: runningAgentsSummary(e),
          })
          return
        }
      }
      setActionErrorFromException(
        mergeRunOperation === "restack" ? "Resume restack" : "Resume merge",
        e,
      )
    } finally {
      setPendingMerge(null)
    }
  }

  async function handleResumeBlockingMerge() {
    if (!blockingMergeRun?.runId) {
      return
    }
    setPendingMerge("resume")
    clearActionError()
    const operation = blockingMergeRun.operation ?? "merge"
    try {
      await resumeMergeRunMutation.mutateAsync({
        epicId: node.epicId,
        runId: blockingMergeRun.runId,
        request: {},
      })
    } catch (e) {
      if (e instanceof ApiHttpError && e.status === 409) {
        if (isRunningAgentsConflict(e)) {
          setAllowRunningPrompt({
            kind: "resume",
            runId: blockingMergeRun.runId,
            operation,
            detail: runningAgentsSummary(e),
          })
          return
        }
      }
      setActionErrorFromException(
        operation === "restack" ? "Resume restack" : "Resume merge",
        e,
      )
    } finally {
      setPendingMerge(null)
    }
  }

  async function handleCancelMergeRun({ abortGit }: { abortGit: boolean }) {
    if (!mergeRun?.runId) {
      return
    }
    setPendingMerge("cancel")
    clearActionError()
    try {
      await cancelMergeRunMutation.mutateAsync({
        epicId: node.epicId,
        runId: mergeRun.runId,
        request: { abortGit },
      })
    } catch (e) {
      setActionErrorFromException(
        abortGit ? "Cancel + abort git" : "Cancel run",
        e,
      )
    } finally {
      setPendingMerge(null)
    }
  }

  async function handleRestack({ scope }: { scope: "descendants" | "spine" }) {
    if (taskId === null) {
      return
    }
    beginGitAction("restack")
    setPendingMerge("restack")
    clearActionError()
    const actionLabel = "Restack"
    try {
      await restackTaskMutation.mutateAsync({
        epicId: node.epicId,
        taskId,
        request: { scope },
      })
    } catch (e) {
      setGitActionInProgress(null)
      if (e instanceof ApiHttpError && e.status === 409) {
        if (isRunningAgentsConflict(e)) {
          setAllowRunningPrompt({
            kind: "restack",
            scope,
            detail: runningAgentsSummary(e),
          })
          return
        }
      }
      setActionErrorFromException(actionLabel, e)
      return
    } finally {
      setPendingMerge(null)
    }
  }

  async function confirmAllowRunning() {
    if (!allowRunningPrompt) {
      return
    }

    setAllowRunningConfirming(true)
    clearActionError()

    const actionLabel =
      allowRunningPrompt.kind === "merge"
        ? allowRunningPrompt.cascade
          ? allowRunningPrompt.restackMode === "merge_then_restack"
            ? "Merge then Restack"
            : "Merge and Restack"
          : "Merge"
        : allowRunningPrompt.kind === "restack"
          ? "Restack"
          : allowRunningPrompt.operation === "restack"
            ? "Resume restack"
            : "Resume merge"

    try {
      if (allowRunningPrompt.kind === "merge") {
        if (taskId === null) {
          throw new Error("Task id missing for merge confirmation.")
        }
        beginGitAction(allowRunningPrompt.cascade ? "mergeStack" : "merge")
        setPendingMerge(allowRunningPrompt.cascade ? "mergeStack" : "merge")
        await mergeTaskMutation.mutateAsync({
          epicId: node.epicId,
          taskId,
          request: {
            cascade: allowRunningPrompt.cascade,
            restackMode: allowRunningPrompt.restackMode,
            allowRunning: true,
          },
        })
      } else if (allowRunningPrompt.kind === "restack") {
        if (taskId === null) {
          throw new Error("Task id missing for restack confirmation.")
        }
        beginGitAction("restack")
        setPendingMerge("restack")
        await restackTaskMutation.mutateAsync({
          epicId: node.epicId,
          taskId,
          request: { scope: allowRunningPrompt.scope, allowRunning: true },
        })
      } else {
        beginGitAction("resume")
        setPendingMerge("resume")
        await resumeMergeRunMutation.mutateAsync({
          epicId: node.epicId,
          runId: allowRunningPrompt.runId,
          request: { allowRunning: true },
        })
      }
      setAllowRunningPrompt(null)
    } catch (e) {
      setAllowRunningPrompt(null)
      setGitActionInProgress(null)
      setActionErrorFromException(actionLabel, e)
    } finally {
      setPendingMerge(null)
      setAllowRunningConfirming(false)
    }
  }

  function renderQuickActionButton({
    tooltip,
    disabledReason,
    ...props
  }: { tooltip: string } & ComponentProps<typeof Button>) {
    if (disabledReason) {
      return <Button {...props} disabledReason={disabledReason} />
    }

    return (
      <Tooltip>
        <TooltipTrigger
          render={(triggerProps) => (
            <Button
              {...triggerProps}
              {...props}
              className={cn(props.className, triggerProps.className)}
            />
          )}
        />
        <TooltipContent side="bottom" sideOffset={10} showArrow={false}>
          {tooltip}
        </TooltipContent>
      </Tooltip>
    )
  }

  const quickActions =
    quickActionsEnabled && taskId !== null ? (
      isRunning ? (
        <>
          {renderQuickActionButton({
            tooltip: "Copy attach command",
            variant: "ghost",
            size: "icon-xs",
            "aria-label": "Copy attach command",
            disabledReason:
              pendingAction !== null ? "Action in progress" : null,
            onClick: (e) => {
              e.preventDefault()
              e.stopPropagation()
              void handleAttach()
            },
            children: <Terminal className="size-3" />,
          })}
          {renderQuickActionButton({
            tooltip: "Restart agent",
            variant: "ghost",
            size: "icon-xs",
            "aria-label": "Restart agent",
            disabledReason:
              pendingAction !== null ? "Action in progress" : null,
            onClick: (e) => {
              e.preventDefault()
              e.stopPropagation()
              void handleRestart()
            },
            children: <RotateCcw className="size-3" />,
          })}
          {renderQuickActionButton({
            tooltip: "Stop agent",
            variant: "ghost",
            size: "icon-xs",
            className: "text-destructive hover:bg-destructive/10",
            "aria-label": "Stop agent",
            disabledReason:
              pendingAction !== null ? "Action in progress" : null,
            onClick: (e) => {
              e.preventDefault()
              e.stopPropagation()
              void handleStop()
            },
            children: <Square className="size-3" />,
          })}
        </>
      ) : canRestart ? (
        renderQuickActionButton({
          tooltip: "Restart agent",
          variant: "ghost",
          size: "icon-xs",
          "aria-label": "Restart agent",
          disabledReason: pendingAction !== null ? "Action in progress" : null,
          onClick: (e) => {
            e.preventDefault()
            e.stopPropagation()
            void handleRestart()
          },
          children: <RotateCcw className="size-3" />,
        })
      ) : canStart ? (
        renderQuickActionButton({
          tooltip: "Start agent",
          variant: "ghost",
          size: "icon-xs",
          "aria-label": "Start agent",
          disabledReason:
            pendingAction !== null
              ? "Action in progress"
              : !harnessCommand.trim()
                ? "Set an agent command in Configure"
                : null,
          onClick: (e) => {
            e.preventDefault()
            e.stopPropagation()
            void handleStart()
          },
          children: <Play className="size-3" />,
        })
      ) : null
    ) : null

  const tooltip = agentStatusTooltip(task, agentSession)

  const rebaseAttachCommand = rebaseRemediation?.attachCommand ?? null
  const rebaseRemediationMessage = rebaseRemediation?.message ?? null
  const blockingAttachCommand = blockingRebaseRemediation?.attachCommand ?? null
  const blockingRemediationMessage = blockingRebaseRemediation?.message ?? null

  const gitAttentionTooltipLines: string[] = []
  if (mergeRunBlocked) {
    const blockedOnSpine =
      (mergeRun as { blockedOnSpine?: boolean | null } | null | undefined)
        ?.blockedOnSpine ?? null
    gitAttentionTooltipLines.push(
      blockedOnSpine === false
        ? "Restack blocked (conflicts)."
        : "Merge blocked (conflicts).",
    )
    if (mergeRun?.blockedBranchName) {
      gitAttentionTooltipLines.push(`Branch: ${mergeRun.blockedBranchName}`)
    }
  }
  if (outOfSync) {
    gitAttentionTooltipLines.push(
      "Branch is out of sync with its effective upstream (ignores merged ancestors).",
    )
    if (!stackProjectionsFresh) {
      gitAttentionTooltipLines.push(
        "Sync projection may be stale (telemetry stale).",
      )
    }
  }

  const mergeAttentionIcon = mergeRunBlocked ? (
    <AlertTriangle className="size-4 text-destructive/80" />
  ) : null

  const syncAttentionIcon = outOfSync ? (
    <GitBranch
      className={cn(
        "size-4",
        task?.state === "done"
          ? "text-muted-foreground/60"
          : "text-amber-300/70",
      )}
    />
  ) : null

  const gitAttentionIcon = mergeAttentionIcon || syncAttentionIcon

  const resumableMergeSummary = useMemo(() => {
    if (!showResumableMergeCallout) {
      return null
    }

    if (!mergeConflictAssist) {
      return "Conflicts resolved; ready to resume."
    }

    if (mergeConflictAssist.state === "timed_out") {
      return "Manual resolution needed (timed out)."
    }
    if (mergeConflictAssist.state === "unsupported") {
      return "Manual resolution needed (unavailable)."
    }

    if (!mergeConflictAssist.active) {
      return "Conflicts resolved; ready to resume."
    }

    switch (mergeConflictAssist.state) {
      case "waiting_for_agent_ready":
        return mergeConflictAssist.detail
          ? "Auto-resolving… unable to start agent."
          : "Auto-resolving… starting agent turn."
      case "sent_waiting_for_turn_complete":
        return mergeConflictAssist.messageSentAt
          ? "Auto-resolving… waiting for agent."
          : "Auto-resolving…"
      case "waiting_for_repo_clean":
        return "Auto-resolving… waiting for repo clean."
      case "ready_to_resume":
        return "Auto-resolving… resuming."
      case "resumed":
        return "Auto-resolving… resumed."
      default:
        return "Auto-resolving…"
    }
  }, [mergeConflictAssist, showResumableMergeCallout])

  const resumableMergeDetail = useMemo(() => {
    if (!showResumableMergeCallout) {
      return null
    }

    const summary = resumableMergeSummary
    const extra = mergeConflictAssist?.detail?.trim() ?? ""

    if (extra) {
      if (!summary || extra === summary) {
        return extra
      }
      return `${summary}\n${extra}`
    }

    if (
      mergeConflictAssist?.active === true ||
      mergeConflictAssist?.state === "timed_out" ||
      mergeConflictAssist?.state === "unsupported"
    ) {
      return summary
    }

    return null
  }, [
    mergeConflictAssist?.active,
    mergeConflictAssist?.detail,
    mergeConflictAssist?.state,
    resumableMergeSummary,
    showResumableMergeCallout,
  ])

  const resumableMergeVariant = useMemo(() => {
    if (!showResumableMergeCallout) {
      return "emerald" as const
    }
    if (
      mergeConflictAssist?.active &&
      mergeConflictAssist.state !== "timed_out" &&
      mergeConflictAssist.state !== "unsupported"
    ) {
      return "amber" as const
    }
    return "emerald" as const
  }, [mergeConflictAssist, showResumableMergeCallout])

  const allowRunningActionLabel =
    allowRunningPrompt?.kind === "merge"
      ? allowRunningPrompt.cascade
        ? allowRunningPrompt.restackMode === "merge_then_restack"
          ? "Merge then Restack"
          : "Merge and Restack"
        : "Merge"
      : allowRunningPrompt?.kind === "restack"
        ? "Restack"
        : allowRunningPrompt?.kind === "resume"
          ? allowRunningPrompt.operation === "restack"
            ? "Resume restack"
            : "Resume merge"
          : null

  const blockedRebaseSummary = useMemo(() => {
    if (!blockingMergeRunBlockedRebase) {
      return null
    }
    const isResumable = blockingMergeRunStatus === "resumable"
    const agentRunning = agentStatus === "running"
    const assist = blockingConflictAssist
    const assistTaskId = assist?.agentTaskId ?? null
    const assistSuffix =
      assistTaskId !== null && assistTaskId !== node.id
        ? ` (on T-${assistTaskId})`
        : ""
    if (!assist) {
      if (agentRunning) {
        return "Auto-resolving… agent working."
      }
      return isResumable
        ? "Conflicts resolved; ready to resume."
        : "Resolve conflicts, then resume."
    }

    if (assist.state === "timed_out") {
      return "Manual resolution needed (timed out)."
    }
    if (assist.state === "unsupported") {
      return "Manual resolution needed (unavailable)."
    }

    switch (assist.state) {
      case "waiting_for_agent_ready":
        return assist.detail
          ? `Auto-resolving… unable to start agent${assistSuffix}.`
          : `Auto-resolving… starting agent turn${assistSuffix}.`
      case "sent_waiting_for_turn_complete":
        return assist.messageSentAt
          ? `Auto-resolving… waiting for agent${assistSuffix}.`
          : `Auto-resolving${assistSuffix}…`
      case "waiting_for_repo_clean":
        return "Auto-resolving… waiting for repo clean."
      case "ready_to_resume":
        return "Auto-resolving… resuming."
      case "resumed":
        return "Auto-resolving… resumed."
      default:
        if (assist.active || agentRunning) {
          return `Auto-resolving${assistSuffix}…`
        }
        return isResumable
          ? "Conflicts resolved; ready to resume."
          : "Resolve conflicts, then resume."
    }
  }, [
    agentStatus,
    blockingConflictAssist,
    blockingMergeRunBlockedRebase,
    blockingMergeRunStatus,
    node.id,
  ])

  const blockedRebaseBadgeLabel = useMemo(() => {
    if (!blockingMergeRunBlockedRebase) {
      return null
    }
    const assist = blockingConflictAssist
    if (
      assist?.active &&
      assist.state !== "timed_out" &&
      assist.state !== "unsupported"
    ) {
      return "Auto-resolving"
    }
    if (blockingMergeRunStatus === "resumable") {
      return "Ready to resume"
    }
    return "Rebase blocked"
  }, [
    blockingConflictAssist,
    blockingMergeRunBlockedRebase,
    blockingMergeRunStatus,
  ])

  const blockedRebaseVariant = useMemo(() => {
    if (!blockingMergeRunBlockedRebase) {
      return "amber" as const
    }
    const assist = blockingConflictAssist
    if (
      assist?.active &&
      assist.state !== "timed_out" &&
      assist.state !== "unsupported"
    ) {
      return "amber" as const
    }
    return blockingMergeRunStatus === "resumable"
      ? "emerald" as const
      : "amber" as const
  }, [
    blockingConflictAssist,
    blockingMergeRunBlockedRebase,
    blockingMergeRunStatus,
  ])

  return (
    <Card
      data-node-card
      data-task-card-id={node.id}
      className={cn(
        "group",
        "relative overflow-visible",
        "py-0",
        "cursor-pointer transition-[background-color,box-shadow,width,height] duration-200 hover:bg-accent/40",
        agentComposerExpanded
          ? "z-[60] w-[480px] bg-card shadow-lg hover:bg-card"
          : "w-full bg-transparent",
        showGitActionGlow
          ? cn(
              "before:pointer-events-none before:absolute before:inset-0 before:rounded-lg before:p-[2px]",
              "before:opacity-90 before:bg-[length:250%_100%] before:[background-repeat:no-repeat]",
              "before:[--spread:22px] before:[--bg:linear-gradient(90deg,#0000_calc(50%_-_var(--spread)),var(--color-ring),#0000_calc(50%_+_var(--spread)))]",
              "before:[background-image:var(--bg)]",
              epicGraphIsFetching
                ? "before:[animation:rn-shimmer_2.4s_linear_infinite]"
                : "before:opacity-60",
              "before:[mask:linear-gradient(#000_0_0)_content-box,linear-gradient(#000_0_0)] before:[mask-composite:exclude]",
              "before:[-webkit-mask-composite:xor]",
            )
          : null,
        actionsMenuOpen ? "bg-accent/40" : null,
        task?.state === "done"
          ? "ring-green-950/80"
          : mergeReady
            ? "ring-emerald-400/50"
            : null,
        isSelected
          ? "ring-[3px] ring-ring"
          : isHighlighted
            ? "ring-1 ring-ring/60"
            : null,
      )}
      onClick={(e) => {
        const additive = e.metaKey || e.ctrlKey
        onSelect({ additive })
        if (!additive) {
          setAgentComposerExpanded((current) => !current)
        }
      }}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.currentTarget !== e.target) {
          return
        }
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault()
          onSelect({ additive: false })
          setAgentComposerExpanded((current) => !current)
        }
      }}
    >
      {linearIssueId ? (
        <div
          className={cn(
            "nodrag nopan absolute left-0 top-0 z-40 flex items-center gap-1 -translate-y-[calc(100%+8px)]",
          )}
          onPointerDown={(e) => e.stopPropagation()}
          onClick={(e) => e.stopPropagation()}
        >
          <Tooltip>
            <TooltipTrigger
              render={(tooltipTriggerProps) => (
                <button
                  {...tooltipTriggerProps}
                  type="button"
                  className={cn(
                    "group/linear inline-flex h-6 max-w-48 items-center overflow-hidden whitespace-nowrap rounded-full border-2 shadow-sm backdrop-blur focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/30",
                    "transition-colors duration-200",
                    actionsMenuOpen
                      ? "bg-accent/40"
                      : "bg-transparent group-hover:bg-accent/40 group-focus-within:bg-accent/40",
                    linearPillBorderClass,
                    linearObservedOld ? "border-opacity-70" : null,
                    linearStateMismatch ? "ring-1 ring-amber-300/25" : null,
                  )}
                  aria-label={linearPillLabel}
                  onClick={() => {
                    window.open(
                      `/v1/linear/issues/${linearIssueId}/open`,
                      "_blank",
                      "noreferrer",
                    )
                  }}
                >
                  <span className="relative inline-flex size-6 shrink-0 items-center justify-center">
                    <LinearIcon className="size-3.5 text-muted-foreground" />
                  </span>
                  <span className="min-w-0 truncate pr-2 font-mono text-[0.625rem] leading-none text-muted-foreground">
                    {linearPillLabel}
                  </span>
                </button>
              )}
            />
            <TooltipContent side="bottom" sideOffset={10}>
              <div className="space-y-1">
                <div className="text-xs">
                  Linear: {linearStateLabel(linearStateName, linearStateType)}
                </div>
                {linearStateMismatch ? (
                  <div className="text-xs text-muted-foreground">
                    Out of sync with local
                  </div>
                ) : null}
                {linearObservedAtLabel ? (
                  <div className="text-xs text-muted-foreground">
                    Last observed {linearObservedAtLabel}
                  </div>
                ) : null}
              </div>
            </TooltipContent>
          </Tooltip>
        </div>
      ) : null}
      {taskId !== null ? (
        <div
          className={cn(
            "nodrag nopan absolute right-0 top-2.5 z-40 translate-x-[calc(100%+8px)] transition-opacity",
            isSelected || actionsMenuOpen
              ? "opacity-100"
              : "opacity-0 group-hover:opacity-100 group-focus-within:opacity-100",
          )}
          onPointerDown={(e) => e.stopPropagation()}
          onClick={(e) => e.stopPropagation()}
        >
          <DropdownMenu onOpenChange={(open) => setActionsMenuOpen(open)}>
            <DropdownMenuTrigger
              render={(triggerProps) => (
                <Tooltip>
                  <TooltipTrigger
                    render={(tooltipTriggerProps) => (
                      <Button
                        {...tooltipTriggerProps}
                        {...triggerProps}
                        variant="ghost"
                        size="icon-sm"
                        aria-label="Task actions"
                        className={cn(
                          "rounded-full border border-border/60 bg-accent/40 shadow-sm backdrop-blur hover:bg-accent/60",
                          triggerProps.className,
                          tooltipTriggerProps.className,
                        )}
                      >
                        <EllipsisVertical className="size-4" />
                      </Button>
                    )}
                  />
                  <TooltipContent side="bottom" sideOffset={10}>
                    Task actions
                  </TooltipContent>
                </Tooltip>
              )}
            />
            <DropdownMenuContent align="end" side="bottom" sideOffset={10}>
              <Tooltip>
                <TooltipTrigger
                  render={(triggerProps) => (
                    <DropdownMenuItem
                      {...triggerProps}
                      disabled={
                        pendingMerge !== null ||
                        canResumeMerge ||
                        gitDisabledReason !== null
                      }
                      onClick={(e) => {
                        e.preventDefault()
                        e.stopPropagation()
                        void handleRestack({ scope: "descendants" })
                      }}
                    >
                      <GitBranch className="size-3.5" />
                      Restack
                    </DropdownMenuItem>
                  )}
                />
                <TooltipContent side="right" sideOffset={12} align="center">
                  {gitDisabledReason ??
                    "Rebase this branch and downstream branches to keep the stack intact."}
                </TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger
                  render={(triggerProps) => (
                    <DropdownMenuItem
                      {...triggerProps}
                      disabled={githubPrDisabledReason !== null}
                      onClick={(e) => {
                        e.preventDefault()
                        e.stopPropagation()
                        void handleOpenGithubPullRequest()
                      }}
                    >
                      <GithubIcon className="size-3.5" />
                      {pendingMerge === "githubPr"
                        ? "Opening PR…"
                        : "Push + Open PR"}
                    </DropdownMenuItem>
                  )}
                />
                <TooltipContent side="right" sideOffset={12} align="center">
                  {githubPrDisabledReason ??
                    "Push this branch to GitHub and open (or create) a PR."}
                </TooltipContent>
              </Tooltip>
              <DropdownMenuSeparator />
              <Tooltip>
                <TooltipTrigger
                  render={(triggerProps) => (
                    <DropdownMenuCheckboxItem
                      {...triggerProps}
                      checked={mergeReady}
                      disabled={
                        !canMerge ||
                        pendingMerge !== null ||
                        canResumeMerge ||
                        (!task?.branchName && !mergeReady)
                      }
                      closeOnClick={false}
                      onClick={(e) => {
                        e.stopPropagation()
                        triggerProps.onClick?.(e)
                      }}
                      onCheckedChange={(checked) =>
                        void handleToggleMergeReady(checked)
                      }
                    >
                      Ready to merge
                    </DropdownMenuCheckboxItem>
                  )}
                />
                <TooltipContent side="right" sideOffset={12} align="center">
                  Marks this task + unmerged ancestors ready to merge.
                </TooltipContent>
              </Tooltip>
              <DropdownMenuSeparator />
              {canResumeMerge ? (
                <>
                  <Tooltip>
                    <TooltipTrigger
                      render={(triggerProps) => (
                        <DropdownMenuItem
                          {...triggerProps}
                          disabled={
                            pendingMerge !== null || gitDisabledReason !== null
                          }
                          onClick={(e) => {
                            e.preventDefault()
                            e.stopPropagation()
                            void handleResumeMerge()
                          }}
                        >
                          <Play className="size-3.5" />
                          {mergeRunOperation === "restack"
                            ? "Resume restack"
                            : "Resume merge"}
                        </DropdownMenuItem>
                      )}
                    />
                    <TooltipContent side="right" sideOffset={12} align="center">
                      {gitDisabledReason ??
                        `Continue a previously-blocked ${
                          mergeRunOperation === "restack" ? "restack" : "merge"
                        } run after resolving conflicts.`}
                    </TooltipContent>
                  </Tooltip>
                  <DropdownMenuSeparator />
                </>
              ) : null}
              {canCancelMergeRun ? (
                <>
                  <DropdownMenuSub>
                    <DropdownMenuSubTrigger
                      disabled={pendingMerge !== null}
                      className="text-destructive focus:bg-destructive/10 focus:text-destructive data-open:bg-destructive/10 data-open:text-destructive"
                    >
                      <Square className="size-3.5" />
                      Cancel…
                    </DropdownMenuSubTrigger>
                    <DropdownMenuSubContent>
                      <Tooltip>
                        <TooltipTrigger
                          render={(triggerProps) => (
                            <DropdownMenuItem
                              {...triggerProps}
                              variant="destructive"
                              disabled={pendingMerge !== null}
                              onClick={(e) => {
                                e.preventDefault()
                                e.stopPropagation()
                                void handleCancelMergeRun({ abortGit: false })
                              }}
                            >
                              <Square className="size-3.5" />
                              Cancel run
                            </DropdownMenuItem>
                          )}
                        />
                        <TooltipContent
                          side="right"
                          sideOffset={12}
                          align="center"
                        >
                          Cancel the in-progress merge/restack run.
                        </TooltipContent>
                      </Tooltip>
                      {canCancelMergeRunAbortGit ? (
                        <Tooltip>
                          <TooltipTrigger
                            render={(triggerProps) => (
                              <DropdownMenuItem
                                {...triggerProps}
                                variant="destructive"
                                disabled={pendingMerge !== null}
                                onClick={(e) => {
                                  e.preventDefault()
                                  e.stopPropagation()
                                  void handleCancelMergeRun({ abortGit: true })
                                }}
                              >
                                <RotateCcw className="size-3.5" />
                                Cancel + abort git
                              </DropdownMenuItem>
                            )}
                          />
                          <TooltipContent
                            side="right"
                            sideOffset={12}
                            align="center"
                          >
                            Cancel the run and attempt to abort the git
                            operation in the blocked worktree (best-effort).
                          </TooltipContent>
                        </Tooltip>
                      ) : null}
                    </DropdownMenuSubContent>
                  </DropdownMenuSub>
                  <DropdownMenuSeparator />
                </>
              ) : null}
              <Tooltip>
                <TooltipTrigger
                  render={(triggerProps) => (
                    <DropdownMenuItem
                      {...triggerProps}
                      disabled={
                        !canMerge ||
                        !mergeReady ||
                        pendingMerge !== null ||
                        canResumeMerge ||
                        gitDisabledReason !== null
                      }
                      onClick={(e) => {
                        e.preventDefault()
                        e.stopPropagation()
                        void handleMerge({ cascade: false })
                      }}
                    >
                      <GitMerge className="size-3.5" />
                      Merge
                    </DropdownMenuItem>
                  )}
                />
                <TooltipContent side="right" sideOffset={12} align="center">
                  {gitDisabledReason ??
                    "Fast-forward merge the task + its unmerged ancestors into the epic base branch."}
                </TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger
                  render={(triggerProps) => (
                    <DropdownMenuItem
                      {...triggerProps}
                      disabled={
                        !canMerge ||
                        !mergeReady ||
                        pendingMerge !== null ||
                        canResumeMerge ||
                        gitDisabledReason !== null
                      }
                      onClick={(e) => {
                        e.preventDefault()
                        e.stopPropagation()
                        void handleMerge({ cascade: true })
                      }}
                    >
                      <Layers className="size-3.5" />
                      Merge and Restack
                    </DropdownMenuItem>
                  )}
                />
                <TooltipContent side="right" sideOffset={12} align="center">
                  {gitDisabledReason ??
                    "Rebase downstream branches to keep the stack intact, then fast-forward merge the task + its unmerged ancestors into the epic base branch."}
                </TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger
                  render={(triggerProps) => (
                    <DropdownMenuItem
                      {...triggerProps}
                      disabled={
                        !canMerge ||
                        !mergeReady ||
                        pendingMerge !== null ||
                        canResumeMerge ||
                        gitDisabledReason !== null
                      }
                      onClick={(e) => {
                        e.preventDefault()
                        e.stopPropagation()
                        void handleMergeThenRestack()
                      }}
                    >
                      <Layers className="size-3.5" />
                      Merge then Restack
                    </DropdownMenuItem>
                  )}
                />
                <TooltipContent side="right" sideOffset={12} align="center">
                  {gitDisabledReason ??
                    "Fast-forward merge the task + its unmerged ancestors as in [Merge], then rebase downstream branches to keep the stack intact."}
                </TooltipContent>
              </Tooltip>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      ) : null}
      <CardContent className="flex h-full flex-col gap-2 p-3">
        <div className="flex min-w-0 items-center justify-between gap-2">
          <div
            className={cn(
              "min-w-0 truncate font-mono text-xs leading-none text-muted-foreground",
              branchLabelProvisional ? "opacity-60" : null,
            )}
          >
            {branchLabel}
          </div>
          <div className="flex items-center gap-1">
            {quickActions ? (
              <div
                className={cn(
                  "flex items-center gap-1 transition-opacity",
                  isSelected || actionsMenuOpen
                    ? "opacity-100"
                    : "opacity-0 group-hover:opacity-100 group-focus-within:opacity-100",
                )}
              >
                {quickActions}
              </div>
            ) : null}
            <div className="inline-flex size-5 items-center justify-center">
              <Tooltip>
                <TooltipTrigger
                  render={(triggerProps) => (
                    <span
                      {...triggerProps}
                      className={cn("inline-flex", triggerProps.className)}
                      aria-label={tooltip}
                    >
                      <AgentStatusIcon
                        status={agentSession?.status ?? null}
                        continuable={agentContinuable}
                        taskState={task?.state}
                        className="size-3"
                        label={tooltip}
                      />
                    </span>
                  )}
                />
                <TooltipContent side="bottom" sideOffset={10}>
                  {tooltip}
                </TooltipContent>
              </Tooltip>
            </div>
          </div>
        </div>

        <div className="min-w-0">
          <div
            className={cn(
              "text-sm font-medium leading-tight",
              agentMessagePreview ? "line-clamp-1" : null,
            )}
          >
            {task?.title ?? "—"}
          </div>
        </div>
        {agentMessagePreview || showAgentPreviewShimmer ? (
          <div
            aria-expanded={agentComposerExpanded}
            className={cn(
              "nodrag nopan",
              "w-full",
              "mt-2 flex min-w-0 items-start gap-1.5 text-left leading-snug",
              capturedFinalOutputText
                ? "text-[11px] text-muted-foreground/75"
                : "text-xs text-muted-foreground/80",
              "transition-colors hover:text-muted-foreground",
              agentPreviewDimmed ? "opacity-60" : null,
            )}
          >
            {capturedFinalOutputText ? (
              <MessageSquareText
                className="mt-0.5 size-3.5 shrink-0 opacity-70"
                aria-hidden="true"
              />
            ) : (
              <Ghost className="mt-0.5 size-3.5 shrink-0 opacity-70" />
            )}
            <div className="min-w-0 flex-1">
              {agentPreviewTitle ? (
                <div
                  className={cn(
                    "line-clamp-1 text-foreground/80",
                    capturedFinalOutputText ? "font-medium" : "font-semibold",
                  )}
                >
                  {showAgentPreviewShimmer ? (
                    <Shimmer as="span" className="inline" duration={3.5}>
                      {agentPreviewTitle.slice(0, 200)}
                    </Shimmer>
                  ) : (
                    <MarkdownInline content={agentPreviewTitle} />
                  )}
                </div>
              ) : null}
              {agentPreviewBody || !agentPreviewTitle ? (
                <div className="line-clamp-1">
                  {showAgentPreviewShimmer && !agentPreviewTitle ? (
                    <Shimmer as="span" className="inline" duration={3.5}>
                      {(
                        agentPreviewBody ??
                        agentMessagePreview ??
                        "Thinking…"
                      ).slice(0, 200)}
                    </Shimmer>
                  ) : agentPreviewBody ? (
                    <MarkdownInline content={agentPreviewBody} />
                  ) : !agentPreviewTitle && agentMessagePreview ? (
                    <MarkdownInline content={agentMessagePreview} />
                  ) : null}
                </div>
              ) : null}
            </div>
          </div>
        ) : null}
        <div
          className={cn(
            "nodrag nopan grid transition-[grid-template-rows,opacity] duration-200",
            agentComposerExpanded
              ? "grid-rows-[1fr] opacity-100"
              : "grid-rows-[0fr] opacity-0 pointer-events-none",
          )}
          onPointerDown={(e) => e.stopPropagation()}
          onClick={(e) => e.stopPropagation()}
        >
          <div className="overflow-hidden pt-2">
            {capturedFinalOutputText ? (
              <div className="mb-2 rounded-md bg-muted/30 p-2 text-[11px] text-foreground/85 shadow-sm">
                <div className="mb-1 flex items-center justify-between gap-2">
                  <div className="inline-flex items-center gap-1.5 text-[11px] font-medium text-muted-foreground">
                    <MessageSquareText className="size-3" aria-hidden="true" />
                    Final output
                  </div>
                  <Button
                    variant="ghost"
                    size="xs"
                    onClick={(e) => {
                      e.preventDefault()
                      e.stopPropagation()
                      void copyToClipboard(capturedFinalOutputText)
                    }}
                  >
                    Copy
                  </Button>
                </div>
                <div className="max-h-52 overflow-auto pr-1">
                  <Markdown
                    content={capturedFinalOutputText}
                    omitFirstHeading
                    omitMetadataSection
                  />
                </div>
              </div>
            ) : null}
            <div className="relative rounded-md border border-border/60 bg-background shadow-sm focus-within:ring-2 focus-within:ring-ring/30">
              <Textarea
                ref={agentComposerTextareaRef}
                value={agentComposerDraft}
                onChange={(event) => {
                  setAgentComposerDraft(event.target.value)
                  if (agentComposerError) {
                    setAgentComposerError(null)
                  }
                }}
                placeholder="Write a message…"
                rows={3}
                className="min-h-20 w-full resize-none border-0 bg-transparent pr-10 shadow-none focus-visible:ring-0"
                onKeyDown={(event) => {
                  if (
                    (event.metaKey || event.ctrlKey) &&
                    event.key === "Enter" &&
                    !agentComposerSendDisabledReason
                  ) {
                    event.preventDefault()
                    event.stopPropagation()
                    if (agentLooksBusy) {
                      setAgentComposerConfirmKind(
                        composerInterfaceMode === "structured"
                          ? "structured_turn_in_progress"
                          : "interactive_busy",
                      )
                      setAgentComposerConfirmOpen(true)
                      return
                    }
                    void submitAgentMessage()
                  }
                }}
              />
              <div className="absolute bottom-2 right-2 flex items-center gap-2">
                {agentComposerPendingReason ? (
                  <div className="select-none text-[11px] text-muted-foreground">
                    Sending
                    <span className="inline-block w-3 animate-pulse">…</span>
                  </div>
                ) : null}
                <Button
                  variant="secondary"
                  size="icon-sm"
                  className="rounded-full"
                  disabledReason={agentComposerSendDisabledReason}
                  onClick={(event) => {
                    event.preventDefault()
                    event.stopPropagation()
                    if (agentLooksBusy) {
                      setAgentComposerConfirmKind(
                        composerInterfaceMode === "structured"
                          ? "structured_turn_in_progress"
                          : "interactive_busy",
                      )
                      setAgentComposerConfirmOpen(true)
                      return
                    }
                    void submitAgentMessage()
                  }}
                >
                  <ArrowUp className="size-3" />
                </Button>
              </div>
            </div>
            {agentComposerError ? (
              <div className="mt-1 text-[11px] text-destructive">
                {agentComposerError}
              </div>
            ) : null}
          </div>
        </div>
        <div className="mt-auto flex items-end justify-between gap-2 pt-2">
          {agentSession ? (
            <div className="flex flex-wrap items-end justify-start gap-2">
              <ResourceBadge
                label={agentSession.agentLabel}
                value={agentKindLabel}
                extendBackground
              />
            </div>
          ) : (
            <div />
          )}
          {gitAttentionIcon ? (
            <Tooltip>
              <TooltipTrigger
                render={(triggerProps) => (
                  <span
                    {...triggerProps}
                    className={cn(
                      "inline-flex items-center gap-1",
                      triggerProps.className,
                    )}
                    aria-label="Git status attention"
                  >
                    {syncAttentionIcon}
                    {mergeAttentionIcon}
                  </span>
                )}
              />
              <TooltipContent side="left" sideOffset={12} align="center">
                <div className="whitespace-pre-line">
                  {gitAttentionTooltipLines.join("\n")}
                </div>
              </TooltipContent>
            </Tooltip>
          ) : null}
        </div>
      </CardContent>
      {actionError ||
      showResumableMergeCallout ||
      blockingMergeRunBlockedRebase ||
      mergeReadySpineNotice ||
      blockedRebaseCompletionNotice ? (
        <div
          className="nodrag nopan absolute left-0 top-full z-50 mt-3 w-[min(100%,480px)] space-y-2"
          onPointerDown={(e) => e.stopPropagation()}
          onClick={(e) => e.stopPropagation()}
        >
          {mergeReadySpineNotice ? (
            <MergeReadySpineWarningCallout
              notice={mergeReadySpineNotice}
              onDismiss={() => setMergeReadySpineNotice(null)}
            />
          ) : null}
          {actionError ? (
            <ActionErrorCallout
              error={actionError}
              mergeRunBlockedRebase={mergeRunBlockedRebase}
              rebaseAttachCommand={rebaseAttachCommand}
              rebaseRemediationMessage={rebaseRemediationMessage}
              onDismiss={clearActionError}
            />
          ) : null}

          {blockingMergeRunBlockedRebase ? (
            <BlockedRebaseCallout
              variant={blockedRebaseVariant}
              badgeLabel={blockedRebaseBadgeLabel ?? "Rebase blocked"}
              summary={
                blockedRebaseSummary ?? "Resolve conflicts, then resume."
              }
              detail={blockingConflictAssist?.detail ?? null}
              status={
                blockingMergeRunStatus === "resumable" ? "resumable" : "blocked"
              }
              assistActive={blockingConflictAssist?.active === true}
              operation={
                blockingMergeRun?.operation === "restack" ? "restack" : "merge"
              }
              attachCommand={blockingAttachCommand}
              remediationMessage={blockingRemediationMessage}
              onResume={() => void handleResumeBlockingMerge()}
            />
          ) : null}

          {showResumableMergeCallout ? (
            <ResumableMergeCallout
              variant={resumableMergeVariant}
              detail={resumableMergeDetail}
              operation={mergeRunOperation}
              disabledReason={
                pendingMerge !== null ? "Action in progress" : gitDisabledReason
              }
              onResume={() => void handleResumeMerge()}
            />
          ) : null}

          {blockedRebaseCompletionNotice ? (
            <SuccessToastCallout
              message={blockedRebaseCompletionNotice}
              onDismiss={() => setBlockedRebaseCompletionNotice(null)}
            />
          ) : null}
          {githubPrNotice ? (
            <SuccessToastCallout
              message={githubPrNotice}
              onDismiss={() => setGithubPrNotice(null)}
            />
          ) : null}
        </div>
      ) : null}
      <ProceedAnywayDialog
        open={allowRunningPrompt !== null}
        pending={allowRunningConfirming}
        actionLabel={allowRunningActionLabel ?? "merge"}
        detail={allowRunningPrompt?.detail}
        onOpenChange={(open) => {
          if (open) {
            return
          }
          setAllowRunningPrompt(null)
        }}
        onProceed={() => void confirmAllowRunning()}
      />
      <AgentMessageConfirmDialog
        open={agentComposerConfirmOpen}
        kind={agentComposerConfirmKind}
        canInterrupt={agentCanInterrupt}
        pendingReason={agentComposerPendingReason}
        onOpenChange={setAgentComposerConfirmOpen}
        onInterruptAndSend={() => {
          setAgentComposerConfirmOpen(false)
          void submitAgentMessage({ onConflict: "interrupt_turn" })
        }}
        onStopAndSend={() => {
          setAgentComposerConfirmOpen(false)
          void submitAgentMessage({ onConflict: "stop_session_and_start_new" })
        }}
        onSendAnyway={() => {
          setAgentComposerConfirmOpen(false)
          void submitAgentMessage({ onConflict: "fail" })
        }}
      />
      <MergeReadySpineConfirmDialog
        open={mergeReadySpineConfirm.dialog.open}
        taskCount={mergeReadySpineConfirm.dialog.taskCount}
        dontAskAgain={mergeReadySpineConfirm.dialog.dontAskAgain}
        pendingReason={pendingMerge !== null ? "Action in progress" : null}
        onOpenChange={mergeReadySpineConfirm.dialog.setOpen}
        onDontAskAgainChange={mergeReadySpineConfirm.dialog.setDontAskAgain}
        onConfirm={mergeReadySpineConfirm.dialog.confirm}
      />
    </Card>
  )
}
