import { useEffect, useMemo, useRef, useState } from "react"
import { createPortal } from "react-dom"
import { Markdown } from "@/components/markdown"
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Alert } from "@/components/ui/alert"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { SlidePanel } from "@/components/ui/slide-panel"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { ApiHttpError, fetchTaskAgentLogs } from "@/api"
import {
  useRestartTaskAgentMutation,
  useCancelMergeRunMutation,
  useResumeMergeRunMutation,
  useSetTaskMergeReadyMutation,
  useStartTaskAgentMutation,
  useStopTaskAgentMutation,
} from "@/api/mutations"
import {
  type AgentSession,
  type GraphNode,
  type MergeRun,
  type Task,
} from "@/lib/graph-utils"
import { useOrchestrationDefaults } from "@/hooks/useOrchestrationDefaults"
import { useMergeReadySpineConfirm } from "@/hooks/useMergeReadySpineConfirm"
import { copyToClipboard } from "@/lib/clipboard"
import { getRebaseRemediation } from "@/lib/merge-remediation"
import {
  isRunningAgentsConflict,
  runningAgentsSummary,
} from "@/lib/runningAgentsConflict"
import { cn } from "@/lib/utils"
import { AgentStatusBadge } from "@/components/agents/AgentStatusBadge"
import { AgentKindSelect } from "@/components/agents/AgentKindSelect"
import { FloatingActions } from "@/components/ui/floating-actions"
import { ProceedAnywayDialog } from "@/components/ui/proceed-anyway-dialog"
import { ResourceBadge } from "@/components/ui/resource-badge"
import { MergeReadySpineConfirmDialog } from "@/components/merge-ready/MergeReadySpineConfirmDialog"
import { Textarea } from "@/components/ui/textarea"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  inferAgentKindFromCommand,
  inferStructuredAgentFromCommand,
  labelForAgentKind,
} from "@/lib/agent-kind"
import {
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  Loader2,
  MessageSquareText,
  Play,
  RotateCcw,
  Square,
  Terminal,
} from "lucide-react"
import { Switch } from "@/components/ui/switch"

type EdgeSelection = {
  id: string
  fromNodeId: number
  toNodeId: number
  fromLabel: string
  toLabel: string
  commitCount?: number | null
  baseSha?: string | null
  headSha?: string | null
}

function stripAnsi(text: string) {
  // CSI sequences (colors, cursor moves, etc.)
  // eslint-disable-next-line no-control-regex
  const withoutCsi = text.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "")
  // OSC sequences (titles, hyperlinks, etc.)
  // eslint-disable-next-line no-control-regex
  return withoutCsi.replace(/\x1b\][^\x07]*(?:\x07|\x1b\\)/g, "")
}

interface DetailsPanelProps {
  open: boolean
  task: Task | null
  node?: GraphNode | null
  tasksById: Map<number, Task>
  mergeRun?: MergeRun | null
  agentSession?: AgentSession | null
  edge?: EdgeSelection | null
}

function MergeRunDetails({
  mergeRun,
  node,
  panelOpen,
}: {
  mergeRun: MergeRun
  node: GraphNode | null | undefined
  panelOpen: boolean
}) {
  const resumeMerge = useResumeMergeRunMutation()
  const cancelMerge = useCancelMergeRunMutation()
  const [calloutExpanded, setCalloutExpanded] = useState(false)
  const [detailsExpanded, setDetailsExpanded] = useState(false)
  const [resumeError, setResumeError] = useState<string | null>(null)
  const [resumePending, setResumePending] = useState(false)
  const [resumePromptOpen, setResumePromptOpen] = useState(false)
  const [resumePromptDetail, setResumePromptDetail] = useState<string | null>(
    null,
  )
  const [cancelError, setCancelError] = useState<string | null>(null)
  const [cancelPending, setCancelPending] = useState(false)
  const [cancelPromptOpen, setCancelPromptOpen] = useState(false)
  const [cancelAbortGit, setCancelAbortGit] = useState(false)
  const remediation = getRebaseRemediation(mergeRun, node?.branchName ?? null)
  const conflictAssist = mergeRun.conflictAssist ?? null

  const statusLabel = mergeRun.status.replace(/^\w/, (c) => c.toUpperCase())
  const operationLabel = mergeRun.operation === "restack" ? "Restack" : "Merge"
  const blockedError = mergeRun.blockedError?.trim() || null
  const blockedBranch = mergeRun.blockedBranchName ?? null
  const blockedWorktree = mergeRun.blockedWorktreePath ?? null
  const blockedStepKind = mergeRun.blockedStepKind ?? null

  const showCallout =
    mergeRun.status === "blocked" ||
    mergeRun.status === "resumable" ||
    mergeRun.status === "failed"

  const panelOpenPrevRef = useRef(panelOpen)
  useEffect(() => {
    const wasOpen = panelOpenPrevRef.current
    panelOpenPrevRef.current = panelOpen

    if (!panelOpen || wasOpen) {
      return
    }

    if (!showCallout) {
      return
    }

    // Auto-expand alert callouts when the drawer opens, but don't fight manual collapse after.
    setCalloutExpanded(true)
  }, [panelOpen, showCallout])

  const calloutTone =
    mergeRun.status === "blocked"
      ? "amber"
      : mergeRun.status === "failed"
        ? "destructive"
        : mergeRun.status === "resumable"
          ? "emerald"
          : "default"

  const calloutTitle =
    // When a descendant rebase conflicts, the merge run is blocked in the restack portion.
    // (The spine merge may or may not have completed depending on the merge mode.)
    mergeRun.status === "blocked"
      ? operationLabel === "Merge" &&
        (mergeRun as { blockedOnSpine?: boolean | null }).blockedOnSpine ===
          false
        ? "Restack blocked (conflicts)"
        : `${operationLabel} blocked (conflicts)`
      : mergeRun.status === "resumable"
        ? `${operationLabel} ready to resume`
        : mergeRun.status === "failed"
          ? `${operationLabel} failed`
          : null

  const calloutIcon =
    mergeRun.status === "resumable" ? (
      <Play className="size-3.5 text-emerald-400" />
    ) : mergeRun.status === "blocked" ? (
      <AlertTriangle className="size-3.5 text-amber-300/80" />
    ) : (
      <AlertTriangle className="size-3.5 text-destructive" />
    )

  const attachCommand = remediation?.attachCommand ?? null
  const agentNote = remediation?.message ?? null
  const assistNoteSent = conflictAssist?.messageSentAt ?? null
  const assistTimedOut = conflictAssist?.state === "timed_out"
  const assistUnsupported = conflictAssist?.state === "unsupported"
  const assistResumed = conflictAssist?.state === "resumed"
  const assistAgentTurnComplete =
    conflictAssist?.state === "waiting_for_repo_clean" ||
    conflictAssist?.state === "ready_to_resume" ||
    conflictAssist?.state === "resumed"
  const assistRepoClean =
    conflictAssist?.state === "ready_to_resume" ||
    conflictAssist?.state === "resumed"

  const statusBadgeVariant =
    mergeRun.status === "blocked"
      ? "amber"
      : mergeRun.status === "failed"
        ? "destructive"
        : mergeRun.status === "resumable"
          ? "emerald"
          : "default"

  async function handleResumeMerge(allowRunning: boolean) {
    setResumePending(true)
    setResumeError(null)
    try {
      await resumeMerge.mutateAsync({
        epicId: mergeRun.epicId,
        runId: mergeRun.runId,
        request: { allowRunning },
      })
    } catch (e) {
      if (e instanceof ApiHttpError && e.status === 409 && !allowRunning) {
        if (isRunningAgentsConflict(e)) {
          setResumePromptDetail(runningAgentsSummary(e))
          setResumePromptOpen(true)
          return
        }
      }
      setResumeError(e instanceof Error ? e.message : String(e))
    } finally {
      setResumePending(false)
    }
  }

  async function handleCancelMerge(abortGit: boolean) {
    setCancelPending(true)
    setCancelError(null)
    try {
      await cancelMerge.mutateAsync({
        epicId: mergeRun.epicId,
        runId: mergeRun.runId,
        request: { abortGit },
      })
    } catch (e) {
      setCancelError(e instanceof Error ? e.message : String(e))
    } finally {
      setCancelPending(false)
    }
  }

  return (
    <div className="grid min-w-0 gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant={statusBadgeVariant}>{statusLabel}</Badge>
        <Badge className="capitalize">{mergeRun.scope}</Badge>
      </div>

      {showCallout && calloutTitle ? (
        <Collapsible
          open={calloutExpanded}
          onOpenChange={setCalloutExpanded}
          className="w-full min-w-0"
        >
          <Alert variant={calloutTone} className="w-full min-w-0 gap-2">
            <div className="flex items-start justify-between gap-3">
              <div className="flex min-w-0 items-center gap-2">
                {calloutIcon}
                <div className="truncate text-xs font-medium text-foreground">
                  {calloutTitle}
                </div>
              </div>
              <div className="flex items-center gap-2">
                {mergeRun.status === "resumable" ? (
                  <Button
                    variant="outline"
                    size="xs"
                    className="border-emerald-400/35 text-emerald-100 hover:bg-emerald-400/10 hover:text-emerald-50"
                    disabledReason={resumePending ? "Action in progress" : null}
                    onClick={() => void handleResumeMerge(false)}
                  >
                    {resumePending ? (
                      <Loader2 className="animate-spin" />
                    ) : (
                      <Play />
                    )}
                    {operationLabel === "Restack"
                      ? "Resume restack"
                      : "Resume merge"}
                  </Button>
                ) : null}

                <DropdownMenu>
                  <DropdownMenuTrigger
                    render={(triggerProps) => (
                      <Button {...triggerProps} variant="ghost" size="xs">
                        {cancelPending ? (
                          <Loader2 className="size-3 animate-spin" />
                        ) : (
                          <Square className="size-3" />
                        )}
                        Cancel
                        <ChevronDown className="size-3" />
                      </Button>
                    )}
                  />
                  <DropdownMenuContent align="end" sideOffset={10}>
                    <DropdownMenuItem
                      disabled={cancelPending}
                      onClick={(e) => {
                        e.preventDefault()
                        e.stopPropagation()
                        setCancelAbortGit(false)
                        setCancelPromptOpen(true)
                      }}
                    >
                      <Square className="size-3.5" />
                      Cancel run
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      variant="destructive"
                      disabled={cancelPending || !blockedWorktree}
                      onClick={(e) => {
                        e.preventDefault()
                        e.stopPropagation()
                        setCancelAbortGit(true)
                        setCancelPromptOpen(true)
                      }}
                    >
                      <Square className="size-3.5" />
                      Cancel + abort git
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>

                <CollapsibleTrigger
                  render={(triggerProps) => (
                    <Tooltip>
                      <TooltipTrigger
                        render={(tooltipTriggerProps) => (
                          <Button
                            {...tooltipTriggerProps}
                            {...triggerProps}
                            variant="ghost"
                            size="icon-xs"
                            aria-label={
                              calloutExpanded
                                ? "Collapse callout"
                                : "Expand callout"
                            }
                            className={cn(
                              triggerProps.className,
                              tooltipTriggerProps.className,
                            )}
                          >
                            {calloutExpanded ? (
                              <ChevronUp className="size-3" />
                            ) : (
                              <ChevronDown className="size-3" />
                            )}
                          </Button>
                        )}
                      />
                      <TooltipContent side="bottom" sideOffset={10}>
                        {calloutExpanded
                          ? "Collapse callout"
                          : "Expand callout"}
                      </TooltipContent>
                    </Tooltip>
                  )}
                />
              </div>
            </div>

            {mergeRun.status === "blocked" && remediation ? (
              <div className="mt-2 flex flex-wrap items-center gap-2">
                <Button
                  variant="ghost"
                  size="xs"
                  disabledReason={
                    attachCommand
                      ? null
                      : "No task id recorded for this blocked step."
                  }
                  onClick={() => {
                    if (!attachCommand) {
                      return
                    }
                    void copyToClipboard(attachCommand)
                  }}
                >
                  <Terminal className="size-3" />
                  Copy attach
                </Button>
                <Button
                  variant="ghost"
                  size="xs"
                  disabledReason={
                    agentNote ? null : "No remediation message available."
                  }
                  onClick={() => {
                    if (!agentNote) {
                      return
                    }
                    void copyToClipboard(agentNote)
                  }}
                >
                  <MessageSquareText className="size-3" />
                  Copy agent note
                </Button>
              </div>
            ) : null}

            {conflictAssist ? (
              <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                <span className="font-medium text-foreground/80">
                  Conflict assist
                </span>
                <Badge variant={assistUnsupported ? "destructive" : "default"}>
                  {assistTimedOut
                    ? "Timed out"
                    : assistUnsupported
                      ? "Unavailable"
                      : assistResumed
                        ? "Resumed"
                        : conflictAssist.active
                          ? "Active"
                          : "Inactive"}
                </Badge>
                <Badge variant={assistNoteSent ? "emerald" : "amber"}>
                  {assistNoteSent ? "Note delivered" : "Note pending"}
                </Badge>
                <Badge variant={assistAgentTurnComplete ? "emerald" : "amber"}>
                  {assistAgentTurnComplete ? "Agent done" : "Waiting for agent"}
                </Badge>
                <Badge variant={assistRepoClean ? "emerald" : "amber"}>
                  {assistRepoClean ? "Repo clean" : "Waiting for repo"}
                </Badge>
                {conflictAssist.detail ? (
                  <div className="w-full text-xs text-muted-foreground">
                    {conflictAssist.detail}
                  </div>
                ) : null}
              </div>
            ) : null}

            <CollapsibleContent className="mt-2 w-full min-w-0">
              {blockedStepKind ? (
                <div className="text-xs text-muted-foreground">
                  Step: <span className="font-mono">{blockedStepKind}</span>
                </div>
              ) : null}

              {blockedBranch ? (
                <div className="mt-1 min-w-0 text-xs text-muted-foreground">
                  Branch:{" "}
                  <span className="break-all font-mono text-foreground/80">
                    {blockedBranch}
                  </span>
                </div>
              ) : null}

              {blockedWorktree ? (
                <div className="mt-1 min-w-0 text-xs text-muted-foreground">
                  Worktree:{" "}
                  <span className="break-all font-mono text-foreground/80">
                    {blockedWorktree}
                  </span>
                </div>
              ) : null}

              {blockedError ? (
                <div className="mt-2">
                  <div className="flex items-center justify-between gap-2">
                    <Button
                      variant="ghost"
                      size="xs"
                      onClick={() => setDetailsExpanded((open) => !open)}
                    >
                      {detailsExpanded ? "Hide" : "Show"} details
                    </Button>
                    <Button
                      variant="ghost"
                      size="xs"
                      onClick={() => void copyToClipboard(blockedError)}
                    >
                      Copy
                    </Button>
                  </div>
                  {detailsExpanded ? (
                    <div className="mt-2 max-h-48 overflow-auto rounded-md bg-background/40 px-2 py-1.5 font-mono text-[0.625rem] text-foreground/80">
                      <div className="whitespace-pre-wrap break-all">
                        {blockedError}
                      </div>
                    </div>
                  ) : null}
                </div>
              ) : null}
            </CollapsibleContent>
          </Alert>
        </Collapsible>
      ) : null}

      {resumeError ? (
        <pre className="whitespace-pre-wrap text-xs text-destructive">
          {resumeError}
        </pre>
      ) : null}

      {cancelError ? (
        <pre className="whitespace-pre-wrap text-xs text-destructive">
          {cancelError}
        </pre>
      ) : null}

      <ProceedAnywayDialog
        open={resumePromptOpen}
        pending={resumePending}
        actionLabel="resume"
        detail={resumePromptDetail}
        onOpenChange={(open) => {
          if (!open) {
            setResumePromptDetail(null)
          }
          setResumePromptOpen(open)
        }}
        onProceed={() => {
          setResumePromptOpen(false)
          void handleResumeMerge(true)
        }}
      />

      <ProceedAnywayDialog
        open={cancelPromptOpen}
        pending={cancelPending}
        actionLabel={cancelAbortGit ? "cancel + abort git" : "cancel"}
        detail={
          cancelAbortGit
            ? "This will cancel the run and attempt to abort the in-progress git operation in the blocked worktree (best-effort)."
            : "This will cancel the run. It will not automatically revert any in-progress git changes."
        }
        onOpenChange={setCancelPromptOpen}
        onProceed={() => {
          setCancelPromptOpen(false)
          void handleCancelMerge(cancelAbortGit)
        }}
      />
    </div>
  )
}

function AgentActions({
  task,
  tasksById,
  agentSession,
  floatingActionsPortalId,
}: {
  task: Task
  tasksById: Map<number, Task>
  agentSession: AgentSession | null
  floatingActionsPortalId?: string
}) {
  const startAgent = useStartTaskAgentMutation()
  const stopAgent = useStopTaskAgentMutation()
  const restartAgent = useRestartTaskAgentMutation()
  const setMergeReadyMutation = useSetTaskMergeReadyMutation()

  const { defaults: orchestrationDefaults } = useOrchestrationDefaults()
  const [pending, setPending] = useState<"start" | "stop" | "restart" | null>(
    null,
  )
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [logsOpen, setLogsOpen] = useState(false)
  const [logsPending, setLogsPending] = useState(false)
  const [logsError, setLogsError] = useState<string | null>(null)
  const [logsText, setLogsText] = useState<string | null>(null)
  const [logsPath, setLogsPath] = useState<string | null>(null)
  const [logsTruncated, setLogsTruncated] = useState<boolean>(false)
  const [oneTimePreludeOpen, setOneTimePreludeOpen] = useState(false)
  const [oneTimePrelude, setOneTimePrelude] = useState("")
  const mergeReady = Boolean(task.mergeReadyAt)
  const mergeReadySpineConfirm = useMergeReadySpineConfirm({
    resetKey: task.id,
  })

  useEffect(() => {
    setPending(null)
    setError(null)
    setNotice(null)
    setLogsOpen(false)
    setLogsPending(false)
    setLogsError(null)
    setLogsText(null)
    setLogsPath(null)
    setLogsTruncated(false)
    setOneTimePreludeOpen(false)
    setOneTimePrelude("")
  }, [task.id])

  useEffect(() => {
    if (!notice) {
      return
    }
    const id = window.setTimeout(() => setNotice(null), 2_000)
    return () => window.clearTimeout(id)
  }, [notice])

  const taskId = task.id
  const hasAgent = agentSession !== null
  const agentLabel = useMemo(
    () => agentSession?.agentLabel ?? "No agent",
    [agentSession?.agentLabel],
  )
  const statusLabel = agentSession?.status ?? null
  const agentContinuable =
    statusLabel === "stopped" &&
    agentSession !== null &&
    agentSession.agentCapabilities.canResumeById &&
    agentSession.externalSessionRef.type !== "none"
  const resolvedAgentKindLabel = agentSession
    ? labelForAgentKind(agentSession.agentKind)
    : null
  const agentKindBadgeValue = useMemo(() => {
    if (!agentSession) {
      return null
    }
    const resolved = labelForAgentKind(agentSession.agentKind)
    if (agentSession.agentKindSelection === "auto") {
      return `Auto→${resolved}`
    }
    return resolved
  }, [agentSession])
  const isRunning = statusLabel === "running" || statusLabel === "blocked"
  const canRestart = hasAgent

  const configuredHarnessCommand =
    orchestrationDefaults?.harness.command?.trim() ?? ""
  const configuredAgentKind = orchestrationDefaults?.harness.agentKind ?? "auto"
  const agentArgv = agentSession?.resolvedLaunchConfiguration?.argv ?? null
  const harnessCommand =
    agentArgv && agentArgv.length > 0
      ? agentArgv.join(" ")
      : configuredHarnessCommand

  const harnessEditable =
    !isRunning && (statusLabel === "stopped" || statusLabel === "error")
  const [harnessDraft, setHarnessDraft] = useState("")
  const [agentKindDraft, setAgentKindDraft] =
    useState<"auto" | "generic" | "codex" | "claude_code">("auto")
  useEffect(() => {
    setHarnessDraft(harnessCommand)
    setAgentKindDraft(agentSession?.agentKindSelection ?? configuredAgentKind)
  }, [
    harnessCommand,
    task.id,
    agentSession?.id,
    agentSession?.agentKindSelection,
    configuredAgentKind,
  ])
  const harnessDraftTrimmed = harnessDraft.trim()
  const harnessDirty =
    harnessEditable &&
    harnessDraftTrimmed !== harnessCommand.trim() &&
    harnessDraftTrimmed.length > 0

  const inferredKindFromCommand = useMemo(() => {
    const effectiveCommand =
      harnessEditable && harnessDirty ? harnessDraftTrimmed : harnessCommand
    if (!effectiveCommand.trim()) {
      return null
    }
    return inferAgentKindFromCommand(effectiveCommand)
  }, [harnessCommand, harnessDirty, harnessDraftTrimmed, harnessEditable])

  const effectiveHarnessForNextRun = useMemo(() => {
    if (!hasAgent) {
      return configuredHarnessCommand
    }
    if (harnessDirty) {
      return harnessDraftTrimmed
    }
    return harnessCommand
  }, [
    configuredHarnessCommand,
    hasAgent,
    harnessCommand,
    harnessDirty,
    harnessDraftTrimmed,
  ])
  const structuredAgentForNextRun = useMemo(() => {
    return inferStructuredAgentFromCommand(effectiveHarnessForNextRun)
  }, [effectiveHarnessForNextRun])
  const structuredModeForNextRun = structuredAgentForNextRun !== null
  const oneTimePreludeValue = oneTimePrelude.trim()

  async function handleStart() {
    setPending("start")
    setError(null)
    try {
      const response = await startAgent.mutateAsync({
        epicId: task.epicId,
        taskId,
        request: {
          harness: configuredHarnessCommand,
          agentKind: agentKindDraft,
          detach: orchestrationDefaults?.harness.detach ?? true,
          prelude: oneTimePreludeValue ? oneTimePreludeValue : null,
        },
      })
      const warnings = response.warnings ?? []
      if (warnings.length > 0) {
        if (!response.started || response.agentStatus === "error") {
          setError(warnings.join("\n"))
        } else {
          setNotice(warnings[0])
        }
      }
      setOneTimePrelude("")
      setOneTimePreludeOpen(false)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setPending(null)
    }
  }

  async function handleStop() {
    setPending("stop")
    setError(null)
    try {
      await stopAgent.mutateAsync({ epicId: task.epicId, taskId })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setPending(null)
    }
  }

  async function handleRestart() {
    setPending("restart")
    setError(null)
    try {
      const response = await restartAgent.mutateAsync({
        epicId: task.epicId,
        taskId,
        request: {
          harness: harnessDirty ? harnessDraftTrimmed : null,
          agentKind: agentKindDraft,
          detach: orchestrationDefaults?.harness.detach ?? true,
          prelude: oneTimePreludeValue ? oneTimePreludeValue : null,
        },
      })
      const warnings = response.warnings ?? []
      if (warnings.length > 0) {
        if (!response.started || response.agentStatus === "error") {
          setError(warnings.join("\n"))
        } else {
          setNotice(warnings[0])
        }
      }
      setOneTimePrelude("")
      setOneTimePreludeOpen(false)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setPending(null)
    }
  }

  async function handleCopy(text: string, label = "Copied") {
    try {
      await copyToClipboard(text)
      setNotice(label)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  async function commitMergeReady(next: boolean, scope?: "task" | "spine") {
    setError(null)
    try {
      await setMergeReadyMutation.mutateAsync({ taskId, ready: next, scope })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  async function handleSetMergeReady(next: boolean) {
    if (!next) {
      await commitMergeReady(false, "task")
      return
    }

    await mergeReadySpineConfirm.requestMergeReadySpine({
      tasksById,
      leafTaskId: taskId,
      onWarn: setNotice,
      onProceed: (scope) => commitMergeReady(true, scope),
    })
  }

  const attachCommand = useMemo(
    () => `rn agent attach --task ${taskId}`,
    [taskId],
  )

  const logsCommand = useMemo(() => `rn agent logs --task ${taskId}`, [taskId])

  const shellCommand = useMemo(() => `rn shell --task-id ${taskId}`, [taskId])

  async function refreshLogs() {
    if (!hasAgent) {
      setLogsText(null)
      setLogsPath(null)
      setLogsTruncated(false)
      setLogsError(null)
      return
    }
    setLogsPending(true)
    setLogsError(null)
    try {
      const response = await fetchTaskAgentLogs(taskId, {
        lines: 200,
        maxBytes: 65_536,
      })
      setLogsText(response.text)
      setLogsPath(response.path)
      setLogsTruncated(response.truncated)
    } catch (e) {
      setLogsError(e instanceof Error ? e.message : String(e))
    } finally {
      setLogsPending(false)
    }
  }

  useEffect(() => {
    if (!logsOpen) {
      return
    }
    if (!hasAgent) {
      setLogsText(null)
      setLogsPath(null)
      setLogsTruncated(false)
      setLogsError(null)
      return
    }
    void refreshLogs()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [logsOpen, taskId])

  const displayLogsText = useMemo(() => {
    if (!logsText) {
      return null
    }
    return stripAnsi(logsText)
  }, [logsText])

  const floatingPortal =
    isRunning && floatingActionsPortalId
      ? document.getElementById(floatingActionsPortalId)
      : null

  const floatingActions = isRunning ? (
    <FloatingActions>
      {pending !== null ? (
        <Button
          variant="ghost"
          size="xs"
          className="h-full rounded-none border-0"
          onClick={() =>
            void handleCopy(attachCommand, "Attach command copied")
          }
          disabledReason="Action in progress"
        >
          <Terminal />
          Attach
        </Button>
      ) : (
        <Tooltip>
          <TooltipTrigger
            render={(triggerProps) => (
              <Button
                {...triggerProps}
                variant="ghost"
                size="xs"
                className={cn(
                  "h-full rounded-none border-0",
                  triggerProps.className,
                )}
                onClick={() =>
                  void handleCopy(attachCommand, "Attach command copied")
                }
              >
                <Terminal />
                Attach
              </Button>
            )}
          />
          <TooltipContent side="bottom" sideOffset={10}>
            Copy attach command
          </TooltipContent>
        </Tooltip>
      )}
      <Button
        variant="ghost"
        size="xs"
        className="h-full rounded-none border-0 border-l"
        onClick={() => void handleRestart()}
        disabledReason={pending !== null ? "Action in progress" : null}
      >
        <RotateCcw />
        Restart
      </Button>
      <Button
        variant="ghost"
        size="xs"
        className="h-full rounded-none border-0 border-l text-destructive hover:bg-destructive/10"
        onClick={() => void handleStop()}
        disabledReason={pending !== null ? "Action in progress" : null}
      >
        <Square />
        Stop
      </Button>
    </FloatingActions>
  ) : null

  return (
    <div className="grid gap-3">
      {floatingPortal && floatingActions
        ? createPortal(floatingActions, floatingPortal)
        : null}

      <div className="flex items-start justify-between gap-3">
        <div className={"min-w-0 " + (isRunning ? "pr-28" : "")}>
          <div className="flex flex-wrap items-center gap-2">
            <ResourceBadge
              label={agentLabel}
              value={agentKindBadgeValue}
              extendBackground
              className="text-xs text-foreground/80"
            />
            <AgentStatusBadge
              status={statusLabel}
              continuable={agentContinuable}
            />
          </div>
        </div>

        {!isRunning ? (
          <div className="flex shrink-0 items-center gap-2">
            <div className="flex h-6 overflow-hidden rounded-md border border-border/60">
              {canRestart ? (
                <Button
                  variant="outline"
                  size="xs"
                  className="h-full rounded-none border-0"
                  onClick={() => void handleRestart()}
                  disabledReason={
                    pending !== null
                      ? "Action in progress"
                      : harnessEditable && !harnessDraftTrimmed
                        ? "Agent command is empty"
                        : null
                  }
                >
                  <RotateCcw />
                  Restart
                </Button>
              ) : (
                <Button
                  variant="outline"
                  size="xs"
                  className="h-full rounded-none border-0"
                  onClick={() => void handleStart()}
                  disabledReason={
                    pending !== null
                      ? "Action in progress"
                      : !configuredHarnessCommand
                        ? "Set an agent command in Configure"
                        : null
                  }
                >
                  Start
                </Button>
              )}
            </div>
          </div>
        ) : null}
      </div>

      <div className="grid gap-1">
        <div className="flex items-center justify-between gap-3">
          <div className="text-xs text-muted-foreground">Agent</div>
          <span className="rounded-md bg-foreground/5 px-2 py-1 text-[0.625rem] text-foreground/70">
            {(orchestrationDefaults?.harness.detach ?? true)
              ? "Detached"
              : "Foreground"}
          </span>
        </div>
        {harnessEditable ? (
          <input
            className="rounded-md border bg-background/40 px-2 py-2 font-mono text-xs text-foreground shadow-sm outline-none focus-visible:ring-2 focus-visible:ring-ring/30"
            value={harnessDraft}
            onChange={(e) => setHarnessDraft(e.target.value)}
            disabled={pending !== null}
          />
        ) : (
          <div className="rounded-md border bg-background/40 px-2 py-2 font-mono text-xs text-foreground shadow-sm">
            {harnessCommand || "—"}
          </div>
        )}
        <div className="text-xs text-muted-foreground">
          {harnessEditable && !harnessDraftTrimmed
            ? "Enter an agent command to use when restarting this agent."
            : harnessEditable && harnessDirty
              ? "Edited command will be used when restarting this agent."
              : agentArgv
                ? "Command used for the most recent run."
                : "Command used when starting this agent."}
        </div>
      </div>

      {!isRunning ? (
        <div className="grid gap-1">
          <div className="flex items-center justify-between gap-3">
            <div className="text-xs text-muted-foreground">Agent kind</div>
            {resolvedAgentKindLabel ? (
              <span className="rounded-md bg-foreground/5 px-2 py-1 text-[0.625rem] text-foreground/70">
                resolved: {resolvedAgentKindLabel}
              </span>
            ) : null}
          </div>
          <AgentKindSelect
            value={agentKindDraft}
            onChange={setAgentKindDraft}
            disabledReason={pending !== null ? "Action in progress" : null}
          />
          <div className="text-xs text-muted-foreground">
            {agentKindDraft === "generic"
              ? "Generic works with any command, but advanced features are disabled."
              : agentKindDraft === "auto" && inferredKindFromCommand
                ? `Auto will infer ${labelForAgentKind(inferredKindFromCommand)} from your command; switch if wrong.`
                : "Auto infers the agent from your command; switch if wrong."}
          </div>
          {structuredModeForNextRun ? (
            <div className="text-xs text-muted-foreground">
              Structured output detected for{" "}
              {labelForAgentKind(structuredAgentForNextRun ?? "generic")}.
              Ensure Agent kind is Auto (or matches) to enable semantic events.
            </div>
          ) : null}
        </div>
      ) : null}

      {agentSession?.agentInterfaceMode === "structured" ||
      structuredModeForNextRun ? (
        <div className="grid gap-1">
          <div className="flex items-center justify-between gap-3">
            <div className="text-xs text-muted-foreground">Interface mode</div>
            <span className="rounded-md bg-foreground/5 px-2 py-1 text-[0.625rem] text-foreground/70">
              Structured
            </span>
          </div>
          <div className="text-xs text-muted-foreground">
            Enabled via command flags for{" "}
            {structuredAgentForNextRun
              ? labelForAgentKind(structuredAgentForNextRun)
              : (resolvedAgentKindLabel ?? "this agent")}
            .
          </div>
        </div>
      ) : null}
      <div className="grid gap-1">
        <div className="flex items-center justify-between gap-3">
          <div className="text-xs text-muted-foreground">One-time prelude</div>
          <Button
            variant="ghost"
            size="xs"
            onClick={() => setOneTimePreludeOpen((open) => !open)}
            disabledReason={
              pending !== null
                ? "Action in progress"
                : structuredModeForNextRun
                  ? "Prelude is disabled in structured mode"
                  : null
            }
          >
            {oneTimePreludeOpen ? "Hide" : "Set"}
          </Button>
        </div>
        {structuredModeForNextRun ? (
          <div className="text-xs text-muted-foreground">
            Prelude is disabled for structured output to keep the stream
            machine-readable.
          </div>
        ) : null}
        {oneTimePreludeOpen ? (
          <>
            <Textarea
              className="min-h-20 font-mono"
              value={oneTimePrelude}
              onChange={(e) => setOneTimePrelude(e.target.value)}
              placeholder="Optional. Sent once on the next Start/Restart. Supports placeholders like {task_id}, {task_title}, {epic_slug}."
              disabled={pending !== null || structuredModeForNextRun}
            />
            <div className="flex items-center justify-between gap-3">
              <div className="text-xs text-muted-foreground">
                Not saved. Cleared after a successful start/restart.
              </div>
              <Button
                variant="ghost"
                size="xs"
                onClick={() => setOneTimePrelude("")}
                disabledReason={pending !== null ? "Action in progress" : null}
              >
                Clear
              </Button>
            </div>
          </>
        ) : null}
      </div>

      <div className="grid gap-1">
        <div className="flex items-center justify-between gap-3">
          <div className="text-xs text-muted-foreground">Merge</div>
          <Switch
            checked={mergeReady}
            onCheckedChange={(checked) => void handleSetMergeReady(checked)}
            disabledReason={
              pending !== null
                ? "Action in progress"
                : setMergeReadyMutation.isPending
                  ? "Saving…"
                  : !task.branchName && !mergeReady
                    ? "Task has no branch"
                    : null
            }
          />
        </div>
        <div className="text-xs text-muted-foreground">
          Mark this task + unmerged ancestors ready to merge (required for{" "}
          <span className="font-mono">rn merge</span> unless forced).
        </div>
      </div>

      <MergeReadySpineConfirmDialog
        open={mergeReadySpineConfirm.dialog.open}
        taskCount={mergeReadySpineConfirm.dialog.taskCount}
        dontAskAgain={mergeReadySpineConfirm.dialog.dontAskAgain}
        pendingReason={setMergeReadyMutation.isPending ? "Saving…" : null}
        onOpenChange={mergeReadySpineConfirm.dialog.setOpen}
        onDontAskAgainChange={mergeReadySpineConfirm.dialog.setDontAskAgain}
        onConfirm={mergeReadySpineConfirm.dialog.confirm}
      />

      <div className="flex flex-wrap gap-2">
        <Button
          variant="ghost"
          size="xs"
          onClick={() => void handleCopy(logsCommand, "Logs command copied")}
          disabledReason={
            pending !== null
              ? "Action in progress"
              : !hasAgent
                ? "Start the agent to see logs"
                : null
          }
        >
          Copy logs
        </Button>
        <Button
          variant="ghost"
          size="xs"
          onClick={() => setLogsOpen((open) => !open)}
          disabledReason={
            pending !== null
              ? "Action in progress"
              : !hasAgent
                ? "Start the agent to see output"
                : null
          }
        >
          {logsOpen ? "Hide logs" : "Show logs"}
        </Button>
        <Button
          variant="ghost"
          size="xs"
          onClick={() =>
            void handleCopy(shellCommand, "Worktree command copied")
          }
          disabledReason={pending !== null ? "Action in progress" : null}
        >
          Copy rn shell
        </Button>
      </div>

      {logsOpen ? (
        <div className="grid gap-2">
          <div className="flex items-center justify-between gap-3">
            <div className="text-xs text-muted-foreground">
              Agent output (tail)
            </div>
            <div className="flex items-center gap-2">
              {logsPath ? (
                <Button
                  variant="ghost"
                  size="xs"
                  onClick={() => void handleCopy(logsPath, "Log path copied")}
                  disabledReason={logsPending ? "Loading…" : null}
                >
                  Copy path
                </Button>
              ) : null}
              <Button
                variant="ghost"
                size="xs"
                onClick={() => void refreshLogs()}
                disabledReason={logsPending ? "Loading…" : null}
              >
                Refresh
              </Button>
            </div>
          </div>
          <div className="relative max-w-full overflow-hidden rounded-md border bg-background/40 px-2 py-2 font-mono text-xs text-foreground shadow-sm">
            <div className="absolute right-2 top-2 z-10">
              <Button
                variant="outline"
                size="xs"
                className="bg-background/70 backdrop-blur"
                onClick={() =>
                  void handleCopy(displayLogsText ?? "", "Output copied")
                }
                disabledReason={
                  logsPending
                    ? "Loading…"
                    : !displayLogsText
                      ? "No output to copy"
                      : null
                }
              >
                Copy
              </Button>
            </div>
            <pre className="max-h-56 min-w-0 overflow-auto whitespace-pre pr-16">
              {!hasAgent
                ? "Start the agent to see output."
                : displayLogsText || (logsPending ? "Loading…" : "No output")}
            </pre>
          </div>
          {logsTruncated ? (
            <div className="text-xs text-muted-foreground">
              Output truncated.
            </div>
          ) : null}
          {logsError ? (
            <pre className="whitespace-pre-wrap text-xs text-destructive">
              {logsError}
            </pre>
          ) : null}
        </div>
      ) : null}

      {error ? (
        <pre className="whitespace-pre-wrap text-xs text-destructive">
          {error}
        </pre>
      ) : null}
      {notice ? (
        <pre className="whitespace-pre-wrap text-xs text-muted-foreground">
          {notice}
        </pre>
      ) : null}
    </div>
  )
}

export function DetailsPanel({
  open,
  task,
  node,
  tasksById,
  mergeRun,
  agentSession,
  edge,
}: DetailsPanelProps) {
  const selectionKey = edge
    ? `edge:${edge.id}`
    : node
      ? `node:${node.id}`
      : task
        ? `task:${task.id}`
        : "none"

  const mergeRunActive =
    mergeRun &&
    (mergeRun.status === "blocked" ||
      mergeRun.status === "resumable" ||
      mergeRun.status === "running" ||
      mergeRun.status === "failed")
      ? mergeRun
      : null

  const defaultSections = edge
    ? ["contract"]
    : mergeRunActive
      ? ["merge-run", "agent", "readme"]
      : ["agent", "readme"]
  const floatingActionsPortalId = "details-panel-floating-actions"

  return (
    <SlidePanel open={open}>
      <div className="relative p-3">
        <div id={floatingActionsPortalId} className="contents" />
        <Accordion
          key={selectionKey}
          multiple
          defaultValue={defaultSections}
          className="border-0"
        >
          {edge ? (
            <>
              <AccordionItem value="contract">
                <AccordionTrigger>Contract</AccordionTrigger>
                <AccordionContent className="pt-3">
                  <div className="text-muted-foreground">
                    No contract recorded yet.
                  </div>
                </AccordionContent>
              </AccordionItem>

              <AccordionItem value="commits">
                <AccordionTrigger>Commits</AccordionTrigger>
                <AccordionContent className="pt-3">
                  <div className="grid gap-3">
                    <div className="grid gap-1">
                      <div className="text-xs text-muted-foreground">Count</div>
                      {edge.commitCount !== null &&
                      edge.commitCount !== undefined ? (
                        <div className="text-sm">{edge.commitCount}</div>
                      ) : (
                        <div className="text-sm text-muted-foreground">
                          Unavailable
                        </div>
                      )}
                    </div>
                    <div className="grid gap-1">
                      <div className="text-xs text-muted-foreground">Range</div>
                      {edge.baseSha && edge.headSha ? (
                        <div className="font-mono text-xs text-foreground/80">
                          {edge.baseSha.slice(0, 7)}..{edge.headSha.slice(0, 7)}
                        </div>
                      ) : (
                        <div className="text-sm text-muted-foreground">
                          Unavailable
                        </div>
                      )}
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>

              <AccordionItem value="messages">
                <AccordionTrigger>Messages</AccordionTrigger>
                <AccordionContent className="pt-3">
                  <div className="text-muted-foreground">No messages.</div>
                </AccordionContent>
              </AccordionItem>
            </>
          ) : (
            <>
              {mergeRunActive ? (
                <AccordionItem value="merge-run">
                  <AccordionTrigger>Merge Run</AccordionTrigger>
                  <AccordionContent className="pt-3">
                    <MergeRunDetails
                      mergeRun={mergeRunActive}
                      node={node}
                      panelOpen={open}
                    />
                  </AccordionContent>
                </AccordionItem>
              ) : null}

              <AccordionItem value="agent">
                <AccordionTrigger>Agent</AccordionTrigger>
                <AccordionContent className="pt-3">
                  {task ? (
                    <AgentActions
                      task={task}
                      tasksById={tasksById}
                      agentSession={agentSession ?? null}
                      floatingActionsPortalId={floatingActionsPortalId}
                    />
                  ) : null}
                </AccordionContent>
              </AccordionItem>

              <AccordionItem value="readme">
                <AccordionTrigger>README</AccordionTrigger>
                <AccordionContent className="pt-3">
                  {task?.readme ? (
                    <Markdown
                      content={task.readme}
                      omitFirstHeading
                      omitMetadataSection
                    />
                  ) : (
                    <div className="text-muted-foreground">No README.</div>
                  )}
                </AccordionContent>
              </AccordionItem>
            </>
          )}
        </Accordion>
      </div>
    </SlidePanel>
  )
}
