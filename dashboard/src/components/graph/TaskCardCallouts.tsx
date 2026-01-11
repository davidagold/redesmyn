import { useState, type ComponentProps, type ReactNode } from "react"
import { Alert } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { copyToClipboard } from "@/lib/clipboard"
import { cn } from "@/lib/utils"
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  MessageSquareText,
  Play,
  Terminal,
  X,
} from "lucide-react"

export type TaskCardActionError = {
  title: string
  summary: string
  raw: string
}

type AlertVariant = NonNullable<ComponentProps<typeof Alert>["variant"]>

type DismissibleCalloutProps = {
  variant: AlertVariant
  icon: ReactNode
  title?: string | null
  message: ReactNode
  dismissLabel: string
  messageClassName?: string
  onDismiss: () => void
}

function BranchReadyLeading() {
  return (
    <Play className="size-4 shrink-0 text-emerald-200" aria-hidden="true" />
  )
}

function DismissibleCallout({
  variant,
  icon,
  title = null,
  message,
  dismissLabel,
  messageClassName,
  onDismiss,
}: DismissibleCalloutProps) {
  if (title) {
    return (
      <Alert variant={variant} className="gap-2 shadow-lg">
        <div className="flex items-start justify-between gap-2">
          <div className="flex min-w-0 items-center gap-2">
            {icon}
            <div className="text-xs font-medium text-foreground">{title}</div>
          </div>
          <Button
            variant="ghost"
            size="icon-xs"
            aria-label={dismissLabel}
            onClick={(e) => {
              e.preventDefault()
              e.stopPropagation()
              onDismiss()
            }}
          >
            <X className="size-3" />
          </Button>
        </div>
        <div className={cn("text-xs text-foreground/80", messageClassName)}>
          {message}
        </div>
      </Alert>
    )
  }

  return (
    <Alert variant={variant} className="gap-1 shadow-lg">
      <div className="flex min-w-0 items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2">
          {icon}
          <div
            className={cn(
              "truncate text-[11px] text-foreground/80",
              messageClassName,
            )}
          >
            {message}
          </div>
        </div>
        <Button
          variant="ghost"
          size="icon-xs"
          aria-label={dismissLabel}
          onClick={(e) => {
            e.preventDefault()
            e.stopPropagation()
            onDismiss()
          }}
        >
          <X className="size-3" />
        </Button>
      </div>
    </Alert>
  )
}

export function MergeReadySpineWarningCallout({
  notice,
  onDismiss,
}: {
  notice: string
  onDismiss: () => void
}) {
  return (
    <DismissibleCallout
      variant="amber"
      icon={<AlertTriangle className="size-3.5 text-amber-400" />}
      title="Merge spine warning"
      message={notice}
      dismissLabel="Dismiss warning"
      onDismiss={onDismiss}
    />
  )
}

export function SuccessToastCallout({
  message,
  onDismiss,
}: {
  message: string
  onDismiss: () => void
}) {
  return (
    <DismissibleCallout
      variant="emerald"
      icon={<CheckCircle2 className="size-3.5 text-emerald-300" />}
      message={message}
      dismissLabel="Dismiss notice"
      onDismiss={onDismiss}
    />
  )
}

export function ActionErrorCallout({
  error,
  mergeRunBlockedRebase,
  rebaseAttachCommand,
  rebaseRemediationMessage,
  onDismiss,
}: {
  error: TaskCardActionError
  mergeRunBlockedRebase: boolean
  rebaseAttachCommand: string | null
  rebaseRemediationMessage: string | null
  onDismiss: () => void
}) {
  const [expanded, setExpanded] = useState(false)

  return (
    <Alert variant="destructive" className="gap-2 shadow-lg">
      <div className="flex items-start justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2">
          <AlertTriangle className="size-3.5 text-destructive" />
          <Tooltip>
            <TooltipTrigger
              render={(triggerProps) => (
                <div
                  {...triggerProps}
                  className={cn(
                    "truncate text-xs font-medium text-foreground",
                    triggerProps.className,
                  )}
                >
                  {error.title}
                </div>
              )}
            />
            <TooltipContent side="top" sideOffset={8}>
              {error.title}
            </TooltipContent>
          </Tooltip>
        </div>
        <Tooltip>
          <TooltipTrigger
            render={(triggerProps) => (
              <Button
                {...triggerProps}
                variant="ghost"
                size="icon-xs"
                aria-label="Dismiss error"
                className={cn(triggerProps.className)}
                onClick={(e) => {
                  e.preventDefault()
                  e.stopPropagation()
                  onDismiss()
                }}
              >
                <X className="size-3" />
              </Button>
            )}
          />
          <TooltipContent side="bottom" sideOffset={10}>
            Dismiss error
          </TooltipContent>
        </Tooltip>
      </div>

      <div
        className={cn(
          "break-words text-xs text-foreground/80",
          expanded ? "whitespace-pre-wrap" : "line-clamp-2",
        )}
      >
        {error.summary}
      </div>

      {mergeRunBlockedRebase ? (
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <Button
            variant="ghost"
            size="xs"
            disabledReason={
              rebaseAttachCommand
                ? null
                : "No task id recorded for this blocked step."
            }
            onClick={(e) => {
              e.preventDefault()
              e.stopPropagation()
              if (!rebaseAttachCommand) {
                return
              }
              void copyToClipboard(rebaseAttachCommand)
            }}
          >
            <Terminal className="size-3" />
            Copy attach
          </Button>
          <Button
            variant="ghost"
            size="xs"
            disabledReason={
              rebaseRemediationMessage
                ? null
                : "No remediation message available."
            }
            onClick={(e) => {
              e.preventDefault()
              e.stopPropagation()
              if (!rebaseRemediationMessage) {
                return
              }
              void copyToClipboard(rebaseRemediationMessage)
            }}
          >
            <MessageSquareText className="size-3" />
            Copy agent note
          </Button>
        </div>
      ) : null}

      <div className="mt-1 flex items-center justify-between gap-2">
        <Button
          variant="ghost"
          size="xs"
          onClick={(e) => {
            e.preventDefault()
            e.stopPropagation()
            setExpanded((current) => !current)
          }}
        >
          {expanded ? (
            <ChevronUp className="size-3" />
          ) : (
            <ChevronDown className="size-3" />
          )}
          {expanded ? "Hide" : "Show"} details
        </Button>
        <Button
          variant="ghost"
          size="xs"
          onClick={(e) => {
            e.preventDefault()
            e.stopPropagation()
            void copyToClipboard(error.raw)
          }}
        >
          Copy
        </Button>
      </div>

      {expanded ? (
        <div className="mt-1 max-h-40 overflow-auto rounded-md bg-background/40 px-2 py-1.5 font-mono text-[0.625rem] text-foreground/80">
          <div className="whitespace-pre-wrap break-words">{error.raw}</div>
        </div>
      ) : null}
    </Alert>
  )
}

export function ExpandableStatusCallout({
  variant,
  leading,
  summary,
  detail = null,
  actions = null,
  actionsVisibility = "hover",
}: {
  variant: AlertVariant
  leading: ReactNode
  summary: string
  detail?: string | null
  actions?: ReactNode
  actionsVisibility?: "hover" | "always"
}) {
  const [expanded, setExpanded] = useState(false)
  const canExpand = detail !== null && detail.trim() !== ""

  return (
    <Alert
      variant={variant}
      className={cn(
        "group gap-1 shadow-none ring-0",
        "max-w-full overflow-hidden",
        canExpand ? "cursor-pointer select-none" : "cursor-default",
      )}
      role={canExpand ? "button" : undefined}
      tabIndex={canExpand ? 0 : undefined}
      aria-expanded={canExpand ? expanded : undefined}
      onClick={(e) => {
        e.preventDefault()
        e.stopPropagation()
        if (!canExpand) {
          return
        }
        setExpanded((current) => !current)
      }}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault()
          e.stopPropagation()
          if (!canExpand) {
            return
          }
          setExpanded((current) => !current)
        }
      }}
    >
      <div className="flex min-w-0 items-center justify-between gap-2">
        <div className="flex min-w-0 flex-1 items-center gap-2">
          {leading}
          <div className="truncate text-[11px] text-foreground/70">
            {summary}
          </div>
        </div>
        {actions ? (
          <div
            className={cn(
              "flex shrink-0 items-center gap-1 transition-opacity",
              actionsVisibility === "always"
                ? "opacity-100"
                : expanded
                  ? "opacity-100"
                  : "opacity-0 group-hover:opacity-100 group-focus-within:opacity-100",
            )}
            onClick={(event) => event.stopPropagation()}
          >
            {actions}
          </div>
        ) : null}
      </div>

      {expanded && canExpand ? (
        <div className="text-[11px] text-foreground/70">{detail}</div>
      ) : null}
    </Alert>
  )
}

export function BlockedRebaseCallout({
  variant,
  badgeLabel,
  summary,
  detail,
  status,
  assistActive,
  operation,
  attachCommand,
  remediationMessage,
  onResume,
}: {
  variant: AlertVariant
  badgeLabel: string
  summary: string
  detail: string | null
  status: "blocked" | "resumable"
  assistActive: boolean
  operation: "merge" | "restack"
  attachCommand: string | null
  remediationMessage: string | null
  onResume: () => void
}) {
  if (status === "resumable" && !assistActive) {
    return (
      <ExpandableStatusCallout
        variant={variant}
        leading={<BranchReadyLeading />}
        summary="Branch is ready"
        detail={null}
        actions={
          <Button
            variant="outline"
            size="xs"
            className="border-emerald-400/35 text-emerald-100 hover:bg-emerald-400/10 hover:text-emerald-50"
            onClick={(e) => {
              e.preventDefault()
              e.stopPropagation()
              onResume()
            }}
          >
            {operation === "restack" ? "Resume restack" : "Resume merge"}
          </Button>
        }
        actionsVisibility="always"
      />
    )
  }

  const actions = (
    <>
      {status === "blocked" && attachCommand ? (
        <Tooltip>
          <TooltipTrigger
            render={(triggerProps) => (
              <Button
                {...triggerProps}
                variant="ghost"
                size="icon-sm"
                aria-label="Copy attach command"
                onClick={(e) => {
                  e.preventDefault()
                  e.stopPropagation()
                  void copyToClipboard(attachCommand)
                }}
              >
                <Terminal />
              </Button>
            )}
          />
          <TooltipContent side="bottom" sideOffset={10} showArrow={false}>
            Copy attach
          </TooltipContent>
        </Tooltip>
      ) : null}

      {status === "blocked" && remediationMessage ? (
        <Tooltip>
          <TooltipTrigger
            render={(triggerProps) => (
              <Button
                {...triggerProps}
                variant="ghost"
                size="icon-sm"
                aria-label="Copy agent note"
                onClick={(e) => {
                  e.preventDefault()
                  e.stopPropagation()
                  void copyToClipboard(remediationMessage)
                }}
              >
                <MessageSquareText />
              </Button>
            )}
          />
          <TooltipContent side="bottom" sideOffset={10} showArrow={false}>
            Copy agent note
          </TooltipContent>
        </Tooltip>
      ) : null}
    </>
  )

  return (
    <ExpandableStatusCallout
      variant={variant}
      leading={
        <Badge variant={variant} size="xs">
          {badgeLabel}
        </Badge>
      }
      summary={summary}
      detail={detail}
      actions={actions}
    />
  )
}

export function ResumableMergeCallout({
  variant,
  detail: _detail,
  operation,
  disabledReason,
  onResume,
}: {
  variant: AlertVariant
  detail: string | null
  operation: "merge" | "restack"
  disabledReason: string | null
  onResume: () => void
}) {
  const actions = (
    <Button
      variant="outline"
      size="xs"
      className="border-emerald-400/35 text-emerald-100 hover:bg-emerald-400/10 hover:text-emerald-50"
      disabledReason={disabledReason}
      onClick={(e) => {
        e.preventDefault()
        e.stopPropagation()
        onResume()
      }}
    >
      {operation === "restack" ? "Resume restack" : "Resume merge"}
    </Button>
  )

  return (
    <ExpandableStatusCallout
      variant={variant}
      leading={<BranchReadyLeading />}
      summary="Branch is ready"
      detail={null}
      actions={actions}
      actionsVisibility="always"
    />
  )
}
