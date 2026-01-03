import { Handle, Position, type Node, type NodeProps } from "@xyflow/react"
import type { ReactNode } from "react"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
import { TRUNK_COMMIT_ROW_HEIGHT } from "./graphConfig"

type TrunkMark = {
  type: "commit" | "base" | "connector" | "ellipsis"
  sha?: string
  title?: string | null
  message?: string | null
  authorName?: string | null
  authorEmail?: string | null
  authoredAt?: string | null
  committerName?: string | null
  committerEmail?: string | null
  committedAt?: string | null
}

export type TrunkNodeData = Record<string, unknown> & {
  className?: string
  marks?: TrunkMark[]
  baseOffset?: number
  commitSpacing?: number
  commitPadding?: number
  lineWidth?: number
  lineX?: number
  titleWidth?: number
  markerWidth?: number
  shaWidth?: number
}

export type TrunkNodeType = Node<TrunkNodeData, "trunk">

function formatTimestamp(value?: string | null) {
  if (!value) {
    return null
  }
  const date = new Date(value)
  if (Number.isNaN(date.valueOf())) {
    return value
  }
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date)
}

export function TrunkNode({ data }: NodeProps<TrunkNodeType>) {
  const marks = data.marks ?? []
  const commitSpacing = data.commitSpacing ?? 28
  const commitPadding = data.commitPadding ?? 16
  const lineWidth = data.lineWidth ?? 2
  const titleWidth = data.titleWidth ?? 200
  const markerWidth = data.markerWidth ?? 28
  const shaWidth = data.shaWidth ?? 110
  const lineX = data.lineX ?? titleWidth + markerWidth / 2
  const rowHeight = TRUNK_COMMIT_ROW_HEIGHT
  const baseOffset = data.baseOffset ?? commitPadding + rowHeight / 2
  const arrowSize = 12
  const labelGapPx = 10

  function renderCommitTooltip(
    mark: TrunkMark,
    child: ReactNode,
    options?: { fullWidth?: boolean },
  ) {
    const title = mark.title ?? ""
    const message = mark.message ?? ""
    const committerName = mark.committerName ?? mark.authorName ?? null
    const committerEmail = mark.committerEmail ?? mark.authorEmail ?? null
    const committer =
      committerName && committerEmail
        ? `${committerName} <${committerEmail}>`
        : committerName || committerEmail || "Unknown committer"
    const timestamp = formatTimestamp(mark.committedAt ?? mark.authoredAt)

    return (
      <Tooltip>
        <TooltipTrigger
          render={(triggerProps) => (
            <span
              {...triggerProps}
              className={cn(
                options?.fullWidth ? "flex w-full" : "inline-flex",
                triggerProps.className,
              )}
            >
              {child}
            </span>
          )}
        />
        <TooltipContent
          side="right"
          sideOffset={14}
          align="center"
          className="max-w-[40rem]"
        >
          <div className="space-y-1">
            <div className="text-xs font-medium">{committer}</div>
            {timestamp ? (
              <div className="text-[11px] text-background/75">{timestamp}</div>
            ) : null}
            {title ? <div className="text-xs font-medium">{title}</div> : null}
            {message ? (
              <div className="max-h-64 overflow-auto whitespace-pre-wrap font-mono text-[11px] text-background/85">
                {message}
              </div>
            ) : null}
          </div>
        </TooltipContent>
      </Tooltip>
    )
  }

  function renderMarker(emphasis: boolean) {
    const wrapperClass = "flex h-3 w-3 items-center justify-center"
    return (
      <span className={wrapperClass}>
        <span
          className={cn(
            "h-2.5 w-2.5 rounded-full border",
            emphasis
              ? "border-foreground/75 bg-foreground/5"
              : "border-foreground/60",
          )}
        />
      </span>
    )
  }

  return (
    <div className={cn("relative h-full w-full", data.className)}>
      <div
        className="absolute top-0"
        style={{
          left: lineX,
          transform: `translate(-50%, -${arrowSize}px)`,
        }}
        aria-hidden="true"
      >
        <svg
          width={arrowSize}
          height={arrowSize}
          viewBox="0 0 10 8"
          className="block"
        >
          <path d="M5 0 L10 8 H0 Z" fill="var(--border)" fillOpacity={0.6} />
        </svg>
      </div>
      <div
        className="absolute top-0 h-full rounded-full bg-border/60"
        style={{ width: lineWidth, left: lineX - lineWidth / 2 }}
      />
      <Handle
        id="base"
        type="source"
        position={Position.Right}
        className="h-2 w-2 border-0 bg-transparent opacity-0"
        style={{
          top: baseOffset,
          left: lineX,
          right: "auto",
          transform: "translate(-50%, -50%)",
        }}
      />
      {marks.length ? (
        <div className="absolute left-0 top-0 h-full w-full">
          {marks.map((mark, index) => {
            const y = commitPadding + index * commitSpacing
            if (mark.type === "connector") {
              return null
            }
            if (mark.type === "ellipsis") {
              return (
                <div
                  key={`ellipsis-${index}`}
                  className="absolute flex items-center text-xs text-foreground/50"
                  style={{
                    top: y,
                    height: rowHeight,
                    left: titleWidth + markerWidth + labelGapPx,
                  }}
                >
                  <span className="tracking-[0.2em]">···</span>
                </div>
              )
            }
            const sha = mark.sha ? mark.sha.slice(0, 7) : ""
            const isBase = mark.type === "base"
            const titleText = mark.title ?? "—"
            return (
              <div
                key={`${mark.type}-${mark.sha ?? index}`}
                className="absolute flex items-center"
                style={{ top: y, height: rowHeight, left: 0 }}
              >
                <div
                  className="flex items-center justify-end"
                  style={{ width: titleWidth, paddingRight: labelGapPx }}
                >
                  {renderCommitTooltip(
                    mark,
                    <span
                      className={cn(
                        "block w-full truncate text-right text-xs",
                        isBase
                          ? "text-foreground/75 font-medium"
                          : "text-foreground/60",
                      )}
                    >
                      {titleText}
                    </span>,
                    { fullWidth: true },
                  )}
                </div>

                <div
                  className="flex items-center justify-center"
                  style={{ width: markerWidth }}
                >
                  {renderCommitTooltip(mark, renderMarker(isBase))}
                </div>

                <div
                  className="flex items-center"
                  style={{ width: shaWidth, paddingLeft: labelGapPx }}
                >
                  {renderCommitTooltip(
                    mark,
                    <span
                      className={cn(
                        "truncate font-mono text-xs",
                        isBase
                          ? "text-foreground/80 font-medium"
                          : "text-foreground/70",
                      )}
                    >
                      {sha}
                    </span>,
                    { fullWidth: true },
                  )}
                </div>
              </div>
            )
          })}
        </div>
      ) : null}
    </div>
  )
}
