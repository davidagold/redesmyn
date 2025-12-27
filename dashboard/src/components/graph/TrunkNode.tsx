import { Handle, Position, type Node, type NodeProps } from "@xyflow/react"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
import { TRUNK_COMMIT_ROW_HEIGHT } from "./graphConfig"

type TrunkMark = {
  type: "commit" | "base" | "ellipsis"
  sha?: string
  authorName?: string | null
  authorEmail?: string | null
  authoredAt?: string | null
}

export type TrunkNodeData = Record<string, unknown> & {
  className?: string
  marks?: TrunkMark[]
  baseOffset?: number
  commitSpacing?: number
  commitPadding?: number
  lineWidth?: number
  labelOffset?: number
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
  const labelOffset = data.labelOffset ?? lineWidth + 12
  const rowHeight = TRUNK_COMMIT_ROW_HEIGHT
  const baseOffset = data.baseOffset ?? commitPadding + rowHeight / 2
  const arrowSize = 12

  function renderCircle() {
    return (
      <span className="h-2.5 w-2.5 rounded-full border border-foreground/60" />
    )
  }

  return (
    <div className={cn("relative h-full w-full", data.className)}>
      <div
        className="absolute top-0"
        style={{
          left: lineWidth / 2,
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
        className="absolute left-0 top-0 h-full rounded-full bg-border/60"
        style={{ width: lineWidth }}
      />
      <Handle
        id="base"
        type="source"
        position={Position.Right}
        className="h-2 w-2 border-0 bg-transparent opacity-0"
        style={{
          top: baseOffset,
          left: lineWidth / 2,
          right: "auto",
          transform: "translateX(-50%)",
        }}
      />
      {marks.length ? (
        <div className="absolute top-0 h-full" style={{ left: labelOffset }}>
          {marks.map((mark, index) => {
            const y = commitPadding + index * commitSpacing
            if (mark.type === "ellipsis") {
              return (
                <div
                  key={`ellipsis-${index}`}
                  className="absolute flex items-center text-xs text-foreground/50"
                  style={{ top: y, height: rowHeight }}
                >
                  <span className="tracking-[0.2em]">···</span>
                </div>
              )
            }
            const sha = mark.sha ? mark.sha.slice(0, 7) : ""
            const isBase = mark.type === "base"
            const authoredAt = formatTimestamp(mark.authoredAt)
            const author =
              mark.authorName || mark.authorEmail || "Unknown author"
            return (
              <Tooltip key={`${mark.type}-${mark.sha ?? index}`}>
                <TooltipTrigger
                  className="absolute flex cursor-default items-center gap-2 border-0 bg-transparent p-0 text-xs"
                  style={{ top: y, height: rowHeight }}
                >
                  {renderCircle()}
                  <span className="font-mono text-foreground/70">{sha}</span>
                </TooltipTrigger>
                <TooltipContent
                  side="right"
                  align="center"
                  className="space-y-1"
                >
                  <div className="font-mono text-[10px] text-background/80">
                    {mark.sha}
                  </div>
                  <div className="text-xs">{author}</div>
                  {authoredAt ? (
                    <div className="text-xs text-background/70">
                      {authoredAt}
                    </div>
                  ) : null}
                  {isBase ? (
                    <div className="text-[10px] uppercase tracking-[0.2em] text-background/70">
                      Base
                    </div>
                  ) : null}
                </TooltipContent>
              </Tooltip>
            )
          })}
        </div>
      ) : null}
    </div>
  )
}
