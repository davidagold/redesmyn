import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"
import { SlidePanel } from "@/components/ui/slide-panel"
import { Markdown } from "@/components/markdown"
import type { Task } from "@/lib/graph-utils"

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

interface DetailsPanelProps {
  open: boolean
  task: Task | null
  edge?: EdgeSelection | null
}

export function DetailsPanel({ open, task, edge }: DetailsPanelProps) {
  const selectionKey = edge
    ? `edge:${edge.id}`
    : task
      ? `task:${task.id}`
      : "none"
  const defaultSections = edge ? ["contract"] : ["readme"]

  return (
    <SlidePanel open={open}>
      <div className="p-3">
        <Accordion
          key={selectionKey}
          multiple={!!edge}
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
          )}
        </Accordion>
      </div>
    </SlidePanel>
  )
}
