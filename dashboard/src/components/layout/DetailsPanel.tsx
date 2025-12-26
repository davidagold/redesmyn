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
}

interface DetailsPanelProps {
  open: boolean
  task: Task | null
  edge?: EdgeSelection | null
}

export function DetailsPanel({ open, task, edge }: DetailsPanelProps) {
  const defaultSections = edge ? ["connection"] : ["readme"]

  return (
    <SlidePanel open={open}>
      <div className="p-3">
        <Accordion defaultValue={defaultSections} className="border-0">
          {edge ? (
            <AccordionItem value="connection">
              <AccordionTrigger>Connection</AccordionTrigger>
              <AccordionContent className="pt-3">
                <div className="grid gap-3">
                  <div className="grid gap-1">
                    <div className="text-xs text-muted-foreground">From</div>
                    <div className="text-sm">{edge.fromLabel}</div>
                  </div>
                  <div className="grid gap-1">
                    <div className="text-xs text-muted-foreground">To</div>
                    <div className="text-sm">{edge.toLabel}</div>
                  </div>
                </div>
              </AccordionContent>
            </AccordionItem>
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
