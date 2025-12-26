import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"
import { SlidePanel } from "@/components/ui/slide-panel"
import { Markdown } from "@/components/markdown"
import type { Task } from "@/lib/graph-utils"

interface DetailsPanelProps {
  open: boolean
  task: Task | null
}

export function DetailsPanel({ open, task }: DetailsPanelProps) {
  return (
    <SlidePanel open={open}>
      <div className="p-3">
        <Accordion defaultValue={["readme"]} className="border-0">
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
        </Accordion>
      </div>
    </SlidePanel>
  )
}
