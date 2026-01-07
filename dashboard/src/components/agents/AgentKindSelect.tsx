import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  type AgentKindSelection,
  labelForAgentKindSelection,
} from "@/lib/agent-kind"
import { cn } from "@/lib/utils"
import { ChevronDown } from "lucide-react"

const OPTIONS: { value: AgentKindSelection label: string }[] = [
  { value: "auto", label: "Auto" },
  { value: "generic", label: "Generic" },
  { value: "codex", label: "Codex" },
  { value: "claude_code", label: "Claude Code" },
]

export type AgentKindSelectProps = {
  value: AgentKindSelection
  onChange: (next: AgentKindSelection) => void
  disabledReason?: string | null
  className?: string
}

export function AgentKindSelect({
  value,
  onChange,
  disabledReason,
  className,
}: AgentKindSelectProps) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        render={(triggerProps) => (
          <Button
            {...triggerProps}
            variant="outline"
            size="xs"
            className={cn(
              "h-7 w-full justify-between bg-background/40 px-2 font-normal text-foreground shadow-sm",
              triggerProps.className,
              className,
            )}
            disabledReason={disabledReason}
          >
            <span className="truncate">
              {labelForAgentKindSelection(value)}
            </span>
            <ChevronDown className="size-3.5 opacity-70" />
          </Button>
        )}
      />
      <DropdownMenuContent align="start" side="bottom" sideOffset={6}>
        <DropdownMenuRadioGroup
          value={value}
          onValueChange={(next) => onChange(next as AgentKindSelection)}
        >
          {OPTIONS.map((option) => (
            <DropdownMenuRadioItem key={option.value} value={option.value}>
              {option.label}
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
