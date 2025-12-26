import { Button } from "@/components/ui/button"
import { ChevronDown } from "lucide-react"

export function WorkspaceButton() {
  return (
    <Button variant="ghost" className="w-fit gap-2" disabled>
      <span className="truncate">Redesmyn</span>
      <ChevronDown className="h-4 w-4 text-muted-foreground" />
    </Button>
  )
}
