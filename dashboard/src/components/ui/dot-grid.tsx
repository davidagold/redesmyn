import { cn } from "@/lib/utils"

export function DotGrid({ className }: { className?: string }) {
  return (
    <div
      aria-hidden="true"
      className={cn(
        "pointer-events-none absolute inset-0 opacity-[0.06] [background-image:radial-gradient(var(--border)_1px,transparent_1px)] [background-size:32px_32px]",
        className,
      )}
    />
  )
}
