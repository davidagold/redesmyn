import { useEffect, useMemo, useState } from "react"
import type { Epic, GithubRepo } from "@/api"
import { useEpicGithubRepoConfigQuery } from "@/api/queries"
import { useUpdateEpicGithubRepoMutation } from "@/api/mutations"
import { Button } from "@/components/ui/button"
import { GithubIcon } from "@/components/github/GithubIcon"
import { cn } from "@/lib/utils"
import { Loader2, RotateCcw, Save, Unlink } from "lucide-react"

function repoFullName(repo: GithubRepo | null | undefined): string | null {
  if (!repo) {
    return null
  }
  return `${repo.owner}/${repo.repo}`
}

export function EpicGithubRepoBadge({ epic }: { epic: Epic }) {
  const [popoverOpen, setPopoverOpen] = useState(false)
  const [repoInput, setRepoInput] = useState("")
  const [repoIsDirty, setRepoIsDirty] = useState(false)

  const configQuery = useEpicGithubRepoConfigQuery(epic.slug, {
    enabled: popoverOpen,
  })
  const updateMutation = useUpdateEpicGithubRepoMutation()

  const configured = configQuery.data?.configured ?? null
  const detected = configQuery.data?.detected ?? null
  const effective = configQuery.data?.effective ?? null

  const configuredFullName = useMemo(
    () => repoFullName(configured),
    [configured],
  )
  const detectedFullName = useMemo(() => repoFullName(detected), [detected])
  const effectiveFullName = useMemo(() => repoFullName(effective), [effective])

  const isSaving = updateMutation.isPending

  const displayedValue = useMemo(() => {
    if (repoIsDirty) {
      return repoInput
    }
    return configuredFullName ?? detectedFullName ?? ""
  }, [configuredFullName, detectedFullName, repoInput, repoIsDirty])

  useEffect(() => {
    if (!popoverOpen) {
      setRepoInput("")
      setRepoIsDirty(false)
      return
    }
    setRepoInput((configuredFullName ?? detectedFullName ?? "").trim())
    setRepoIsDirty(false)
  }, [popoverOpen, configuredFullName, detectedFullName])

  function handleClose() {
    setPopoverOpen(false)
  }

  async function handleSave() {
    const next = displayedValue.trim()
    await updateMutation.mutateAsync({
      epicSlug: epic.slug,
      repo: next ? next : null,
    })
  }

  async function handleUseDetected() {
    if (!detectedFullName) {
      return
    }
    await updateMutation.mutateAsync({
      epicSlug: epic.slug,
      repo: detectedFullName,
    })
  }

  async function handleUnset() {
    await updateMutation.mutateAsync({ epicSlug: epic.slug, repo: null })
  }

  return (
    <div className="relative">
      <Button
        variant="ghost"
        size="sm"
        className={cn(
          "h-7 w-7 p-0",
          effectiveFullName ? null : "text-muted-foreground",
        )}
        onClick={() => setPopoverOpen((open) => !open)}
        aria-expanded={popoverOpen}
        title={effectiveFullName ? `GitHub repo: ${effectiveFullName}` : "GitHub repo"}
      >
        <GithubIcon className="size-3.5" />
      </Button>

      {popoverOpen ? (
        <>
          <button
            type="button"
            aria-label="Close"
            className="fixed inset-0 z-10 cursor-default bg-transparent"
            onClick={handleClose}
          />
          <div className="absolute left-0 top-full z-20 mt-2 min-w-56 max-w-72 rounded-lg border bg-popover p-2 shadow">
            <div className="mb-2 px-2 text-xs font-medium text-muted-foreground">
              GitHub Repo
            </div>

            {configQuery.isLoading ? (
              <div className="flex items-center justify-center py-4">
                <Loader2 className="size-4 animate-spin text-muted-foreground" />
              </div>
            ) : configQuery.isError ? (
              <div className="px-2 py-2 text-xs text-destructive">
                Failed to load repo config
              </div>
            ) : (
              <div className="space-y-3 px-1.5 pb-1.5">
                <div className="grid gap-1 text-xs">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-muted-foreground">Detected</span>
                    <span className="truncate">
                      {detectedFullName ?? "—"}
                    </span>
                  </div>
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-muted-foreground">Saved</span>
                    <span className="truncate">
                      {configuredFullName ?? "—"}
                    </span>
                  </div>
                </div>

                <div className="space-y-1.5">
                  <div className="text-xs text-muted-foreground">Owner/repo</div>
                  <input
                    type="text"
                    className="h-7 w-full rounded-md bg-background/40 px-2 text-sm shadow-sm outline-none focus-visible:ring-2 focus-visible:ring-ring/30"
                    placeholder="owner/repo"
                    value={displayedValue}
                    onChange={(e) => {
                      setRepoIsDirty(true)
                      setRepoInput(e.target.value)
                    }}
                    disabled={isSaving}
                  />
                </div>

                <div className="flex flex-wrap items-center gap-1.5">
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-7"
                    onClick={() => void handleSave()}
                    disabledReason={isSaving ? "Saving..." : null}
                  >
                    <Save className="size-3.5" />
                    Save
                  </Button>

                  {detectedFullName ? (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7"
                      onClick={() => void handleUseDetected()}
                      disabledReason={isSaving ? "Saving..." : null}
                    >
                      <RotateCcw className="size-3.5" />
                      Use detected
                    </Button>
                  ) : null}

                  {configuredFullName ? (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7 text-muted-foreground"
                      onClick={() => void handleUnset()}
                      disabledReason={isSaving ? "Saving..." : null}
                    >
                      <Unlink className="size-3.5" />
                      Unset
                    </Button>
                  ) : null}
                </div>
              </div>
            )}
          </div>
        </>
      ) : null}
    </div>
  )
}

