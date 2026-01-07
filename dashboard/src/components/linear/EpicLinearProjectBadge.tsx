import { useState } from "react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import type { Epic, LinearMilestone, LinearProject } from "@/api"
import {
  useEpicLinearConfigQuery,
  useLinearMilestonesQuery,
  useLinearProjectsQuery,
} from "@/api/queries"
import {
  useUpdateEpicLinearConfigMutation,
  useUpdateEpicLinearProjectMutation,
} from "@/api/mutations"
import { LinearIcon } from "@/components/linear/LinearIcon"
import {
  ArrowLeft,
  Check,
  ChevronRight,
  Loader2,
  Tag,
  Target,
  Unlink,
} from "lucide-react"

type PanelView = "projects" | "config"

interface EpicLinearProjectBadgeProps {
  epic: Epic
  linearConnected: boolean
}

export function EpicLinearProjectBadge({
  epic,
  linearConnected,
}: EpicLinearProjectBadgeProps) {
  const [popoverOpen, setPopoverOpen] = useState(false)
  const [view, setView] = useState<PanelView>("projects")
  const [labelInput, setLabelInput] = useState("")
  const [labelIsDirty, setLabelIsDirty] = useState(false)

  const projectsQuery = useLinearProjectsQuery({
    enabled: linearConnected && popoverOpen,
  })

  const milestonesQuery = useLinearMilestonesQuery(epic.linearProjectId, {
    enabled: linearConnected && popoverOpen && view === "config",
  })

  const configQuery = useEpicLinearConfigQuery(epic.slug, {
    enabled: linearConnected && popoverOpen && view === "config",
  })

  const updateProjectMutation = useUpdateEpicLinearProjectMutation()
  const updateConfigMutation = useUpdateEpicLinearConfigMutation()

  if (!linearConnected) {
    return null
  }

  const hasProject = !!epic.linearProjectId
  const currentProject = projectsQuery.data?.find(
    (p) => p.id === epic.linearProjectId,
  )

  function handleClose() {
    setPopoverOpen(false)
    setView("projects")
    setLabelInput("")
    setLabelIsDirty(false)
  }

  async function handleSelectProject(project: LinearProject) {
    await updateProjectMutation.mutateAsync({
      epicSlug: epic.slug,
      linearProjectId: project.id,
    })
    setView("config")
  }

  function handleConfigureProject() {
    setView("config")
  }

  async function handleUnlink() {
    await updateProjectMutation.mutateAsync({
      epicSlug: epic.slug,
      linearProjectId: null,
    })
    handleClose()
  }

  async function handleSelectMilestone(milestone: LinearMilestone) {
    await updateConfigMutation.mutateAsync({
      epicSlug: epic.slug,
      milestoneId: milestone.id,
    })
  }

  async function handleClearMilestone() {
    const labelName =
      (labelIsDirty
        ? labelInput
        : (configQuery.data?.labelName ?? "")
      ).trim() || epic.slug
    await updateConfigMutation.mutateAsync({
      epicSlug: epic.slug,
      milestoneId: null,
      labelName,
    })
  }

  async function handleSetLabel() {
    const name =
      (labelIsDirty
        ? labelInput
        : (configQuery.data?.labelName ?? "")
      ).trim() || epic.slug
    await updateConfigMutation.mutateAsync({
      epicSlug: epic.slug,
      labelName: name,
      milestoneId: null,
    })
    setLabelInput(name)
    setLabelIsDirty(true)
  }

  const isSaving =
    updateProjectMutation.isPending || updateConfigMutation.isPending

  return (
    <div className="relative">
      <Button
        variant="ghost"
        size="sm"
        className={cn(
          "h-7 gap-1.5 px-2",
          !hasProject && "text-muted-foreground",
        )}
        onClick={() => setPopoverOpen((open) => !open)}
        aria-expanded={popoverOpen}
        title={
          hasProject
            ? `Linked to Linear project: ${currentProject?.name ?? epic.linearProjectId}`
            : "Link to Linear project"
        }
      >
        <LinearIcon className="size-3.5" />
        {hasProject ? (
          <span className="max-w-32 truncate">
            {currentProject?.name ?? "Project"}
          </span>
        ) : (
          <span>Link</span>
        )}
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
            {view === "projects" ? (
              <>
                <div className="mb-2 px-2 text-xs font-medium text-muted-foreground">
                  Linear Project
                </div>

                {projectsQuery.isLoading ? (
                  <div className="flex items-center justify-center py-4">
                    <Loader2 className="size-4 animate-spin text-muted-foreground" />
                  </div>
                ) : projectsQuery.isError ? (
                  <div className="px-2 py-2 text-xs text-destructive">
                    Failed to load projects
                  </div>
                ) : (
                  <div className="grid gap-0.5">
                    {projectsQuery.data?.map((project) => {
                      const isSelected = project.id === epic.linearProjectId
                      return (
                        <Button
                          key={project.id}
                          variant="ghost"
                          className="w-full justify-between"
                          onClick={() =>
                            isSelected
                              ? handleConfigureProject()
                              : void handleSelectProject(project)
                          }
                          disabledReason={isSaving ? "Saving..." : null}
                        >
                          <span className="flex items-center gap-1.5">
                            {isSelected ? <Check className="size-3.5" /> : null}
                            <span className="truncate">{project.name}</span>
                          </span>
                          {isSelected ? (
                            <ChevronRight className="size-3.5 text-muted-foreground" />
                          ) : null}
                        </Button>
                      )
                    })}

                    {hasProject ? (
                      <>
                        <div className="my-1 h-px bg-border/50" />
                        <Button
                          variant="ghost"
                          className="w-full justify-start text-muted-foreground"
                          onClick={() => void handleUnlink()}
                          disabledReason={isSaving ? "Saving..." : null}
                        >
                          <Unlink className="size-3.5" />
                          Unlink project
                        </Button>
                      </>
                    ) : null}
                  </div>
                )}
              </>
            ) : (
              <>
                <div className="mb-2 flex items-center gap-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-6 w-6 p-0"
                    onClick={() => setView("projects")}
                  >
                    <ArrowLeft className="size-3.5" />
                  </Button>
                  <span className="text-xs font-medium text-muted-foreground">
                    Sync Settings
                  </span>
                </div>

                {configQuery.isLoading || milestonesQuery.isLoading ? (
                  <div className="flex items-center justify-center py-4">
                    <Loader2 className="size-4 animate-spin text-muted-foreground" />
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <div className="flex items-center gap-1.5 text-xs font-medium">
                        <Tag className="size-3" />
                        <span>Label</span>
                        {configQuery.data?.syncMode === "label" ? (
                          <span className="rounded-full bg-foreground/10 px-1.5 py-0.5 text-[0.625rem] text-foreground/70">
                            Active
                          </span>
                        ) : null}
                      </div>
                      <div className="flex items-center gap-1.5">
                        <input
                          type="text"
                          className="h-7 flex-1 rounded-md bg-background/40 px-2 text-sm shadow-sm outline-none focus-visible:ring-2 focus-visible:ring-ring/30"
                          placeholder={epic.slug}
                          value={
                            labelIsDirty
                              ? labelInput
                              : (configQuery.data?.labelName ?? "")
                          }
                          onChange={(e) => {
                            setLabelIsDirty(true)
                            setLabelInput(e.target.value)
                          }}
                          disabled={isSaving}
                        />
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-7"
                          onClick={() => void handleSetLabel()}
                          disabledReason={isSaving ? "Saving..." : null}
                        >
                          {configQuery.data?.syncMode === "label" ? (
                            <Check className="size-3.5" />
                          ) : (
                            "Use"
                          )}
                        </Button>
                      </div>
                    </div>

                    <div className="h-px bg-border/50" />

                    <div className="space-y-2">
                      <div className="flex items-center gap-1.5 text-xs font-medium">
                        <Target className="size-3" />
                        <span>Milestone</span>
                        {configQuery.data?.syncMode === "milestone" ? (
                          <span className="rounded-full bg-foreground/10 px-1.5 py-0.5 text-[0.625rem] text-foreground/70">
                            Active
                          </span>
                        ) : null}
                      </div>

                      {milestonesQuery.data?.length === 0 ? (
                        <div className="text-xs text-muted-foreground">
                          No milestones in this project.
                        </div>
                      ) : (
                        <div className="grid gap-0.5">
                          {milestonesQuery.data?.map((milestone) => {
                            const isSelected =
                              milestone.id === configQuery.data?.milestoneId
                            return (
                              <Button
                                key={milestone.id}
                                variant="ghost"
                                size="sm"
                                className="h-7 w-full justify-start"
                                onClick={() =>
                                  void handleSelectMilestone(milestone)
                                }
                                disabledReason={isSaving ? "Saving..." : null}
                              >
                                {isSelected ? (
                                  <Check className="size-3.5" />
                                ) : null}
                                <span className="truncate">
                                  {milestone.name}
                                </span>
                              </Button>
                            )
                          })}
                          {configQuery.data?.milestoneId ? (
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-7 w-full justify-start text-muted-foreground"
                              onClick={() => void handleClearMilestone()}
                              disabledReason={isSaving ? "Saving..." : null}
                            >
                              <Unlink className="size-3" />
                              Use label instead
                            </Button>
                          ) : null}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </>
      ) : null}
    </div>
  )
}
