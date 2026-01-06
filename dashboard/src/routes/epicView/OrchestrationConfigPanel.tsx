import { useCallback, useEffect, useMemo, useState } from "react"
import { SlidePanel } from "@/components/ui/slide-panel"
import { Button } from "@/components/ui/button"
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import { Switch } from "@/components/ui/switch"
import { updateOrchestrationDefaults, type OrchestrationDefaults } from "@/api"

export function OrchestrationConfigPanel({
  open,
  defaults,
  defaultsLoading,
  onClose,
  onRefresh,
}: {
  open: boolean
  defaults: OrchestrationDefaults | null
  defaultsLoading: boolean
  onClose: () => void
  onRefresh: () => Promise<void>
}) {
  const [configPending, setConfigPending] = useState(false)
  const [configError, setConfigError] = useState<string | null>(null)
  const [configNotice, setConfigNotice] = useState<string | null>(null)
  const [configHarness, setConfigHarness] = useState("")
  const [configAgentKind, setConfigAgentKind] =
    useState<"auto" | "generic" | "codex" | "claude_code">("auto")
  const [configDetach, setConfigDetach] = useState(true)
  const [configSandboxType, setConfigSandboxType] =
    useState<"none" | "worktree">("none")
  const [configSandboxNetwork, setConfigSandboxNetwork] =
    useState<"allow" | "deny">("allow")
  const [configPrelude, setConfigPrelude] = useState("")
  const [configSendPrelude, setConfigSendPrelude] = useState(true)
  const [configSubmitPrelude, setConfigSubmitPrelude] = useState(true)
  const [defaultPreludeOpen, setDefaultPreludeOpen] = useState(false)

	  const resetConfigFields = useCallback(() => {
	    setConfigError(null)
	    setConfigNotice(null)
	    setDefaultPreludeOpen(false)
	    setConfigHarness(defaults?.harness.command ?? "")
	    setConfigAgentKind(defaults?.harness.agentKind ?? "auto")
	    setConfigDetach(defaults?.harness.detach ?? true)
	    setConfigSandboxType(defaults?.sandbox.type ?? "none")
	    setConfigSandboxNetwork(defaults?.sandbox.network ?? "allow")
	    setConfigPrelude(defaults?.harness.prelude ?? "")
	    setConfigSendPrelude(defaults?.harness.sendPrelude ?? true)
	    setConfigSubmitPrelude(defaults?.harness.submitPrelude ?? true)
	  }, [defaults])

  useEffect(() => {
    if (!open) {
      return
    }
    resetConfigFields()
  }, [open, resetConfigFields])

	  const configDirty = useMemo(() => {
	    const currentCommand = defaults?.harness.command ?? ""
	    const currentAgentKind = defaults?.harness.agentKind ?? "auto"
	    const currentDetach = defaults?.harness.detach ?? true
	    const currentSandboxType = defaults?.sandbox.type ?? "none"
	    const currentSandboxNetwork = defaults?.sandbox.network ?? "allow"
	    const currentPrelude = defaults?.harness.prelude ?? ""
	    const currentSendPrelude = defaults?.harness.sendPrelude ?? true
	    const currentSubmitPrelude = defaults?.harness.submitPrelude ?? true
	    return (
	      configHarness !== currentCommand ||
	      configAgentKind !== currentAgentKind ||
	      configDetach !== currentDetach ||
	      configSandboxType !== currentSandboxType ||
	      configSandboxNetwork !== currentSandboxNetwork ||
	      configPrelude !== currentPrelude ||
      configSendPrelude !== currentSendPrelude ||
      configSubmitPrelude !== currentSubmitPrelude
    )
	  }, [
	    configAgentKind,
	    configDetach,
	    configHarness,
	    configPrelude,
	    configSandboxNetwork,
    configSandboxType,
    configSendPrelude,
    configSubmitPrelude,
    defaults,
  ])

  async function handleSaveConfig() {
    if (configPending || !configDirty) {
      return
    }
    setConfigPending(true)
    setConfigError(null)
    setConfigNotice(null)
	    try {
	      await updateOrchestrationDefaults({
	        harness: {
	          command: configHarness.trim() ? configHarness.trim() : null,
	          agentKind: configAgentKind,
	          detach: configDetach,
	          prelude: configPrelude.trim() ? configPrelude : null,
	          sendPrelude: configSendPrelude,
	          submitPrelude: configSubmitPrelude,
        },
        sandbox: {
          type: configSandboxType,
          network: configSandboxNetwork,
        },
      })
      await onRefresh()
      setConfigNotice("Saved")
    } catch (e) {
      setConfigError(e instanceof Error ? e.message : String(e))
    } finally {
      setConfigPending(false)
    }
  }

  return (
    <>
      {open ? (
        <button
          type="button"
          aria-label="Close configuration panel"
          className="absolute inset-0 z-20 cursor-default bg-transparent"
          onClick={onClose}
        />
      ) : null}

      <SlidePanel
        open={open}
        side="left"
        className="z-30 w-[28rem] border-border/60 bg-background/80 backdrop-blur"
      >
        <div className="relative grid gap-4 p-4">
          <div className="sticky top-4 z-20 h-0 pointer-events-none">
            <div className="flex justify-end">
              <div className="inline-flex h-6 w-fit overflow-hidden rounded-md border border-border/60 bg-background/40 shadow-sm backdrop-blur pointer-events-auto">
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-full rounded-none border-0 leading-none"
                  onClick={() => void handleSaveConfig()}
                  disabledReason={
                    configPending
                      ? "Saving…"
                      : !configDirty
                        ? "No changes"
                        : null
                  }
                >
                  Save
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-full rounded-none border-0 border-l leading-none"
                  onClick={resetConfigFields}
                  disabledReason={
                    configPending
                      ? "Saving…"
                      : !defaults
                        ? "Defaults not loaded"
                        : !configDirty
                          ? "No changes"
                          : null
                  }
                >
                  Reset
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-full rounded-none border-0 border-l leading-none"
                  onClick={onClose}
                >
                  Close
                </Button>
              </div>
            </div>
          </div>

          <div className="pr-24">
            <div className="text-sm font-medium">Configure</div>
            <div className="mt-1.5 text-xs text-muted-foreground">
              Updates `config.toml` (repo scope).
            </div>
            {configNotice ? (
              <div className="mt-1.5 text-xs text-muted-foreground">
                {configNotice}
              </div>
            ) : null}
          </div>

          {configError ? (
            <div className="text-xs text-destructive">{configError}</div>
          ) : null}

          <Accordion
            multiple
            defaultValue={["harness", "prelude"]}
            className="border-0"
          >
            <AccordionItem value="harness">
              <AccordionTrigger>Harness</AccordionTrigger>
              <AccordionContent className="grid gap-4 pt-3">
	                <div className="grid gap-1">
	                  <div className="text-xs text-muted-foreground">
	                    Harness command
	                  </div>
	                  <input
                    className="h-7 rounded-md border bg-background/40 px-2 text-xs text-foreground shadow-sm outline-none focus-visible:ring-2 focus-visible:ring-ring/30"
                    value={configHarness}
                    onChange={(e) => setConfigHarness(e.target.value)}
                    placeholder={defaults?.harness.command ?? "codex"}
                    disabled={configPending}
                  />
	                  <div className="mt-1.5 text-xs text-muted-foreground">
	                    Shell command used to start the harness inside each task’s
	                    worktree (e.g. <span className="font-mono">codex</span>).
	                  </div>
	                </div>

	                <div className="grid gap-1">
	                  <div className="text-xs text-muted-foreground">
	                    Agent kind
	                  </div>
	                  <select
	                    className="h-7 rounded-md border bg-background/40 px-2 text-xs text-foreground shadow-sm outline-none focus-visible:ring-2 focus-visible:ring-ring/30"
	                    value={configAgentKind}
	                    onChange={(e) =>
	                      setConfigAgentKind(
	                        e.target.value as typeof configAgentKind,
	                      )
	                    }
	                    disabled={configPending}
	                  >
	                    <option value="auto">Auto</option>
	                    <option value="generic">Generic</option>
	                    <option value="codex">Codex</option>
	                    <option value="claude_code">Claude Code</option>
	                  </select>
	                  <div className="text-xs text-muted-foreground">
	                    {configAgentKind === "generic"
	                      ? "Generic works with any command, but advanced features are disabled."
	                      : "Auto infers the agent from your command; switch if wrong."}
	                  </div>
	                </div>

                <div className="grid gap-1">
                  <div className="flex items-center justify-between gap-3">
                    <div className="text-xs text-muted-foreground">
                      Run mode
                    </div>
                    {defaultsLoading ? (
                      <span className="text-xs text-muted-foreground">
                        loading…
                      </span>
                    ) : null}
                  </div>
                  <div className="inline-flex h-6 w-fit overflow-hidden rounded-md border border-border/60">
                    <Button
                      variant={configDetach ? "secondary" : "ghost"}
                      size="sm"
                      className="h-full rounded-none border-0 leading-none"
                      onClick={() => setConfigDetach(true)}
                      disabledReason={configPending ? "Saving…" : null}
                    >
                      Detached
                    </Button>
                    <Button
                      variant={!configDetach ? "secondary" : "ghost"}
                      size="sm"
                      className="h-full rounded-none border-0 border-l leading-none"
                      onClick={() => setConfigDetach(false)}
                      disabledReason={configPending ? "Saving…" : null}
                    >
                      Foreground
                    </Button>
                  </div>
                  <div className="mt-1.5 text-xs text-muted-foreground">
                    Detached runs in a tmux session; foreground runs in your
                    current terminal.
                  </div>
                </div>

                <div className="grid gap-1">
                  <div className="text-xs text-muted-foreground">Sandbox</div>
                  <div className="grid gap-2">
                    <div className="flex items-center justify-between gap-3">
                      <div className="text-xs text-foreground">
                        Worktree sandbox
                      </div>
                      <Switch
                        checked={configSandboxType === "worktree"}
                        onCheckedChange={(checked) => {
                          setConfigSandboxType(checked ? "worktree" : "none")
                          if (!checked) {
                            setConfigSandboxNetwork("allow")
                          }
                        }}
                        disabledReason={configPending ? "Saving…" : null}
                      />
                    </div>
                    <div className="flex items-center justify-between gap-3">
                      <div className="text-xs text-foreground">
                        Deny network
                      </div>
                      <Switch
                        checked={configSandboxNetwork === "deny"}
                        onCheckedChange={(checked) =>
                          setConfigSandboxNetwork(checked ? "deny" : "allow")
                        }
                        disabledReason={
                          configPending
                            ? "Saving…"
                            : configSandboxType !== "worktree"
                              ? "Enable sandbox first"
                              : null
                        }
                      />
                    </div>
                  </div>
                  <div className="mt-1.5 text-xs text-muted-foreground">
                    Restricts agent writes to the task worktree and Redesmyn
                    state. Enable “Deny network” to force offline operation.
                  </div>
                </div>
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="prelude">
              <AccordionTrigger>Prelude</AccordionTrigger>
              <AccordionContent className="grid gap-4 pt-3">
                <div className="grid gap-1">
                  <div className="flex items-center justify-between gap-3">
                    <div className="text-xs text-muted-foreground">
                      Agent prelude
                    </div>
                    <Popover
                      open={defaultPreludeOpen}
                      onOpenChange={setDefaultPreludeOpen}
                    >
                      <PopoverTrigger
                        render={(triggerProps) => {
                          return (
                            <Button
                              variant="ghost"
                              size="xs"
                              className="h-5 px-2"
                              disabledReason={
                                defaultsLoading
                                  ? "Loading…"
                                  : defaults?.harness.builtInPreludeTemplate
                                    ? null
                                    : "Default prelude unavailable"
                              }
                              {...triggerProps}
                            >
                              Show default
                            </Button>
                          )
                        }}
                      />
                      <PopoverContent className="w-[24rem]">
                        <div className="flex items-center justify-between gap-3">
                          <div className="text-xs font-medium text-foreground">
                            Default prelude
                          </div>
                          <Button
                            variant="outline"
                            size="sm"
                            className="h-6"
                            disabledReason={
                              defaults?.harness.builtInPreludeTemplate
                                ? null
                                : "Default prelude unavailable"
                            }
                            onClick={() => {
                              const template =
                                defaults?.harness.builtInPreludeTemplate ?? ""
                              setConfigPrelude(template)
                              setDefaultPreludeOpen(false)
                            }}
                          >
                            Fill as starting point
                          </Button>
                        </div>
                        <pre className="mt-2 max-h-60 overflow-auto whitespace-pre-wrap rounded-md border border-border/60 bg-background/30 p-2 font-mono text-[0.625rem] text-foreground/80">
                          {defaults?.harness.builtInPreludeTemplate ?? ""}
                        </pre>
                      </PopoverContent>
                    </Popover>
                  </div>
                  <textarea
                    className="min-h-[10rem] resize-y rounded-md border bg-background/40 px-2 py-2 text-xs text-foreground shadow-sm outline-none focus-visible:ring-2 focus-visible:ring-ring/30"
                    value={configPrelude}
                    onChange={(e) => setConfigPrelude(e.target.value)}
                    placeholder="Optional. Leave blank to use the built-in prelude."
                    disabled={configPending}
                  />
                  <div className="mt-1.5 text-xs text-muted-foreground">
                    Sent to the agent right after the harness starts. Use it to
                    point the agent at relevant docs and guidance.
                  </div>
                </div>

                <div className="grid gap-1">
                  <div className="text-xs text-muted-foreground">
                    Prelude delivery
                  </div>
                  <div className="grid gap-2">
                    <div className="flex items-center justify-between gap-3">
                      <div className="text-xs text-foreground">
                        Auto-send prelude
                      </div>
                      <Switch
                        checked={configSendPrelude}
                        onCheckedChange={(checked) => {
                          setConfigSendPrelude(checked)
                          if (!checked) {
                            setConfigSubmitPrelude(false)
                          }
                        }}
                        disabledReason={configPending ? "Saving…" : null}
                      />
                    </div>
                    <div className="flex items-center justify-between gap-3">
                      <div className="text-xs text-foreground">
                        Press Enter to submit
                      </div>
                      <Switch
                        checked={configSubmitPrelude}
                        onCheckedChange={setConfigSubmitPrelude}
                        disabledReason={
                          configPending
                            ? "Saving…"
                            : !configSendPrelude
                              ? "Enable Auto-send first"
                              : null
                        }
                      />
                    </div>
                  </div>
                  <div className="mt-1.5 text-xs text-muted-foreground">
                    Auto-send types the prelude into the harness. Press Enter
                    submits it so the agent starts working immediately.
                  </div>
                </div>

                <div className="rounded-md border border-border/60 bg-background/30 p-2 text-xs">
                  <div className="text-xs text-muted-foreground">
                    Available placeholders
                  </div>
                  <div className="mt-1 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
                    <div className="font-mono text-foreground/80">{`{task_id}`}</div>
                    <div className="text-muted-foreground">
                      Task numeric id.
                    </div>
                    <div className="font-mono text-foreground/80">{`{task_title}`}</div>
                    <div className="text-muted-foreground">Task title.</div>
                    <div className="font-mono text-foreground/80">{`{task_doc}`}</div>
                    <div className="text-muted-foreground">
                      Task README path (if available).
                    </div>
                    <div className="font-mono text-foreground/80">{`{epic_slug}`}</div>
                    <div className="text-muted-foreground">Epic slug.</div>
                    <div className="font-mono text-foreground/80">{`{epic_readme}`}</div>
                    <div className="text-muted-foreground">
                      Epic README path.
                    </div>
                    <div className="font-mono text-foreground/80">{`{branch}`}</div>
                    <div className="text-muted-foreground">
                      Branch name for the task.
                    </div>
                    <div className="font-mono text-foreground/80">{`{worktree}`}</div>
                    <div className="text-muted-foreground">
                      Absolute path to the task worktree.
                    </div>
                  </div>
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </div>
      </SlidePanel>
    </>
  )
}
