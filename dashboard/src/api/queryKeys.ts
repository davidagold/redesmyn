export const queryKeys = {
  epics: () => ["epics"] as const,
  epicGraph: (epic: number | string) => ["epics", epic, "graph"] as const,
  hosts: () => ["hosts"] as const,
  daemons: () => ["daemons"] as const,
  orchestrationDefaults: () => ["config", "orchestrationDefaults"] as const,
  linearStatus: () => ["linear", "status"] as const,
}
