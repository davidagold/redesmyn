export const queryKeys = {
  epics: () => ["epics"] as const,
  epicGraph: (epic: number | string) => ["epics", epic, "graph"] as const,
  epicLinearConfig: (epic: string) =>
    ["epics", epic, "linear", "config"] as const,
  epicGithubRepoConfig: (epic: string) =>
    ["epics", epic, "github", "repo"] as const,
  hosts: () => ["hosts"] as const,
  daemons: () => ["daemons"] as const,
  orchestrationDefaults: () => ["config", "orchestrationDefaults"] as const,
  linearStatus: () => ["linear", "status"] as const,
  githubStatus: () => ["github", "status"] as const,
  githubPullRequest: (owner: string, repo: string, number: number) =>
    ["github", "pulls", owner, repo, number] as const,
  linearProjects: () => ["linear", "projects"] as const,
  linearMilestones: (projectId: string) =>
    ["linear", "projects", projectId, "milestones"] as const,
}
