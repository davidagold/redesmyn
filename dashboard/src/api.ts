import type { components } from "@/api/v1"

export type ApiStatus = components["schemas"]["ApiStatusResponse"]
export type Epic = components["schemas"]["EpicResponse"]
export type EpicGraph = components["schemas"]["EpicGraphResponse"]
export type OrchestrationDefaults = components["schemas"]["OrchestrationDefaultsResponse"]
export type OrchestrationDefaultsUpdateRequest = components["schemas"]["OrchestrationDefaultsUpdateRequest"]
export type Host = components["schemas"]["HostResponse"]
export type DaemonPresence = components["schemas"]["DaemonPresenceResponse"]
export type AgentSession = components["schemas"]["AgentSessionResponse"]
export type LinearStatus = components["schemas"]["LinearStatusResponse"]
export type GitHubStatus = components["schemas"]["GitHubStatusResponse"]
export type GitHubIntegrationConfig = components["schemas"]["GitHubIntegrationConfigResponse"]
export type GitHubIntegrationConfigUpdateRequest = components["schemas"]["GitHubIntegrationConfigUpdateRequest"]
export type GitHubPullRequestOpenResponse = components["schemas"]["GitHubPullRequestOpenResponse"]
export type SyncStats = components["schemas"]["SyncStatsResponse"]
export type LinearPushStats = components["schemas"]["LinearPushStatsResponse"]
export type GithubRepo = components["schemas"]["GithubRepoResponse"]
export type EpicGithubRepoConfig = components["schemas"]["EpicGithubRepoConfigResponse"]
export type GitHubPullRequest = components["schemas"]["GitHubPullRequestResponse"]

export type LinearProject = {
  id: string
  name: string
  slug: string | null
}

export type LinearLabel = {
  id: string
  name: string
}

export type LinearMilestone = {
  id: string
  name: string
}

export type EpicLinearConfig = {
  syncMode: "label" | "milestone"
  labelId: string | null
  labelName: string | null
  milestoneId: string | null
  milestoneName: string | null
}
export type Task = components["schemas"]["TaskResponse"]
export type TaskAgentRestartRequest = components["schemas"]["TaskAgentRestartRequest"] & {
  prelude?: string | null
}
export type TaskAgentStartRequest = components["schemas"]["TaskAgentStartRequest"] & {
  prelude?: string | null
}
export type TaskAgentMessageRequest = components["schemas"]["TaskAgentMessageRequest"]
export type TaskAgentMessageConflictAction = components["schemas"]["TaskAgentMessageConflictAction"]
export type TaskAgentMessageResponse = components["schemas"]["TaskAgentMessageResponse"]
export type TaskAgentStartResponse = components["schemas"]["TaskAgentStartResponse"]
export type TaskAgentStopResponse = components["schemas"]["TaskAgentStopResponse"]
export type TaskAgentBulkRunRequest = components["schemas"]["TaskAgentBulkRunRequest"] & {
  prelude?: string | null
}
export type TaskAgentBulkRunResponse = components["schemas"]["TaskAgentBulkRunResponse"]
export type TaskAgentBulkActionRequest = components["schemas"]["TaskAgentBulkActionRequest"] & {
  prelude?: string | null
}
export type TaskAgentBulkActionResponse = components["schemas"]["TaskAgentBulkActionResponse"]

export type TaskMergeRequest = {
  runId?: string | null
  hostKey?: string | null
  cascade?: boolean
  scope?: "descendants" | "spine"
  restackMode?: "strict" | "merge_then_restack"
  dryRun?: boolean
  allowRunning?: boolean
  force?: boolean
}

export type TaskRestackRequest = {
  runId?: string | null
  hostKey?: string | null
  scope?: "descendants" | "spine"
  dryRun?: boolean
  allowRunning?: boolean
}

export type MergeRunResumeRequest = {
  allowRunning?: boolean
  hostKey?: string | null
}

export type MergeRunResumeResponse = {
  runId: string
  baseBranch?: string | null
}

export type MergeRunCancelRequest = {
  hostKey?: string | null
  abortGit?: boolean
}

export type MergeRunCancelResponse = {
  runId: string
  canceled: boolean
  abortedGit?: boolean
  detail?: string | null
}

export type TaskMergePlanStep = {
  kind: "rebase" | "mergeFf"
  nodeId: number | null
  taskId: number | null
  branchName: string
  worktreePath: string
  upstreamRef?: string | null
  baseBranch?: string | null
}

export type TaskMergeResponse = {
  runId: string
  dryRun: boolean
  baseBranch?: string | null
  steps: TaskMergePlanStep[]
}

export type TaskRestackResponse = {
  runId: string
  dryRun: boolean
  baseBranch?: string | null
  steps: TaskMergePlanStep[]
}

export type TaskAgentLogsResponse = {
  path: string
  text: string
  truncated: boolean
}

export class ApiHttpError extends Error {
  status: number
  detail: string | null
  code: string | null

  constructor(
    message: string,
    status: number,
    detail: string | null,
    code: string | null,
  ) {
    super(message)
    this.name = "ApiHttpError"
    this.status = status
    this.detail = detail
    this.code = code
  }
}

interface ApiErrorPayload {
  detail: string | null
  code: string | null
}

async function readErrorPayload(response: Response): Promise<ApiErrorPayload> {
  let detail: string | null = null
  let code: string | null = null
  try {
    const payloadRaw = (await response.json()) as unknown
    if (payloadRaw && typeof payloadRaw === "object") {
      const payload = payloadRaw as Record<string, unknown>
      if (typeof payload.detail === "string") {
        detail = payload.detail
      }
      if (typeof payload.code === "string") {
        code = payload.code
      }
    }
  } catch {
    // Ignore.
  }
  return { detail, code }
}

async function requestJson<T>(
  path: string,
  options: {
    method?: string
    body?: unknown
    signal?: AbortSignal
  } = {},
): Promise<T> {
  const method = options.method ?? "GET"
  const init: RequestInit = {
    method,
    signal: options.signal,
    headers: {
      Accept: "application/json",
    },
  }

  if (options.body !== undefined) {
    init.headers = {
      ...init.headers,
      "Content-Type": "application/json",
    }
    init.body = JSON.stringify(options.body)
  }

  const response = await fetch(path, init)
  if (!response.ok) {
    const payload = await readErrorPayload(response)
    const detail = payload.detail
    throw new ApiHttpError(
      `${method} ${path} failed (${response.status})${
        detail ? `: ${detail}` : ""
      }`,
      response.status,
      detail,
      payload.code,
    )
  }

  return response.json() as Promise<T>
}

export async function fetchStatus(options?: {
  signal?: AbortSignal
}): Promise<ApiStatus> {
  return requestJson("/v1/status", { signal: options?.signal })
}

export async function fetchEpics(options?: {
  signal?: AbortSignal
}): Promise<Epic[]> {
  return requestJson("/v1/epics", { signal: options?.signal })
}

export async function fetchEpicGraph(
  epic: string | number,
  options?: { signal?: AbortSignal },
): Promise<EpicGraph> {
  return requestJson(`/v1/epics/${epic}/graph`, { signal: options?.signal })
}

export async function fetchHosts(options?: {
  signal?: AbortSignal
}): Promise<Host[]> {
  return requestJson("/v1/hosts", { signal: options?.signal })
}

export async function fetchDaemons(options?: {
  signal?: AbortSignal
}): Promise<DaemonPresence[]> {
  return requestJson("/v1/daemons", { signal: options?.signal })
}

export async function fetchLinearStatus(options?: {
  signal?: AbortSignal
}): Promise<LinearStatus> {
  return requestJson("/v1/linear/status", { signal: options?.signal })
}

export async function fetchGitHubStatus(options?: {
  signal?: AbortSignal
}): Promise<GitHubStatus> {
  return requestJson("/v1/github/status", { signal: options?.signal })
}

export async function fetchGitHubPullRequest(
  owner: string,
  repo: string,
  number: number,
  options?: { signal?: AbortSignal },
): Promise<GitHubPullRequest> {
  return requestJson(`/v1/github/pulls/${owner}/${repo}/${number}`, {
    signal: options?.signal,
  })
}

export async function postLinearLogout(): Promise<LinearStatus> {
  return requestJson("/v1/linear/logout", { method: "POST" })
}

export async function postGitHubLogout(): Promise<GitHubStatus> {
  return requestJson("/v1/github/logout", { method: "POST" })
}

export async function updateGitHubIntegrationConfig(
  request: GitHubIntegrationConfigUpdateRequest,
): Promise<GitHubIntegrationConfig> {
  return requestJson("/v1/github/config", {
    method: "POST",
    body: request,
  })
}

export async function openTaskGithubPullRequest(
  taskId: number,
): Promise<GitHubPullRequestOpenResponse> {
  return requestJson(`/v1/tasks/${taskId}/github/pr/open`, { method: "POST" })
}

export async function postSyncFromLinear(epic: string): Promise<SyncStats> {
  return requestJson(`/v1/epics/${epic}/sync/from/linear`, { method: "POST" })
}

export async function postSyncToLinear(epic: string): Promise<LinearPushStats> {
  return requestJson(`/v1/epics/${epic}/sync/to/linear`, { method: "POST" })
}

export async function setTaskMergeReady(
  taskId: number,
  ready: boolean,
  options?: { scope?: "task" | "spine" },
): Promise<Task> {
  return requestJson(`/v1/tasks/${taskId}/merge-ready`, {
    method: "POST",
    body: { ready, ...(options?.scope ? { scope: options.scope } : {}) },
  })
}

export async function mergeTask(
  taskId: number,
  request: TaskMergeRequest,
): Promise<TaskMergeResponse> {
  return requestJson(`/v1/tasks/${taskId}/merge`, {
    method: "POST",
    body: request,
  })
}

export async function restackTask(
  taskId: number,
  request: TaskRestackRequest,
): Promise<TaskRestackResponse> {
  return requestJson(`/v1/tasks/${taskId}/restack`, {
    method: "POST",
    body: request,
  })
}

export async function resumeMergeRun(
  runId: string,
  request: MergeRunResumeRequest,
): Promise<MergeRunResumeResponse> {
  return requestJson(`/v1/merge-runs/${runId}/resume`, {
    method: "POST",
    body: request,
  })
}

export async function cancelMergeRun(
  runId: string,
  request: MergeRunCancelRequest,
): Promise<MergeRunCancelResponse> {
  return requestJson(`/v1/merge-runs/${runId}/cancel`, {
    method: "POST",
    body: request,
  })
}

export async function fetchOrchestrationDefaults(options?: {
  signal?: AbortSignal
}): Promise<OrchestrationDefaults> {
  return requestJson("/v1/config", { signal: options?.signal })
}

export async function updateOrchestrationDefaults(
  request: OrchestrationDefaultsUpdateRequest,
): Promise<OrchestrationDefaults> {
  return requestJson("/v1/config", {
    method: "POST",
    body: request,
  })
}

export async function startTaskAgent(
  taskId: number,
  request: TaskAgentStartRequest,
): Promise<TaskAgentStartResponse> {
  return requestJson(`/v1/tasks/${taskId}/agent/start`, {
    method: "POST",
    body: request,
  })
}

export async function runTaskAgentsBulk(
  request: TaskAgentBulkRunRequest,
): Promise<TaskAgentBulkRunResponse> {
  return requestJson("/v1/tasks/agent/run", { method: "POST", body: request })
}

export async function bulkTaskAgentActions(
  request: TaskAgentBulkActionRequest,
): Promise<TaskAgentBulkActionResponse> {
  return requestJson("/v1/tasks/agent/actions", {
    method: "POST",
    body: request,
  })
}

export async function stopTaskAgent(
  taskId: number,
): Promise<TaskAgentStopResponse> {
  return requestJson(`/v1/tasks/${taskId}/agent/stop`, { method: "POST" })
}

export async function postTaskAgentMessage(
  taskId: number,
  request: TaskAgentMessageRequest,
): Promise<TaskAgentMessageResponse> {
  return requestJson(`/v1/tasks/${taskId}/agent/message`, {
    method: "POST",
    body: request,
  })
}

export async function restartTaskAgent(
  taskId: number,
  request: TaskAgentRestartRequest = { detach: true },
): Promise<TaskAgentStartResponse> {
  return requestJson(`/v1/tasks/${taskId}/agent/restart`, {
    method: "POST",
    body: request,
  })
}

export async function fetchTaskAgentLogs(
  taskId: number,
  params: {
    lines?: number
    maxBytes?: number
  } = {},
): Promise<TaskAgentLogsResponse> {
  const url = new URL(`/v1/tasks/${taskId}/agent/logs`, window.location.origin)
  if (typeof params.lines === "number") {
    url.searchParams.set("lines", String(params.lines))
  }
  if (typeof params.maxBytes === "number") {
    url.searchParams.set("max_bytes", String(params.maxBytes))
  }
  return requestJson(url.pathname + url.search)
}

export async function fetchLinearProjects(options?: {
  signal?: AbortSignal
}): Promise<LinearProject[]> {
  return requestJson("/v1/linear/projects", { signal: options?.signal })
}

export async function updateEpicLinearProject(
  epicSlug: string,
  linearProjectId: string | null,
): Promise<Epic> {
  return requestJson(`/v1/epics/${epicSlug}/linear/project`, {
    method: "PATCH",
    body: { linear_project_id: linearProjectId },
  })
}

export async function fetchLinearMilestones(
  projectId: string,
  options?: { signal?: AbortSignal },
): Promise<LinearMilestone[]> {
  return requestJson(`/v1/linear/projects/${projectId}/milestones`, {
    signal: options?.signal,
  })
}

export async function fetchEpicLinearConfig(
  epicSlug: string,
  options?: { signal?: AbortSignal },
): Promise<EpicLinearConfig> {
  return requestJson(`/v1/epics/${epicSlug}/linear/config`, {
    signal: options?.signal,
  })
}

export async function fetchEpicGithubRepoConfig(
  epicSlug: string,
  options?: { signal?: AbortSignal },
): Promise<EpicGithubRepoConfig> {
  return requestJson(`/v1/epics/${epicSlug}/github/repo`, {
    signal: options?.signal,
  })
}

export async function updateEpicGithubRepo(
  epicSlug: string,
  repo: string | null,
): Promise<EpicGithubRepoConfig> {
  return requestJson(`/v1/epics/${epicSlug}/github/repo`, {
    method: "PATCH",
    body: { repo },
  })
}

export async function updateEpicLinearConfig(
  epicSlug: string,
  config: {
    labelId?: string | null
    labelName?: string | null
    milestoneId?: string | null
  },
): Promise<EpicLinearConfig> {
  return requestJson(`/v1/epics/${epicSlug}/linear/config`, {
    method: "PATCH",
    body: {
      label_id: config.labelId,
      label_name: config.labelName,
      milestone_id: config.milestoneId,
    },
  })
}
