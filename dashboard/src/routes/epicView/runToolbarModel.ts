export type RunToolbarTaskState = "todo" | "in_progress" | "blocked" | "done"

export type RunToolbarTask = {
  id: number
  branchName: string | null
  state: RunToolbarTaskState
  stackInSync?: boolean | null
}

export type RunToolbarAgentSessionStatus = "running" | "blocked" | "error" | "stopped"

export type RunToolbarAgentSession = {
  status: RunToolbarAgentSessionStatus
}

export type RunToolbarActionTargets = {
  start: number[]
  restart: number[]
  stop: number[]
}

export type RunToolbarSummary = {
  eligible: number
  running: number
  blocked: number
  failed: number
  outOfSync: number
}

export type RunToolbarBuckets = {
  eligible: number[]
  running: number[]
  blocked: number[]
  failed: number[]
  outOfSync: number[]
}

export type RunToolbarModel = {
  summary: RunToolbarSummary | null
  buckets: RunToolbarBuckets | null
  actionTargets: {
    all: RunToolbarActionTargets
    selected: RunToolbarActionTargets
  }
  harnessRequired: {
    all: boolean
    selected: boolean
  }
}

export function computeRunToolbarModel(options: {
  tasks: RunToolbarTask[] | null
  tasksById: Map<number, RunToolbarTask>
  agentSessionsByNodeId: Map<number, RunToolbarAgentSession>
  selectedNodeIds: ReadonlySet<number>
  stackProjectionsFresh: boolean
}): RunToolbarModel {
  const emptyTargets: RunToolbarActionTargets = {
    start: [],
    restart: [],
    stop: [],
  }

  const tasks = options.tasks
  if (!tasks) {
    return {
      summary: null,
      buckets: null,
      actionTargets: { all: emptyTargets, selected: emptyTargets },
      harnessRequired: { all: false, selected: false },
    }
  }

  const allStart = new Set<number>()
  const allRestart = new Set<number>()
  const allStop = new Set<number>()

  let eligible = 0
  let running = 0
  let blocked = 0
  let failed = 0
  let outOfSync = 0

  const bucketEligible = new Set<number>()
  const bucketRunning = new Set<number>()
  const bucketBlocked = new Set<number>()
  const bucketFailed = new Set<number>()
  const bucketOutOfSync = new Set<number>()

  for (const task of tasks) {
    if (task.branchName === null) {
      continue
    }
    if (task.state === "done") {
      continue
    }

    if (task.stackInSync === false) {
      bucketOutOfSync.add(task.id)
    }

    if (task.state === "blocked") {
      blocked += 1
      bucketBlocked.add(task.id)
      continue
    }

    eligible += 1
    bucketEligible.add(task.id)

    if (task.stackInSync === false) {
      outOfSync += 1
    }

    const session = options.agentSessionsByNodeId.get(task.id) ?? null
    const status = session?.status ?? null

    if (!session || status === "stopped") {
      allStart.add(task.id)
    } else if (status === "error") {
      allRestart.add(task.id)
    }

    if (status === "running") {
      running += 1
      bucketRunning.add(task.id)
      allStop.add(task.id)
    } else if (status === "blocked") {
      blocked += 1
      bucketBlocked.add(task.id)
      allStop.add(task.id)
    } else if (status === "error") {
      failed += 1
      bucketFailed.add(task.id)
    }
  }

  const selectedStart = new Set<number>()
  const selectedRestart = new Set<number>()
  const selectedStop = new Set<number>()

  for (const selectedNodeId of options.selectedNodeIds) {
    const task = options.tasksById.get(selectedNodeId) ?? null
    if (!task || task.branchName === null) {
      continue
    }
    if (task.state === "blocked" || task.state === "done") {
      continue
    }
    const session = options.agentSessionsByNodeId.get(task.id) ?? null
    const status = session?.status ?? null

    if (!session || status === "stopped") {
      selectedStart.add(task.id)
    } else if (status === "error") {
      selectedRestart.add(task.id)
    }

    if (status === "running" || status === "blocked") {
      selectedStop.add(task.id)
    }
  }

  const actionTargets = {
    all: {
      start: [...allStart].sort((a, b) => a - b),
      restart: [...allRestart].sort((a, b) => a - b),
      stop: [...allStop].sort((a, b) => a - b),
    },
    selected: {
      start: [...selectedStart].sort((a, b) => a - b),
      restart: [...selectedRestart].sort((a, b) => a - b),
      stop: [...selectedStop].sort((a, b) => a - b),
    },
  }

  return {
    summary: { eligible, running, blocked, failed, outOfSync },
    buckets: {
      eligible: [...bucketEligible].sort((a, b) => a - b),
      running: [...bucketRunning].sort((a, b) => a - b),
      blocked: [...bucketBlocked].sort((a, b) => a - b),
      failed: [...bucketFailed].sort((a, b) => a - b),
      outOfSync: [...bucketOutOfSync].sort((a, b) => a - b),
    },
    actionTargets,
    harnessRequired: {
      all: actionTargets.all.start.length > 0,
      selected: actionTargets.selected.start.length > 0,
    },
  }
}
