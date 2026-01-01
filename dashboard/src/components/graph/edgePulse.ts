export type EdgePulseKind = "merge" | "rebase"

export type EdgePulseData = {
  kind: EdgePulseKind
  token: number
}

export function pulseDurationMs(kind: EdgePulseKind) {
  return kind === "merge" ? 560 : 680
}
