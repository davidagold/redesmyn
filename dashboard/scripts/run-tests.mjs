import { spawnSync } from "node:child_process"
import { readdirSync } from "node:fs"

function parseNodeVersion(version) {
  const [major, minor, patch] = version.replace(/^v/, "").split(".").map(Number)
  return {
    major: Number.isFinite(major) ? major : 0,
    minor: Number.isFinite(minor) ? minor : 0,
    patch: Number.isFinite(patch) ? patch : 0,
  }
}

function nodeGte(a, b) {
  if (a.major !== b.major) return a.major > b.major
  if (a.minor !== b.minor) return a.minor > b.minor
  return a.patch >= b.patch
}

const testFiles = readdirSync(new URL("../tests", import.meta.url))
  .filter((name) => name.endsWith(".test.ts"))
  .map((name) => `tests/${name}`)

if (testFiles.length === 0) {
  process.exit(0)
}

const nodeVersion = parseNodeVersion(process.version)
const supportsStripTypes = nodeGte(nodeVersion, {
  major: 22,
  minor: 12,
  patch: 0,
})

if (supportsStripTypes) {
  const res = spawnSync(
    process.execPath,
    ["--test", "--experimental-strip-types", ...testFiles],
    { stdio: "inherit" },
  )
  process.exit(res.status ?? 1)
}

const res = spawnSync("tsx", ["--test", ...testFiles], { stdio: "inherit" })
process.exit(res.status ?? 1)
