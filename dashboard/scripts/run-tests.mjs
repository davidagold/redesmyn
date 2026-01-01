import { spawnSync } from "node:child_process"
import { readdirSync } from "node:fs"

const testFiles = readdirSync(new URL("../tests", import.meta.url))
  .filter((name) => name.endsWith(".test.ts"))
  .map((name) => `tests/${name}`)

if (!testFiles.length) {
  process.exit(0)
}

const supportsStripTypes =
  spawnSync(process.execPath, ["--experimental-strip-types", "-e", ""], {
    stdio: "ignore",
  }).status === 0

if (supportsStripTypes) {
  const result = spawnSync(
    process.execPath,
    ["--test", "--experimental-strip-types", ...testFiles],
    { stdio: "inherit" },
  )
  process.exit(result.status ?? 1)
}

const tsxBin = process.platform === "win32" ? "tsx.cmd" : "tsx"
const result = spawnSync(tsxBin, ["--test", ...testFiles], { stdio: "inherit" })
process.exit(result.status ?? 1)
