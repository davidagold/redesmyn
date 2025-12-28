export async function copyToClipboard(text: string) {
  if (
    typeof navigator !== "undefined" &&
    navigator.clipboard &&
    typeof navigator.clipboard.writeText === "function"
  ) {
    await navigator.clipboard.writeText(text)
    return
  }

  if (typeof window !== "undefined") {
    window.prompt("Copy to clipboard:", text)
  }
}
