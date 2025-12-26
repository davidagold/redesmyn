import { useEffect, useState } from "react"
import {
  applyThemePreference,
  getStoredThemePreference,
  setStoredThemePreference,
  type ThemePreference,
} from "@/lib/theme"

export function useTheme() {
  const [theme, setTheme] = useState<ThemePreference>(
    () => getStoredThemePreference() ?? "dark",
  )

  useEffect(() => {
    applyThemePreference(theme)
    setStoredThemePreference(theme)
  }, [theme])

  useEffect(() => {
    if (theme !== "system") {
      return
    }
    const media = window.matchMedia("(prefers-color-scheme: dark)")
    const handler = () => applyThemePreference(theme)
    media.addEventListener?.("change", handler)
    media.addListener?.(handler)
    return () => {
      media.removeEventListener?.("change", handler)
      media.removeListener?.(handler)
    }
  }, [theme])

  function cycleTheme() {
    setTheme(theme === "dark" ? "light" : theme === "light" ? "system" : "dark")
  }

  return { theme, setTheme, cycleTheme }
}
