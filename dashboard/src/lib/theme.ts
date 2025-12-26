export type ThemePreference = "dark" | "light" | "system"

export const THEME_PREFERENCE_KEY = "redesmyn-theme"

export function getStoredThemePreference(): ThemePreference | null {
  const raw = localStorage.getItem(THEME_PREFERENCE_KEY)
  if (raw === "dark" || raw === "light" || raw === "system") {
    return raw
  }
  return null
}

export function setStoredThemePreference(value: ThemePreference): void {
  localStorage.setItem(THEME_PREFERENCE_KEY, value)
}

export function prefersDarkMode(): boolean {
  return window.matchMedia("(prefers-color-scheme: dark)").matches
}

export function applyThemePreference(value: ThemePreference): void {
  const shouldBeDark =
    value === "dark" || (value === "system" && prefersDarkMode())
  document.documentElement.classList.toggle("dark", shouldBeDark)
}
