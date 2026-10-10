// Dark/light theme (feature 14). Preference is stored in localStorage and
// applied as a `data-theme` attribute on <html>, which the CSS variables in
// index.css key off. Charts read the same variables (see charts/Plot.tsx),
// so switching themes re-colours Plotly output too, not just the chrome.

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'

export type Theme = 'light' | 'dark'

const THEME_KEY = 'edupredict.theme'

function readStoredTheme(): Theme {
  // Light (mint green) is the primary theme: it is the default until the
  // user explicitly switches, regardless of the OS preference.
  if (typeof window === 'undefined') return 'light'
  try {
    const stored = window.localStorage.getItem(THEME_KEY)
    if (stored === 'light' || stored === 'dark') return stored
  } catch {
    // storage unavailable: fall through to the default
  }
  return 'light'
}

interface ThemeValue {
  theme: Theme
  toggle: () => void
  setTheme: (theme: Theme) => void
}

const ThemeContext = createContext<ThemeValue | null>(null)

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(readStoredTheme)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    document.documentElement.style.colorScheme = theme
    window.localStorage.setItem(THEME_KEY, theme)
  }, [theme])

  const setTheme = useCallback((next: Theme) => setThemeState(next), [])
  const toggle = useCallback(() => setThemeState((prev) => (prev === 'light' ? 'dark' : 'light')), [])

  const value = useMemo<ThemeValue>(() => ({ theme, toggle, setTheme }), [theme, toggle, setTheme])

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export function useTheme(): ThemeValue {
  const value = useContext(ThemeContext)
  if (!value) throw new Error('useTheme must be used inside ThemeProvider')
  return value
}
