import { useEffect } from 'react'
import { useLocation } from 'react-router-dom'

/** Resets the window scroll position to the top whenever the route changes,
 *  so navigating to a new page never inherits the previous page's scroll
 *  position. Mount this once, inside the router, above the routed pages. */
export function ScrollToTop() {
  const { pathname } = useLocation()

  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: 'auto' })
  }, [pathname])

  return null
}
