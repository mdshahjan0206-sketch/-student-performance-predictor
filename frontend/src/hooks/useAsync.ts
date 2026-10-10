// A small loader for read-only requests: tracks loading/error/data, ignores
// results that arrive after the inputs changed, and never leaves a stale
// error on screen when a new request starts.

import { useCallback, useEffect, useRef, useState } from 'react'

import { ApiRequestError } from '../services/api'

interface AsyncState<T> {
  data: T | null
  loading: boolean
  error: string | null
  reload: () => void
}

export function useAsync<T>(
  loader: () => Promise<T>,
  deps: unknown[],
  options: { enabled?: boolean } = {},
): AsyncState<T> {
  const enabled = options.enabled ?? true
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(enabled)
  const [error, setError] = useState<string | null>(null)
  const [nonce, setNonce] = useState(0)
  const requestId = useRef(0)

  useEffect(() => {
    if (!enabled) {
      setLoading(false)
      return
    }

    const id = ++requestId.current
    setLoading(true)
    setError(null)

    loader()
      .then((result) => {
        if (id === requestId.current) {
          setData(result)
          setLoading(false)
        }
      })
      .catch((err: unknown) => {
        if (id !== requestId.current) return
        setError(err instanceof ApiRequestError ? err.message : 'Something went wrong loading this view.')
        setLoading(false)
      })
    // `loader` is recreated every render, so the caller's deps drive this.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, enabled, nonce])

  const reload = useCallback(() => setNonce((n) => n + 1), [])

  return { data, loading, error, reload }
}
