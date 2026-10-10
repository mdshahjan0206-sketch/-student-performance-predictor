// Feature 16 (Performance / Processing-Time Monitor). Every row here is a
// real, measured duration logged by app.core.timing.track() as an operation
// actually ran — see backend/app/core/timing.py. Nothing is benchmarked or
// faked; an empty log just means nothing has run yet in this server session.

import { Alert, Badge, Card, EmptyState, PageHeader, Skeleton } from '../components/ui'
import { useAsync } from '../hooks/useAsync'
import * as api from '../services/api'

export function PerformancePage() {
  const { data: entries, loading, error, reload } = useAsync(() => api.getPerformanceLog(100), [])

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Settings"
        title="Performance monitor"
        description="Actual measured durations for operations run against this backend in the current server session."
        action={
          <button className="btn-secondary" onClick={reload}>
            Refresh
          </button>
        }
      />

      {error && <Alert tone="danger" title="Could not load the performance log">{error}</Alert>}
      {loading && <Skeleton className="h-64 rounded-card" />}

      {!loading && entries && entries.length === 0 && (
        <EmptyState
          title="Nothing measured yet"
          description="Upload a dataset, run analysis, train a model or make a prediction — each operation is timed and will show up here."
        />
      )}

      {entries && entries.length > 0 && (
        <Card>
          <div className="table-scroll">
            <table className="w-full text-[13px]">
              <thead>
                <tr className="text-left text-muted border-b border-line">
                  <th className="py-2 pr-4 font-medium">Operation</th>
                  <th className="py-2 px-3 font-medium">Duration</th>
                  <th className="py-2 px-3 font-medium">Status</th>
                  <th className="py-2 pl-3 font-medium">When</th>
                </tr>
              </thead>
              <tbody>
                {entries.map((entry, index) => (
                  <tr key={`${entry.timestamp}-${index}`} className="border-b border-line last:border-0">
                    <td className="py-2.5 pr-4 font-semibold text-ink">{entry.operation}</td>
                    <td className="py-2.5 px-3 tabular-nums text-ink">{entry.duration.toFixed(3)}s</td>
                    <td className="py-2.5 px-3">
                      <Badge tone={entry.status === 'success' ? 'brand' : 'danger'}>{entry.status}</Badge>
                    </td>
                    <td className="py-2.5 pl-3 text-muted">{new Date(entry.timestamp * 1000).toLocaleTimeString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  )
}
