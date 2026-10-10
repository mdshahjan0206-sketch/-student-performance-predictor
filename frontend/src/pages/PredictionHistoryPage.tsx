// Feature 7 (Prediction History). Lightweight metadata only — the full
// result frame stays where it already lives (backend/outputs/), and is only
// pulled down when the user opens or downloads a specific run.

import { Link } from 'react-router-dom'

import { Alert, Badge, Card, EmptyState, PageHeader, Skeleton } from '../components/ui'
import { useAsync } from '../hooks/useAsync'
import { useWorkspace } from '../hooks/useWorkspace'
import * as api from '../services/api'

export function PredictionHistoryPage() {
  const { dataset } = useWorkspace()

  const { data: history, loading, error } = useAsync(
    () => api.getPredictionHistory(dataset!.dataset_id),
    [dataset?.dataset_id],
    { enabled: Boolean(dataset) },
  )

  if (!dataset) {
    return (
      <div className="space-y-6">
        <PageHeader eyebrow="Results" title="Prediction history" />
        <EmptyState
          title="No dataset loaded"
          description="Upload a dataset and run a batch prediction to see history here."
          action={<Link to="/upload" className="btn-primary">Upload a dataset</Link>}
        />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Results"
        title="Prediction history"
        description="Every batch prediction run against the active dataset, most recent first."
      />

      {error && <Alert tone="danger" title="Could not load prediction history">{error}</Alert>}
      {loading && <Skeleton className="h-48 rounded-card" />}

      {!loading && history && history.length === 0 && (
        <EmptyState
          title="No batch predictions yet"
          description="Run a batch prediction from the Prediction page to see it listed here."
          action={<Link to="/predict" className="btn-primary">Go to prediction</Link>}
        />
      )}

      {history && history.length > 0 && (
        <div className="space-y-2">
          {history.map((entry) => (
            <Card key={entry.prediction_id} className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="text-[14px] font-semibold text-ink">
                  {entry.model_choice ?? 'Model'} <span className="text-muted font-normal">on {entry.target_column ?? '—'}</span>
                </p>
                <p className="text-[12px] text-muted mt-0.5">
                  {new Date(entry.created_at * 1000).toLocaleString()} · {entry.row_count.toLocaleString()} rows scored
                  {entry.warnings.length > 0 && ` · ${entry.warnings.length} warning(s)`}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <Badge tone={entry.has_confidence ? 'brand' : 'neutral'}>{entry.problem_type}</Badge>
                {entry.kind === 'batch' && (
                  <>
                    <Link className="btn-secondary h-8 px-3 text-[12px]" to={`/insights/explorer/${entry.prediction_id}`}>
                      Explore
                    </Link>
                    <Link className="btn-secondary h-8 px-3 text-[12px]" to={`/insights/at-risk/${entry.prediction_id}`}>
                      At-risk
                    </Link>
                  </>
                )}
                <a
                  className="btn-secondary h-8 px-3 text-[12px]"
                  href={api.predictionDownloadUrl(entry.prediction_id)}
                  download
                >
                  Download CSV
                </a>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
