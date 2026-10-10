// Feature 20 (Interactive Prediction Explorer). The full, unfiltered
// StudentExplorer — search, prediction/risk/confidence filters, sort, column
// selection, pagination and CSV export, all server-side (see
// app/services/student_service.py).

import { useEffect } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'

import { Alert, EmptyState, PageHeader, Skeleton } from '../../components/ui'
import { StudentExplorer } from '../../components/StudentExplorer'
import { useAsync } from '../../hooks/useAsync'
import { useWorkspace } from '../../hooks/useWorkspace'
import * as api from '../../services/api'

export function PredictionExplorerPage() {
  const { dataset } = useWorkspace()
  const { predictionId } = useParams()
  const navigate = useNavigate()

  const { data: history, loading, error } = useAsync(
    () => api.getPredictionHistory(dataset!.dataset_id),
    [dataset?.dataset_id],
    { enabled: Boolean(dataset) && !predictionId },
  )

  const batchRuns = history?.filter((h) => h.kind === 'batch') ?? []

  useEffect(() => {
    if (!predictionId && batchRuns.length > 0) {
      navigate(`/insights/explorer/${batchRuns[0].prediction_id}`, { replace: true })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [predictionId, batchRuns.length])

  if (!dataset) {
    return (
      <div className="space-y-6">
        <PageHeader eyebrow="Insights" title="Prediction explorer" />
        <EmptyState
          title="No dataset loaded"
          description="Upload a dataset, train a model and run a batch prediction to explore results here."
          action={<Link to="/upload" className="btn-primary">Upload a dataset</Link>}
        />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Insights"
        title="Prediction explorer"
        description="Search, filter and sort every scored student from a batch prediction run, and jump into an individual profile or comparison."
      />

      {error && <Alert tone="danger" title="Could not load prediction history">{error}</Alert>}
      {(loading || (!predictionId && batchRuns.length > 0)) && <Skeleton className="h-64 rounded-card" />}

      {!loading && !predictionId && batchRuns.length === 0 && (
        <EmptyState
          title="No batch predictions yet"
          description="Run a batch prediction from the Prediction page to explore results here."
          action={<Link to="/predict" className="btn-primary">Go to prediction</Link>}
        />
      )}

      {predictionId && <StudentExplorer predictionId={predictionId} />}
    </div>
  )
}
