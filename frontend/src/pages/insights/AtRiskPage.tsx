// Feature 18 (At-Risk Student Detection). A preset view of StudentExplorer
// (feature 20) with the risk filter locked to High + Medium — the backend's
// risk derivation (app/ml/risk.py) is the single source of truth here, this
// page just opens onto the students it already flagged.

import { useEffect } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'

import { Alert, EmptyState, PageHeader, Skeleton } from '../../components/ui'
import { StudentExplorer } from '../../components/StudentExplorer'
import { useAsync } from '../../hooks/useAsync'
import { useWorkspace } from '../../hooks/useWorkspace'
import * as api from '../../services/api'

export function AtRiskPage() {
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
      navigate(`/insights/at-risk/${batchRuns[0].prediction_id}`, { replace: true })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [predictionId, batchRuns.length])

  if (!dataset) {
    return (
      <div className="space-y-6">
        <PageHeader eyebrow="Insights" title="At-risk students" />
        <EmptyState
          title="No dataset loaded"
          description="Upload a dataset, train a model and run a batch prediction to see at-risk students here."
          action={<Link to="/upload" className="btn-primary">Upload a dataset</Link>}
        />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Insights"
        title="At-risk students"
        description="Students the current model predicts are failing or borderline, ranked by how confident that prediction is."
      />

      {error && <Alert tone="danger" title="Could not load prediction history">{error}</Alert>}
      {(loading || (!predictionId && batchRuns.length > 0)) && <Skeleton className="h-64 rounded-card" />}

      {!loading && !predictionId && batchRuns.length === 0 && (
        <EmptyState
          title="No batch predictions yet"
          description="Run a batch prediction from the Prediction page — at-risk detection reads directly from that result."
          action={<Link to="/predict" className="btn-primary">Go to prediction</Link>}
        />
      )}

      {predictionId && (
        <StudentExplorer predictionId={predictionId} lockRisk={['high', 'medium']} />
      )}
    </div>
  )
}
