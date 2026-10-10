// Feature 25 (Explainable Predictions / SHAP) — model-level view. Per-student
// explanations live on the Student Profile page, next to the row they
// explain; this page covers the model as a whole.
//
// Global importance is never computed on page load — only when the person
// asks for it — and it runs against a capped background sample, never the
// full dataset (see app/services/explain_service.py).

import { useState } from 'react'
import { Link } from 'react-router-dom'

import { FeatureImportanceChart } from '../../charts'
import { Alert, Card, EmptyState, PageHeader, SectionHeading, Skeleton } from '../../components/ui'
import { useAsync } from '../../hooks/useAsync'
import { useWorkspace } from '../../hooks/useWorkspace'
import * as api from '../../services/api'
import type { GlobalImportanceResult } from '../../types/api'

export function ExplainabilityPage() {
  const { model } = useWorkspace()
  const [importance, setImportance] = useState<GlobalImportanceResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const { data: capability, loading: capLoading } = useAsync(
    () => api.getExplainCapability(model!.model_id),
    [model?.model_id],
    { enabled: Boolean(model) },
  )

  async function loadImportance() {
    if (!model) return
    setLoading(true)
    setError(null)
    try {
      setImportance(await api.getGlobalImportance(model.model_id))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not compute feature importance.')
    } finally {
      setLoading(false)
    }
  }

  if (!model) {
    return (
      <div className="space-y-6">
        <PageHeader eyebrow="Insights" title="Explainability" />
        <EmptyState
          title="No trained model"
          description="Train a model first — explainability reads its own fitted pipeline directly."
          action={<Link to="/train" className="btn-primary">Go to training</Link>}
        />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Insights"
        title="Explainability"
        description={`Which features drive ${model.model_choice}'s predictions of ${model.target_column}, and why.`}
      />

      {capLoading && <Skeleton className="h-24 rounded-card" />}

      {capability && !capability.supported && (
        <div className="space-y-4">
          <Alert tone="info" title="Explanations are not available for this model">
            {capability.description}
          </Alert>
          {model.charts?.feature_importance && (
            <Card>
              <SectionHeading
                title="Feature importance from training"
                description="Gini importance computed when this model was trained — shown here since a live explanation isn't available for this algorithm."
              />
              <FeatureImportanceChart chart={model.charts.feature_importance} />
            </Card>
          )}
        </div>
      )}

      {capability?.supported && (
        <Card>
          <SectionHeading
            title="Global feature importance"
            description="Ranking of feature influence across students."
            action={
              <button type="button" className="btn-primary h-9 px-4 text-[13px]" onClick={loadImportance} disabled={loading}>
                {loading ? 'Computing…' : importance ? 'Recompute' : 'Compute feature importance'}
              </button>
            }
          />

          {error && <Alert tone="danger">{error}</Alert>}
          {loading && <Skeleton className="h-72 rounded-card" />}

          {importance?.importance && (() => {
            const imp = importance.importance
            return (
              <div className="space-y-4">
                <div className="p-3.5 rounded-xl border border-line bg-canvas/70 space-y-2">
                  <p className="text-[11px] font-bold uppercase tracking-wider text-muted">Top Factors</p>
                  <div className="grid gap-2 sm:grid-cols-3">
                    {imp.features.slice(0, 3).map((feat, idx) => (
                      <div key={feat} className="p-2.5 rounded-lg bg-panel border border-line flex items-center justify-between">
                        <span className="font-semibold text-ink text-[13px]">{feat}</span>
                        <span className="font-mono text-brand font-bold text-[13px]">
                          +{imp.importance[idx]?.toFixed(2)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                <FeatureImportanceChart chart={imp} />
              </div>
            )
          })()}
        </Card>
      )}

      <Alert tone="info" title="Explain one student">
        Open a student's profile from the Prediction Explorer or At-Risk page and use "Explain this prediction" there
        for a per-student breakdown.
      </Alert>
    </div>
  )
}
