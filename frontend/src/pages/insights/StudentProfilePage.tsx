// Feature 19 (Individual Student Profile). Organizes one scored student's
// own columns into Academic / Behavioral / Demographic sections (a
// best-effort grouping by column name — see categorize_column in
// app/ml/risk.py), plus the model's Prediction / Confidence / Risk for that
// row.

import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'

import { Alert, Badge, Card, PageHeader, Skeleton, formatCell } from '../../components/ui'
import { useAsync } from '../../hooks/useAsync'
import * as api from '../../services/api'
import type { Contribution, ExplainInstanceResult, ProfileField, RiskLevel } from '../../types/api'

const RISK_TONE: Record<RiskLevel, 'danger' | 'warn' | 'brand'> = {
  high: 'danger',
  medium: 'warn',
  low: 'brand',
}

function FieldGrid({ fields }: { fields: ProfileField[] }) {
  if (fields.length === 0) return <p className="text-[13px] text-muted">No columns in this category.</p>
  return (
    <dl className="grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-3">
      {fields.map((f) => (
        <div key={f.name} className="min-w-0">
          <dt className="text-[11px] font-medium text-muted truncate" title={f.name}>
            {f.name}
          </dt>
          <dd className="text-[14px] font-semibold text-ink truncate" title={formatCell(f.value)}>
            {formatCell(f.value)}
          </dd>
        </div>
      ))}
    </dl>
  )
}

function ContributionBar({ contribution, maxAbs }: { contribution: Contribution; maxAbs: number }) {
  const width = maxAbs > 0 ? Math.max(4, (Math.abs(contribution.contribution) / maxAbs) * 100) : 0
  const positive = contribution.direction === 'positive'
  return (
    <div className="flex items-center gap-3 py-1.5">
      <span className="w-40 shrink-0 text-[12px] text-muted truncate" title={contribution.feature}>
        {contribution.feature}
      </span>
      <div className="flex-1 h-2.5 rounded-full bg-canvas relative overflow-hidden">
        <div
          className={`h-full rounded-full ${positive ? 'bg-brand' : 'bg-danger'}`}
          style={{ width: `${width}%` }}
        />
      </div>
      <span className={`w-16 shrink-0 text-right text-[12px] font-semibold tabular-nums ${positive ? 'text-brand-deep' : 'text-danger'}`}>
        {positive ? '+' : ''}
        {contribution.contribution.toFixed(3)}
      </span>
    </div>
  )
}

function ExplainSection({ predictionId, rowIndex }: { predictionId: string; rowIndex: number }) {
  const [result, setResult] = useState<ExplainInstanceResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function run() {
    setLoading(true)
    setError(null)
    try {
      setResult(await api.explainForPrediction(predictionId, rowIndex))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not explain this prediction.')
    } finally {
      setLoading(false)
    }
  }

  const maxAbs = result?.explanation
    ? Math.max(...result.explanation.top_contributions.map((c) => Math.abs(c.contribution)))
    : 0

  return (
    <Card>
      <div className="flex items-center justify-between gap-3 mb-3">
        <div>
          <h2 className="font-display text-[15px] font-bold text-ink">Explain this prediction</h2>
          <p className="text-[12px] text-muted mt-0.5">
            {result?.capability.description ?? 'Computed only when you ask — never automatically.'}
          </p>
        </div>
        <button type="button" className="btn-secondary h-9 px-3 text-[12px] shrink-0" onClick={run} disabled={loading}>
          {loading ? 'Explaining…' : result ? 'Refresh' : 'Explain'}
        </button>
      </div>

      {error && <Alert tone="danger">{error}</Alert>}

      {result && !result.capability.supported && (
        <p className="text-[13px] text-muted">{result.capability.description}</p>
      )}

      {result?.explanation && (
        <div>
          {result.explanation.top_contributions.map((c) => (
            <ContributionBar key={c.feature} contribution={c} maxAbs={maxAbs} />
          ))}
          <p className="text-[11px] text-muted mt-2">
            Blue bars push the prediction up; red bars pull it down. Bars are this student's own feature values
            compared against what the model typically sees.
          </p>
        </div>
      )}
    </Card>
  )
}

export function StudentProfilePage() {
  const { predictionId, rowIndex } = useParams()
  const navigate = useNavigate()
  const idx = Number(rowIndex)

  const { data: profile, loading, error } = useAsync(
    () => api.getStudentProfile(predictionId!, idx),
    [predictionId, idx],
    { enabled: Boolean(predictionId) && Number.isFinite(idx) },
  )

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Insights"
        title={profile ? profile.display_label : 'Student profile'}
        description="Every column for this student, organized by category, alongside the model's prediction and risk."
        action={
          <div className="flex items-center gap-2">
            {predictionId && (
              <Link className="btn-secondary h-9 px-3 text-[12px]" to={`/insights/whatif/${predictionId}/${idx}`}>
                What-if simulator
              </Link>
            )}
            <button type="button" className="btn-secondary h-9 px-3 text-[12px]" onClick={() => navigate(-1)}>
              Back
            </button>
          </div>
        }
      />

      {error && (
        <Alert tone="danger" title="Could not load this student">
          {error}
          <div className="mt-2">
            <Link className="underline" to={`/insights/explorer/${predictionId}`}>
              Back to the explorer
            </Link>
          </div>
        </Alert>
      )}
      {loading && <Skeleton className="h-72 rounded-card" />}

      {profile && (
        <>
          <div className="grid sm:grid-cols-3 gap-4">
            <Card>
              <p className="text-[12px] font-medium text-muted">Prediction</p>
              <p className="font-display text-[20px] font-bold text-ink mt-1">{formatCell(profile.prediction.value)}</p>
              <p className="text-[11px] text-muted mt-0.5">
                Target: {profile.prediction.target_column ?? '—'} ({profile.prediction.problem_type})
              </p>
            </Card>
            <Card>
              <p className="text-[12px] font-medium text-muted">Confidence</p>
              <p className="font-display text-[20px] font-bold text-ink mt-1">
                {profile.confidence.value !== null ? `${(profile.confidence.value * 100).toFixed(0)}%` : 'Not available'}
              </p>
              {profile.confidence.tier && (
                <div className="mt-1">
                  <Badge tone={RISK_TONE[profile.confidence.tier]}>{profile.confidence.tier} confidence</Badge>
                </div>
              )}
            </Card>
            <Card>
              <p className="text-[12px] font-medium text-muted">Risk</p>
              {profile.risk.value ? (
                <div className="mt-1.5">
                  <Badge tone={RISK_TONE[profile.risk.value]}>{profile.risk.value[0].toUpperCase() + profile.risk.value.slice(1)} risk</Badge>
                </div>
              ) : (
                <p className="text-[13px] text-muted mt-1">{profile.risk.capability.reason ?? 'Not available'}</p>
              )}
            </Card>
          </div>

          <Card>
            <h2 className="font-display text-[15px] font-bold text-ink mb-3">Academic</h2>
            <FieldGrid fields={profile.sections.academic} />
          </Card>
          <Card>
            <h2 className="font-display text-[15px] font-bold text-ink mb-3">Behavioral</h2>
            <FieldGrid fields={profile.sections.behavioral} />
          </Card>
          <Card>
            <h2 className="font-display text-[15px] font-bold text-ink mb-3">Demographic</h2>
            <FieldGrid fields={profile.sections.demographic} />
          </Card>
          {profile.sections.other.length > 0 && (
            <Card>
              <h2 className="font-display text-[15px] font-bold text-ink mb-3">Other</h2>
              <FieldGrid fields={profile.sections.other} />
            </Card>
          )}

          <ExplainSection predictionId={predictionId!} rowIndex={idx} />
        </>
      )}
    </div>
  )
}
