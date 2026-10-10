// Feature 36 (What-If Simulator). Starts from one real, already-scored
// student's own values, lets the person edit them, and reruns the SAME
// trained model on the edited values via the existing single-prediction
// endpoint (POST /api/predict/single) — nothing here retrains the model,
// changes the dataset, or touches the stored prediction. It's a pure
// what-if: edit -> predict -> compare to the original.

import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import { FeatureInput } from '../../components/FeatureInput'
import { Alert, Card, EmptyState, PageHeader, SectionHeading, Skeleton, formatCell, formatNumber } from '../../components/ui'
import { useAsync } from '../../hooks/useAsync'
import * as api from '../../services/api'
import type { Cell, SingleResult } from '../../types/api'

function cellToInput(value: Cell): string | number {
  if (value === null || value === undefined) return ''
  if (typeof value === 'number') return value
  if (typeof value === 'boolean') return value ? 'True' : 'False'
  return value
}

export function WhatIfPage() {
  const { predictionId, rowIndex } = useParams()
  const idx = Number(rowIndex)

  const { data: capability, loading: capLoading, error: capError } = useAsync(
    () => api.getStudentCapability(predictionId!),
    [predictionId],
    { enabled: Boolean(predictionId) },
  )
  const { data: profile, loading: profileLoading, error: profileError } = useAsync(
    () => api.getStudentProfile(predictionId!, idx),
    [predictionId, idx],
    { enabled: Boolean(predictionId) && Number.isFinite(idx) },
  )
  const { data: schema, loading: schemaLoading } = useAsync(
    () => api.getFeatureSchema(capability!.model_id!),
    [capability?.model_id],
    { enabled: Boolean(capability?.model_id) },
  )

  const [inputs, setInputs] = useState<Record<string, string | number>>({})
  const [result, setResult] = useState<SingleResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!profile || !schema) return
    const allFields = [...profile.sections.academic, ...profile.sections.behavioral, ...profile.sections.demographic, ...profile.sections.other]
    const byName = new Map(allFields.map((f) => [f.name, f.value]))
    const starting: Record<string, string | number> = {}
    for (const field of schema.fields) {
      starting[field.name] = byName.has(field.name) ? cellToInput(byName.get(field.name)!) : field.default
    }
    setInputs(starting)
    setResult(null)
  }, [profile, schema])

  async function run() {
    if (!capability?.model_id) return
    setBusy(true)
    setError(null)
    try {
      setResult(await api.predictSingle(capability.model_id, inputs))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'The what-if prediction could not be completed.')
    } finally {
      setBusy(false)
    }
  }

  function resetToActual() {
    if (!profile || !schema) return
    const allFields = [...profile.sections.academic, ...profile.sections.behavioral, ...profile.sections.demographic, ...profile.sections.other]
    const byName = new Map(allFields.map((f) => [f.name, f.value]))
    const starting: Record<string, string | number> = {}
    for (const field of schema.fields) {
      starting[field.name] = byName.has(field.name) ? cellToInput(byName.get(field.name)!) : field.default
    }
    setInputs(starting)
    setResult(null)
  }

  const loading = capLoading || profileLoading || schemaLoading
  const loadError = capError || profileError

  if (!predictionId) {
    return (
      <div className="space-y-6">
        <PageHeader eyebrow="Insights" title="What-if simulator" />
        <EmptyState title="No student selected" description="Open a student profile and choose What-if simulator there." />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Insights"
        title={profile ? `What-if: ${profile.display_label}` : 'What-if simulator'}
        description="Edit student values to simulate outcome changes."
        action={
          <Link className="btn-secondary h-9 px-3 text-[12px]" to={`/insights/profile/${predictionId}/${idx}`}>
            Back to profile
          </Link>
        }
      />

      {loadError && <Alert tone="danger" title="Could not load this student">{loadError}</Alert>}
      {loading && <Skeleton className="h-96 rounded-card" />}

      {profile && (
        <Card>
          <p className="text-[12px] font-medium text-muted">Original prediction</p>
          <p className="font-display text-[20px] font-bold text-ink mt-1">{formatCell(profile.prediction.value)}</p>
          {profile.confidence.value !== null && (
            <p className="text-[12px] text-muted mt-0.5">{(profile.confidence.value * 100).toFixed(0)}% confidence</p>
          )}
        </Card>
      )}

      {schema && (
        <Card>
          <SectionHeading
            title="Edit values"
            description="Adjust parameters to simulate outcome."
          />
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {schema.fields.map((field) => (
              <FeatureInput
                key={field.name}
                field={field}
                value={inputs[field.name] ?? field.default}
                onChange={(value) => setInputs((current) => ({ ...current, [field.name]: value }))}
              />
            ))}
          </div>

          <div className="flex flex-wrap items-center gap-3 mt-6">
            <button type="button" className="btn-primary px-6" onClick={run} disabled={busy}>
              {busy ? 'Simulating…' : 'Simulate'}
            </button>
            <button type="button" className="btn-secondary" onClick={resetToActual} disabled={busy}>
              Reset to actual
            </button>
          </div>

          {error && (
            <div className="mt-5">
              <Alert tone="danger" title="Simulation stopped">{error}</Alert>
            </div>
          )}

          {result && (
            <div className="mt-6 rounded-card border border-brand/25 bg-brand-soft/60 p-6 animate-rise">
              <p className="text-[13px] font-semibold text-brand-deep">Simulated {result.target_column}</p>
              <p className="font-display text-[36px] leading-none font-extrabold text-ink mt-2 tabular-nums">
                {result.problem_type === 'classification'
                  ? String(result.predicted_class ?? result.prediction)
                  : formatNumber(Number(result.prediction))}
              </p>
              {result.confidence !== null && result.confidence !== undefined && (
                <p className="text-[12px] text-muted mt-2">{(result.confidence * 100).toFixed(1)}% confidence</p>
              )}
              {profile && (
                <p className="text-[12px] text-muted mt-3">
                  Original: {formatCell(profile.prediction.value)}
                  {profile.confidence.value !== null ? ` (${(profile.confidence.value * 100).toFixed(0)}% confidence)` : ''}
                </p>
              )}
              {result.notice && (
                <div className="mt-3 p-3 rounded-lg bg-panel/70 border border-brand/20 text-[12px] text-ink flex items-center gap-2">
                  <span className="text-brand font-bold">ℹ</span>
                  <span>{result.notice}</span>
                </div>
              )}
            </div>
          )}
        </Card>
      )}
    </div>
  )
}
