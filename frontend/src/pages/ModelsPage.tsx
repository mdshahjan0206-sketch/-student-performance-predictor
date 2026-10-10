// Features 5 (Model Comparison) and 6 (Training History) share one data
// source — GET /api/model/history/{dataset_id}, which already existed for
// the Train page's history — so this page just presents it two ways instead
// of adding a parallel endpoint. No model is ever picked as "best": figures
// are shown side by side and the person compares them.

import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'

import { Plot, useChartPalette } from '../charts/Plot'
import { Alert, Badge, Card, EmptyState, PageHeader, SegmentedControl, Skeleton } from '../components/ui'
import { useAsync } from '../hooks/useAsync'
import { useWorkspace } from '../hooks/useWorkspace'
import * as api from '../services/api'
import type { ModelInfo } from '../types/api'

const CLASSIFICATION_METRICS = ['accuracy', 'precision', 'recall', 'f1']
const REGRESSION_METRICS = ['mae', 'mse', 'rmse', 'r2']

function MetricsBarChart({ models }: { models: ModelInfo[] }) {
  const c = useChartPalette()
  const problemType = models[0]?.problem_type
  const metricKeys = problemType === 'regression' ? REGRESSION_METRICS : CLASSIFICATION_METRICS
  const present = metricKeys.filter((key) => models.some((m) => key in m.metrics))
  if (!present.length) return null

  const data = present.map((key, index) => ({
    type: 'bar' as const,
    name: key.toUpperCase(),
    x: models.map((m) => m.model_choice),
    y: models.map((m) => m.metrics[key] ?? null),
    marker: { color: [c.brand, c.blue, c.amber, c.violet][index % 4] },
  }))

  return (
    <Plot
      title="Metric comparison across trained models"
      height={340}
      data={data}
      layout={{ barmode: 'group', showlegend: true, yaxis: { title: { text: 'Score' } } }}
    />
  )
}

function ComparisonTable({ models }: { models: ModelInfo[] }) {
  const problemType = models[0]?.problem_type
  const metricKeys = problemType === 'regression' ? REGRESSION_METRICS : CLASSIFICATION_METRICS
  const present = metricKeys.filter((key) => models.some((m) => key in m.metrics))

  return (
    <div className="table-scroll">
      <table className="w-full text-[13px]">
        <thead>
          <tr className="text-left text-muted border-b border-line">
            <th className="py-2 pr-4 font-medium">Model</th>
            {present.map((key) => (
              <th key={key} className="py-2 px-3 font-medium uppercase">{key}</th>
            ))}
            <th className="py-2 px-3 font-medium">Training time</th>
            <th className="py-2 px-3 font-medium">Features</th>
            <th className="py-2 pl-3 font-medium">Trained</th>
          </tr>
        </thead>
        <tbody>
          {models.map((model) => (
            <tr key={model.model_id} className="border-b border-line last:border-0">
              <td className="py-2.5 pr-4 font-semibold text-ink">{model.model_choice}</td>
              {present.map((key) => (
                <td key={key} className="py-2.5 px-3 tabular-nums text-ink">
                  {model.metrics[key] !== undefined ? model.metrics[key].toFixed(4) : '—'}
                </td>
              ))}
              <td className="py-2.5 px-3 tabular-nums text-muted">{model.training_time.toFixed(3)}s</td>
              <td className="py-2.5 px-3 tabular-nums text-muted">{model.feature_count}</td>
              <td className="py-2.5 pl-3 text-muted">{new Date(model.trained_at * 1000).toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function HistoryList({ models }: { models: ModelInfo[] }) {
  const [expanded, setExpanded] = useState<string | null>(models[0]?.model_id ?? null)

  return (
    <div className="space-y-2">
      {models.map((model) => {
        const open = expanded === model.model_id
        return (
          <Card key={model.model_id} padded={false}>
            <button
              className="w-full flex items-center justify-between gap-3 p-4 text-left"
              onClick={() => setExpanded(open ? null : model.model_id)}
              aria-expanded={open}
            >
              <div>
                <p className="text-[14px] font-semibold text-ink">
                  {model.model_choice} <span className="text-muted font-normal">on {model.target_column}</span>
                </p>
                <p className="text-[12px] text-muted mt-0.5">
                  {new Date(model.trained_at * 1000).toLocaleString()} · {model.dataset_filename}
                </p>
              </div>
              <Badge tone="neutral">{model.problem_type}</Badge>
            </button>
            {open && (
              <div className="px-4 pb-4 border-t border-line pt-3 grid gap-3 sm:grid-cols-2">
                <div>
                  <p className="text-[12px] text-muted mb-1">Metrics</p>
                  <div className="grid grid-cols-2 gap-2">
                    {Object.entries(model.metrics).map(([key, value]) => (
                      <div key={key} className="rounded-lg bg-canvas border border-line px-3 py-1.5">
                        <p className="text-[10px] text-muted uppercase">{key}</p>
                        <p className="text-[13px] font-bold text-ink tabular-nums">{value.toFixed(4)}</p>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="text-[13px] text-muted space-y-1">
                  <p>Training rows: <span className="text-ink font-medium">{model.n_train.toLocaleString()}</span></p>
                  <p>Test rows: <span className="text-ink font-medium">{model.n_test.toLocaleString()}</span></p>
                  <p>Training time: <span className="text-ink font-medium">{model.training_time.toFixed(3)}s</span></p>
                  <p>Feature count: <span className="text-ink font-medium">{model.feature_count}</span></p>
                  {model.warnings.length > 0 && (
                    <p className="text-amber">{model.warnings.length} warning(s) at training time</p>
                  )}
                </div>
              </div>
            )}
          </Card>
        )
      })}
    </div>
  )
}

export function ModelsPage() {
  const { dataset } = useWorkspace()
  const [view, setView] = useState<'comparison' | 'history'>('comparison')

  const { data: models, loading, error } = useAsync(
    () => api.getTrainingHistory(dataset!.dataset_id),
    [dataset?.dataset_id],
    { enabled: Boolean(dataset) },
  )

  const sorted = useMemo(() => (models ?? []).slice().sort((a, b) => b.trained_at - a.trained_at), [models])

  if (!dataset) {
    return (
      <div className="space-y-6">
        <PageHeader eyebrow="Models" title="Model comparison & training history" />
        <EmptyState
          title="No dataset loaded"
          description="Upload a dataset and train a model to see comparisons here."
          action={<Link to="/upload" className="btn-primary">Upload a dataset</Link>}
        />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Models"
        title="Model comparison & training history"
        description="Every model trained on the active dataset, with its actual measured results. Nothing here is ranked as 'best' — figures are shown so you can compare them yourself."
        action={
          <SegmentedControl
            label="View"
            name="models-view"
            value={view}
            onChange={(v) => setView(v as 'comparison' | 'history')}
            options={[
              { value: 'comparison', label: 'Comparison' },
              { value: 'history', label: 'History' },
            ]}
          />
        }
      />

      {error && <Alert tone="danger" title="Could not load model history">{error}</Alert>}
      {loading && <Skeleton className="h-64 rounded-card" />}

      {!loading && sorted.length === 0 && (
        <EmptyState
          title="No models trained yet"
          description="Train at least one model on this dataset to compare results here."
          action={<Link to="/train" className="btn-primary">Train a model</Link>}
        />
      )}

      {sorted.length > 0 && view === 'comparison' && (
        <>
          <Card>
            <MetricsBarChart models={sorted} />
          </Card>
          <Card>
            <ComparisonTable models={sorted} />
          </Card>
        </>
      )}

      {sorted.length > 0 && view === 'history' && <HistoryList models={sorted} />}
    </div>
  )
}
