// Dashboard — visual-first, minimal text, green glass.
//
// Sections:
// 1. Page header (title + 1 short line)
// 2. KPI cards with accent colors and icons
// 3. Charts: Score Distribution, Performance Overview, Top Factors
// 4. Recent Activity, Model Status, Quick Actions

import { useMemo } from 'react'
import { Link } from 'react-router-dom'
import {
  Users,
  Columns3,
  FileWarning,
  Target,
  TriangleAlert,
  TrendingUp,
  Upload,
  BarChart3,
  Brain,
  Sparkles,
  CheckCircle2,
  Activity,
  LayoutDashboard,
} from 'lucide-react'

import {
  DistributionHistogram,
  MissingValuesChart,
  PredictionDistribution,
  ScatterPlot,
} from '../charts'
import {
  Alert,
  Card,
  EmptyState,
  Skeleton,
} from '../components/ui'
import { useAsync } from '../hooks/useAsync'
import { useWorkspace } from '../hooks/useWorkspace'
import * as api from '../services/api'
import type {
  ChartResponse,
  DatasetIntelligence,
  EnhancedDashboardSummary,
  HistogramChart,
  MissingChart,
  ScatterChart,
} from '../types/api'

const METRIC_LABELS: Record<string, string> = {
  accuracy: 'Accuracy',
  precision: 'Precision',
  recall: 'Recall',
  f1: 'F1',
  r2: 'R²',
  rmse: 'RMSE',
  mae: 'MAE',
  mse: 'MSE',
}

function KpiCard({
  icon: Icon,
  value,
  label,
  accent = 'kpi-brand',
  subtitle,
  highlight = false,
}: {
  icon: React.ElementType
  value: string | number
  label: string
  accent?: string
  subtitle?: string
  highlight?: boolean
}) {
  return (
    <div className={`kpi-card ${accent} ${highlight ? 'border-brand/45 shadow-glow-brand' : ''}`}>
      <div className="kpi-icon">
        <Icon size={18} strokeWidth={1.8} />
      </div>
      <p className={`font-display text-[24px] font-semibold tabular-nums leading-none ${highlight ? 'text-brand-mid' : 'text-ink'}`}>{value}</p>
      <p className="text-[11.5px] font-medium text-muted mt-1">{label}</p>
      {subtitle && <p className="text-[10px] text-muted/60 mt-0.5">{subtitle}</p>}
    </div>
  )
}

export function DashboardPage() {
  const { dataset, model } = useWorkspace()
  const datasetId = dataset?.dataset_id ?? ''

  const {
    data: dashData,
    loading: dashLoading,
    error: dashError,
    reload: reloadDash,
  } = useAsync<EnhancedDashboardSummary>(
    () => api.getDashboard(datasetId),
    [datasetId],
    { enabled: Boolean(datasetId) },
  )

  const {
    data: intelData,
    loading: intelLoading,
    reload: reloadIntel,
  } = useAsync<DatasetIntelligence>(
    () => api.getDatasetIntelligence(datasetId),
    [datasetId],
    { enabled: Boolean(datasetId) },
  )

  const numericCols = useMemo(
    () => intelData?.composition?.numeric?.columns ?? dashData?.numeric_columns ?? [],
    [intelData, dashData],
  )

  const targetCol = useMemo(() => {
    return (
      dashData?.model?.target_column ||
      dashData?.detected_target ||
      intelData?.target_column ||
      (numericCols.length > 0 ? numericCols[numericCols.length - 1] : '')
    )
  }, [dashData, intelData, numericCols])

  const isNumericTarget = useMemo(() => {
    return Boolean(targetCol && numericCols.includes(targetCol))
  }, [targetCol, numericCols])

  const attendanceCol = useMemo(() => {
    return numericCols.find((c) => /attend|absence/i.test(c) && c !== targetCol) ?? ''
  }, [numericCols, targetCol])

  const studyCol = useMemo(() => {
    return numericCols.find((c) => /study|hour/i.test(c) && c !== targetCol && c !== attendanceCol) ?? ''
  }, [numericCols, targetCol, attendanceCol])

  const scoreHist = useAsync<ChartResponse<HistogramChart>>(
    () => api.getHistogram(datasetId, targetCol),
    [datasetId, targetCol],
    { enabled: Boolean(datasetId && targetCol && isNumericTarget) },
  )

  const attendanceScatter = useAsync<ChartResponse<ScatterChart>>(
    () => api.getScatter(datasetId, attendanceCol, targetCol),
    [datasetId, attendanceCol, targetCol],
    { enabled: Boolean(datasetId && attendanceCol && targetCol && isNumericTarget) },
  )

  const studyScatter = useAsync<ChartResponse<ScatterChart>>(
    () => api.getScatter(datasetId, studyCol, targetCol),
    [datasetId, studyCol, targetCol],
    { enabled: Boolean(datasetId && studyCol && targetCol && isNumericTarget) },
  )

  const missingChart = useAsync<ChartResponse<MissingChart>>(
    () => api.getMissing(datasetId),
    [datasetId],
    { enabled: Boolean(datasetId && dashData && dashData.missing_values > 0) },
  )

  if (!dataset) {
    return (
      <div className="space-y-5">
        <div>
          <h1 className="font-display text-[22px] font-extrabold text-ink tracking-tight flex items-center gap-2">
            <LayoutDashboard size={22} className="text-brand" />
            Dashboard
          </h1>
          <p className="text-[12.5px] text-muted mt-0.5">Dataset, model and prediction overview</p>
        </div>
        <EmptyState
          title="No dataset loaded"
          description="Upload a CSV dataset to get started."
          action={
            <Link to="/upload" className="btn-primary">
              <Upload size={15} /> Upload Dataset
            </Link>
          }
        />
      </div>
    )
  }

  // KPI values from real data
  const studentCount = dataset.num_rows >= 1000
    ? `${(dataset.num_rows / 1000).toFixed(dataset.num_rows % 1000 === 0 ? 0 : 1)}K`
    : dataset.num_rows.toLocaleString()

  const missingCount = dashData?.missing_values ?? dataset.total_missing ?? 0
  const isRefinementRequired = missingCount > 0 && !dataset.refined

  let modelAccuracy = '—'
  if (dashData?.model) {
    if (dashData.model.problem_type === 'classification' && dashData.model.metrics?.accuracy !== undefined) {
      modelAccuracy = `${(dashData.model.metrics.accuracy * 100).toFixed(1)}%`
    } else if (dashData.model.metrics?.r2 !== undefined) {
      modelAccuracy = `${dashData.model.metrics.r2.toFixed(2)}`
    }
  }

  const atRiskDisplay = dashData?.at_risk_count != null ? dashData.at_risk_count.toLocaleString() : '—'

  let avgScore = '—'
  if (dashData?.target_breakdown?.kind === 'regression' && dashData.target_breakdown.average !== undefined) {
    avgScore = dashData.target_breakdown.average.toFixed(1)
  } else if (dashData?.detected_target_breakdown?.kind === 'regression' && dashData.detected_target_breakdown.average !== undefined) {
    avgScore = dashData.detected_target_breakdown.average.toFixed(1)
  }

  // Feature importance for Top Factors card
  const featureImportance = model?.charts?.feature_importance

  // Recent activity items
  const activities: { icon: React.ElementType; text: string; time: string }[] = []
  if (dataset) activities.push({ icon: Upload, text: 'Dataset uploaded', time: dataset.filename })
  if (dataset?.refined) activities.push({ icon: CheckCircle2, text: 'Data refined', time: `${missingCount.toLocaleString()} cells cleaned` })
  if (model) activities.push({ icon: Brain, text: 'Model trained', time: `${model.model_choice} (${modelAccuracy})` })
  if (dashData?.prediction_runs && dashData.prediction_runs > 0) {
    activities.push({ icon: Sparkles, text: 'Predictions ready', time: `${dashData.records_scored?.toLocaleString() ?? '—'} students` })
  }

  return (
    <div className="space-y-5">
      {/* Page header */}
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="font-display text-[22px] font-extrabold text-ink tracking-tight flex items-center gap-2">
            <LayoutDashboard size={22} className="text-brand" />
            Dashboard
          </h1>
          <p className="text-[12.5px] text-muted mt-0.5">Dataset, model and prediction overview</p>
        </div>
        <div className="flex items-center gap-2">
          <Link to="/analysis" className="btn-secondary text-[12px] h-8 px-3 gap-1.5">
            <BarChart3 size={14} /> Analyse
          </Link>
          <Link
            to={isRefinementRequired ? '/analysis' : '/train'}
            className={`btn-secondary text-[12px] h-8 px-3 gap-1.5 ${isRefinementRequired ? 'border-amber/40 text-amber' : ''}`}
          >
            <Brain size={14} /> Train
          </Link>
          {model ? (
            <Link to="/predict" className="btn-primary text-[12px] h-8 px-3.5 gap-1.5">
              <Sparkles size={14} /> Predict
            </Link>
          ) : (
            <span className="btn text-[12px] h-8 px-3 bg-canvas text-muted/50 cursor-not-allowed border border-line" title="Train a model first">
              Predict · Locked
            </span>
          )}
        </div>
      </div>

      {dashError && (
        <Alert tone="danger" title="Could not load dashboard">
          {dashError}{' '}
          <button type="button" className="underline ml-1 font-semibold" onClick={() => { reloadDash(); reloadIntel() }}>
            Retry
          </button>
        </Alert>
      )}

      {/* KPI Cards */}
      {(dashLoading || intelLoading) && !dashData ? (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-[100px] rounded-card" />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          <KpiCard icon={Users} value={studentCount} label="Students" accent="kpi-brand" />
          <KpiCard icon={Columns3} value={dataset.num_cols.toString()} label="Features" accent="kpi-violet" />
          <KpiCard
            icon={FileWarning}
            value={missingCount.toLocaleString()}
            label="Missing"
            accent="kpi-amber"
            subtitle={isRefinementRequired ? 'Missing values detected' : dataset.refined ? 'Handled' : undefined}
          />
          <KpiCard icon={Target} value={modelAccuracy} label={dashData?.model?.problem_type === 'classification' ? 'Accuracy' : 'R²'} accent="kpi-brand" highlight={modelAccuracy !== '—'} />
          <KpiCard icon={TriangleAlert} value={atRiskDisplay} label="At Risk" accent="kpi-rose" />
          <KpiCard icon={TrendingUp} value={avgScore} label="Avg Score" accent="kpi-cyan" />
        </div>
      )}

      {/* Charts Row */}
      <div className="grid gap-4 lg:grid-cols-3">
        {/* Score Distribution */}
        {isNumericTarget && (
          <Card className="p-4 lg:col-span-2">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-[13px] font-bold text-ink">Score Distribution</h3>
              <span className="font-mono text-[10.5px] text-muted">{targetCol}</span>
            </div>
            {scoreHist.loading ? (
              <Skeleton className="h-56 rounded-lg" />
            ) : scoreHist.data?.chart ? (
              <DistributionHistogram chart={scoreHist.data.chart} height={220} />
            ) : null}
          </Card>
        )}

        {/* Target classification distribution */}
        {dashData?.target_breakdown?.kind === 'classification' && dashData.target_breakdown.labels && (
          <Card className="p-4 lg:col-span-2">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-[13px] font-bold text-ink">Target Distribution</h3>
              <span className="font-mono text-[10.5px] text-muted">{targetCol}</span>
            </div>
            <PredictionDistribution
              distribution={{
                kind: 'classification',
                labels: dashData.target_breakdown.labels,
                counts: dashData.target_breakdown.counts,
              }}
              height={220}
            />
          </Card>
        )}

        {/* Top Factors */}
        {featureImportance && (
          <Card className="p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-[13px] font-bold text-ink">Top Factors</h3>
              <span className="text-[10.5px] text-muted">Importance</span>
            </div>
            <div className="space-y-2">
              {featureImportance.features.slice(0, 6).map((feat, idx) => {
                const imp = featureImportance.importance[idx]
                const maxImp = featureImportance.importance[0] || 1
                const widthPct = Math.max(8, (imp / maxImp) * 100)
                return (
                  <div key={feat} className="flex items-center gap-2">
                    <span className="text-[11.5px] text-ink font-medium w-28 truncate shrink-0" title={feat}>{feat}</span>
                    <div className="flex-1 h-2 bg-brand-soft rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-brand-mid to-brand transition-all duration-500"
                        style={{ width: `${widthPct}%` }}
                      />
                    </div>
                    <span className="text-[11px] font-mono text-brand tabular-nums w-8 text-right">{imp.toFixed(2)}</span>
                  </div>
                )
              })}
            </div>
          </Card>
        )}
      </div>

      {/* Scatter charts row */}
      {(attendanceCol || studyCol) && isNumericTarget && (
        <div className="grid gap-4 md:grid-cols-2">
          {attendanceCol && (
            <Card className="p-4">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-[13px] font-bold text-ink">Attendance vs Score</h3>
                <span className="font-mono text-[10.5px] text-muted">{attendanceCol}</span>
              </div>
              {attendanceScatter.loading ? (
                <Skeleton className="h-48 rounded-lg" />
              ) : attendanceScatter.data?.chart ? (
                <ScatterPlot chart={attendanceScatter.data.chart} height={200} />
              ) : null}
            </Card>
          )}
          {studyCol && (
            <Card className="p-4">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-[13px] font-bold text-ink">Study Hours vs Score</h3>
                <span className="font-mono text-[10.5px] text-muted">{studyCol}</span>
              </div>
              {studyScatter.loading ? (
                <Skeleton className="h-48 rounded-lg" />
              ) : studyScatter.data?.chart ? (
                <ScatterPlot chart={studyScatter.data.chart} height={200} />
              ) : null}
            </Card>
          )}
        </div>
      )}

      {/* Missing values chart */}
      {dashData && dashData.missing_values > 0 && (
        <Card className="p-4">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-[13px] font-bold text-ink flex items-center gap-1.5">
              <FileWarning size={14} className="text-amber" />
              Missing Value Overview
            </h3>
            <span className={`inline-flex items-center gap-1 text-[10.5px] font-medium ${dataset.refined ? 'text-ok' : 'text-amber'}`}>
              {dataset.refined && <CheckCircle2 size={12} aria-hidden />}
              {dataset.refined ? 'Data refined' : `${dashData.missing_values.toLocaleString()} cells`}
            </span>
          </div>
          {missingChart.loading ? (
            <Skeleton className="h-48 rounded-lg" />
          ) : missingChart.data?.chart ? (
            <MissingValuesChart chart={missingChart.data.chart} height={200} />
          ) : null}
        </Card>
      )}

      {/* Bottom row: Recent Activity + Model Status + Quick Actions */}
      <div className="grid gap-4 md:grid-cols-3">
        {/* Recent Activity */}
        <Card className="p-4">
          <h3 className="text-[13px] font-bold text-ink mb-3 flex items-center gap-1.5">
            <Activity size={14} className="text-brand" />
            Recent Activity
          </h3>
          <div className="space-y-2">
            {activities.length === 0 && (
              <p className="text-[12px] text-muted">No activity yet</p>
            )}
            {activities.map((act, i) => (
              <div key={i} className="flex items-center gap-2.5 py-1.5 border-b border-line last:border-0">
                <div className="w-7 h-7 rounded-lg border border-brand/25 bg-brand/10 flex items-center justify-center shrink-0">
                  <act.icon size={13} className="text-brand-mid" />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-[12px] font-semibold text-ink">{act.text}</p>
                  <p className="text-[10.5px] text-muted truncate">{act.time}</p>
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* Model Status */}
        <Card className="p-4">
          <h3 className="text-[13px] font-bold text-ink mb-3 flex items-center gap-1.5">
            <Brain size={14} className="text-brand" />
            Model Status
          </h3>
          {model ? (
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <div className="w-9 h-9 rounded-xl border border-brand/35 bg-brand/10 flex items-center justify-center">
                  <Brain size={18} className="text-brand-mid" />
                </div>
                <div>
                  <p className="text-[13px] font-bold text-ink">{model.model_choice}</p>
                  <p className="text-[10.5px] text-muted">
                    {model.problem_type} · {model.n_train.toLocaleString()} rows · {model.feature_count} features
                  </p>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2">
                {Object.entries(model.metrics).slice(0, 4).map(([key, val], index) => (
                  <div key={key} className={`rounded-lg bg-canvas/60 border px-2.5 py-1.5 ${index === 0 ? 'border-brand/40' : 'border-line'}`}>
                    <p className="text-[11px] text-muted font-medium">{METRIC_LABELS[key] ?? key.toUpperCase()}</p>
                    <p className={`text-[15px] font-semibold tabular-nums ${index === 0 ? 'text-brand-mid' : 'text-ink'}`}>{typeof val === 'number' ? val.toFixed(val < 1 ? 2 : 1) : '—'}</p>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p className="text-[12px] text-muted">No model trained yet</p>
          )}
        </Card>

        {/* Quick Actions */}
        <Card className="p-4">
          <h3 className="text-[13px] font-bold text-ink mb-3 flex items-center gap-1.5">
            <Sparkles size={14} className="text-brand" />
            Quick Actions
          </h3>
          <div className="grid grid-cols-2 gap-2">
            {[
              { icon: Upload, label: 'Upload Dataset', to: '/upload' },
              { icon: BarChart3, label: 'Analyse Data', to: '/analysis' },
              { icon: Brain, label: 'Train Model', to: '/train' },
              { icon: Sparkles, label: 'Predict', to: model ? '/predict' : '/train' },
            ].map((action) => (
              <Link
                key={action.to + action.label}
                to={action.to}
                className="group flex flex-col items-center gap-1.5 p-3 rounded-xl border border-line bg-canvas/40 hover:bg-brand/10 hover:border-brand/50 hover:-translate-y-px hover:shadow-glow-brand transition-all duration-200"
              >
                <action.icon size={18} strokeWidth={1.8} className="text-ink/80 group-hover:text-brand-mid transition-colors" />
                <span className="text-[11.5px] font-medium text-muted group-hover:text-ink text-center">{action.label}</span>
              </Link>
            ))}
          </div>
        </Card>
      </div>
    </div>
  )
}
