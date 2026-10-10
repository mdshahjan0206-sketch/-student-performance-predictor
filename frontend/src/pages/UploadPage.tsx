import { useCallback, useEffect, useState } from 'react'
import { X } from 'lucide-react'
import { useNavigate } from 'react-router-dom'

import { BoxPlot, CorrelationHeatmap, DistributionHistogram, ScatterPlot } from '../charts'
import { Plot, useChartPalette } from '../charts/Plot'
import { DataTable } from '../components/DataTable'
import { FileDropzone } from '../components/FileDropzone'
import {
  Alert,
  Badge,
  Card,
  MetricCard,
  MetricGrid,
  PageHeader,
  SectionHeading,
  Skeleton,
  formatCell,
  formatNumber,
} from '../components/ui'
import { useAsync } from '../hooks/useAsync'
import { useWorkspace } from '../hooks/useWorkspace'
import {
  ApiRequestError,
  getBoxplot,
  getCorrelation,
  getDatasetIntelligence,
  getHistogram,
  getPreview,
  getScatter,
  uploadDataset,
} from '../services/api'
import type {
  BoxChart,
  ChartResponse,
  CorrelationChart,
  DatasetIntelligence,
  HistogramChart,
  IntelligenceProfile,
  PreviewPage,
  ScatterChart,
} from '../types/api'

// ── Feature Detail Drawer ────────────────────────────────────────────────
function FeatureDetailDrawer({
  profile,
  datasetId,
  onClose,
}: {
  profile: IntelligenceProfile
  datasetId: string
  onClose: () => void
}) {
  const [tab, setTab] = useState<'stats' | 'distribution'>('stats')
  const histData = useAsync<ChartResponse<HistogramChart>>(
    () => getHistogram(datasetId, profile.name),
    [datasetId, profile.name],
    { enabled: tab === 'distribution' },
  )
  const boxData = useAsync<ChartResponse<BoxChart>>(
    () => getBoxplot(datasetId, profile.name),
    [datasetId, profile.name],
    { enabled: tab === 'distribution' && profile.kind === 'numeric' },
  )

  return (
    <div className="fixed inset-0 z-50 flex justify-end" role="dialog" aria-label={`${profile.name} details`}>
      <div className="absolute inset-0 bg-ink/30 backdrop-blur-sm animate-fade-in" onClick={onClose} />
      <aside className="relative w-full max-w-lg panel overflow-y-auto animate-scale-in m-3 ml-0 flex flex-col" style={{ transformOrigin: 'right' }}>
        <div className="p-5 border-b border-line/70">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <h2 className="font-display text-[20px] font-bold text-ink truncate">{profile.name}</h2>
              <p className="text-[13px] text-muted mt-0.5 font-mono">{profile.dtype}</p>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <Badge tone={profile.kind === 'numeric' ? 'brand' : profile.kind === 'text' ? 'info' : 'neutral'}>
                {profile.kind}
              </Badge>
              <button type="button" onClick={onClose} className="btn-ghost h-8 w-8 p-0 rounded-full text-muted hover:text-ink" aria-label="Close">
                <X size={16} aria-hidden />
              </button>
            </div>
          </div>
        </div>

        <div className="p-5 flex-1 space-y-5">
          {/* Meaning */}
          <div className="card p-4">
            <p className="text-[12px] font-semibold text-muted uppercase tracking-wide mb-1">Meaning</p>
            <p className="text-[13px] text-ink">{profile.meaning}</p>
          </div>

          {/* Tab selector */}
          <div className="inline-flex p-0.5 bg-canvas border border-line rounded-lg">
            {(['stats', 'distribution'] as const).map((t) => (
              <button
                key={t}
                type="button"
                onClick={() => setTab(t)}
                className={`px-3 h-8 rounded-[7px] text-[13px] font-semibold transition-colors ${
                  tab === t ? 'bg-surface text-ink shadow-card' : 'text-muted hover:text-ink'
                }`}
              >
                {t === 'stats' ? 'Statistics' : 'Distribution'}
              </button>
            ))}
          </div>

          {tab === 'stats' && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <MetricCard label="Missing" value={profile.missing.toLocaleString()} hint={`${profile.missing_pct}%`} tone={profile.missing > 0 ? 'warn' : 'default'} />
                <MetricCard label="Distinct" value={profile.distinct.toLocaleString()} />
              </div>

              {profile.kind === 'numeric' && (
                <div className="grid grid-cols-2 gap-3">
                  {profile.min != null && <MetricCard label="Min" value={formatNumber(profile.min)} />}
                  {profile.max != null && <MetricCard label="Max" value={formatNumber(profile.max)} />}
                  {profile.mean != null && <MetricCard label="Mean" value={formatNumber(profile.mean)} />}
                  {profile.median != null && <MetricCard label="Median" value={formatNumber(profile.median)} />}
                  {profile.std != null && <MetricCard label="Std Dev" value={formatNumber(profile.std)} />}
                  {profile.range != null && <MetricCard label="Range" value={formatNumber(profile.range)} />}
                </div>
              )}

              {profile.top_categories && profile.top_categories.length > 0 && (
                <div>
                  <p className="text-[12px] font-semibold text-muted uppercase tracking-wide mb-2">Top categories</p>
                  <div className="space-y-1.5">
                    {profile.top_categories.map((cat) => (
                      <div key={cat.value} className="flex items-center gap-3 text-[13px]">
                        <span className="text-ink font-medium truncate flex-1" title={cat.value}>{cat.value}</span>
                        <span className="text-muted tabular-nums shrink-0">{cat.count.toLocaleString()}</span>
                        <span className="text-muted tabular-nums shrink-0 w-14 text-right">{cat.pct}%</span>
                        <div className="w-20 h-1.5 bg-line rounded-full shrink-0 overflow-hidden">
                          <div className="h-full bg-brand rounded-full" style={{ width: `${Math.min(cat.pct, 100)}%` }} />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {profile.examples.length > 0 && (
                <div>
                  <p className="text-[12px] font-semibold text-muted uppercase tracking-wide mb-2">Example values</p>
                  <div className="flex flex-wrap gap-1.5">
                    {profile.examples.map((v, i) => (
                      <code key={i} className="px-2.5 py-1 rounded-md bg-canvas border border-line text-[12px] font-mono text-ink">
                        {formatCell(v)}
                      </code>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {tab === 'distribution' && (
            <div className="space-y-4">
              {histData.loading && <Skeleton className="h-72 rounded-card" />}
              {histData.data?.chart && <DistributionHistogram chart={histData.data.chart} height={260} />}
              {histData.data && !histData.data.chart && (
                <p className="text-[13px] text-muted py-6 text-center">{histData.data.message ?? 'Not available.'}</p>
              )}
              {boxData.data?.chart && <BoxPlot chart={boxData.data.chart} height={200} />}
            </div>
          )}
        </div>
      </aside>
    </div>
  )
}

/** Maps the shared palette onto the three series colours this page uses:
 *  numeric = green, categorical = blue, text = amber. */
function useChartPaletteCompat() {
  const c = useChartPalette()
  return { brand: c.brand, mid: c.blue, amber: c.amber }
}

// ── Feature Composition Donut ────────────────────────────────────────────
function FeatureCompositionChart({ intelligence }: { intelligence: DatasetIntelligence }) {
  const brandColors = useChartPaletteCompat()
  const { composition } = intelligence
  const labels = []
  const values = []
  const colors = []
  if (composition.numeric.count) { labels.push('Numeric'); values.push(composition.numeric.count); colors.push(brandColors.brand) }
  if (composition.categorical.count) { labels.push('Categorical'); values.push(composition.categorical.count); colors.push(brandColors.mid) }
  if (composition.text.count) { labels.push('Text'); values.push(composition.text.count); colors.push(brandColors.amber) }

  if (!values.length) return null

  return (
    <Plot
      title="Feature composition"
      height={260}
      data={[{
        type: 'pie',
        labels,
        values,
        hole: 0.55,
        marker: { colors },
        textinfo: 'label+percent',
        textfont: { size: 12 },
        hovertemplate: '%{label}: %{value} features (%{percent})<extra></extra>',
      }]}
      layout={{ margin: { l: 20, r: 20, t: 10, b: 10 }, showlegend: false }}
    />
  )
}

// ── Target Distribution Chart ────────────────────────────────────────────
function TargetDistribution({ intelligence }: { intelligence: DatasetIntelligence }) {
  const brandColors = useChartPaletteCompat()
  const target = intelligence.target_overview
  if (!target) return null

  if (target.problem_type === 'classification' && target.distribution) {
    return (
      <Plot
        title={`${target.column} distribution`}
        height={260}
        data={[{
          type: 'bar',
          x: target.distribution.labels,
          y: target.distribution.counts,
          marker: { color: brandColors.brand },
          text: target.distribution.counts.map(c => c.toLocaleString()),
          textposition: 'auto' as const,
          hovertemplate: '%{x}: %{y:,} (%{customdata}%)<extra></extra>',
          customdata: target.distribution.percentages,
        }]}
        layout={{
          xaxis: { title: { text: target.column } },
          yaxis: { title: { text: 'Count' } },
          bargap: 0.35,
        }}
      />
    )
  }

  if (target.problem_type === 'regression' && target.histogram) {
    return (
      <Plot
        title={`${target.column} distribution`}
        height={260}
        data={[{
          type: 'bar',
          x: target.histogram.bins,
          y: target.histogram.counts,
          marker: { color: brandColors.brand },
          hovertemplate: '%{x}<br>%{y:,} rows<extra></extra>',
        }]}
        layout={{
          xaxis: { title: { text: target.column } },
          yaxis: { title: { text: 'Count' } },
          bargap: 0.04,
        }}
      />
    )
  }

  return null
}

// ── Main Upload Page ─────────────────────────────────────────────────────
export function UploadPage() {
  const brandColors = useChartPaletteCompat()
  const { dataset, setDataset } = useWorkspace()
  const navigate = useNavigate()

  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [justUploaded, setJustUploaded] = useState(false)

  const [preview, setPreview] = useState<PreviewPage | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(10)
  const [search, setSearch] = useState('')
  const [searchInput, setSearchInput] = useState('')

  const [selectedFeature, setSelectedFeature] = useState<IntelligenceProfile | null>(null)

  // Intelligence data
  const intelligence = useAsync<DatasetIntelligence>(
    () => getDatasetIntelligence(dataset!.dataset_id),
    [dataset?.dataset_id],
    { enabled: Boolean(dataset) },
  )

  // Correlation data
  const corrData = useAsync<ChartResponse<CorrelationChart>>(
    () => getCorrelation(dataset!.dataset_id),
    [dataset?.dataset_id],
    { enabled: Boolean(dataset) },
  )

  // Scatter explore state
  const [scatterX, setScatterX] = useState('')
  const [scatterY, setScatterY] = useState('')
  const scatterData = useAsync<ChartResponse<ScatterChart>>(
    () => getScatter(dataset!.dataset_id, scatterX, scatterY),
    [dataset?.dataset_id, scatterX, scatterY],
    { enabled: Boolean(dataset) && Boolean(scatterX) && Boolean(scatterY) && scatterX !== scatterY },
  )

  // Set default scatter columns when intelligence loads
  useEffect(() => {
    if (intelligence.data?.composition.numeric.columns) {
      const numCols = intelligence.data.composition.numeric.columns
      if (numCols.length >= 2 && !scatterX && !scatterY) {
        setScatterX(numCols[0])
        setScatterY(numCols[1])
      }
    }
  }, [intelligence.data, scatterX, scatterY])

  // Debounce the search box so typing does not fire a request per keystroke.
  useEffect(() => {
    const timer = setTimeout(() => {
      setSearch(searchInput)
      setPage(1)
    }, 250)
    return () => clearTimeout(timer)
  }, [searchInput])

  const loadPreview = useCallback(
    async (datasetId: string, nextPage: number, size: number, term: string) => {
      setPreviewLoading(true)
      try {
        setPreview(await getPreview(datasetId, nextPage, size, term))
      } catch (err) {
        setError(err instanceof ApiRequestError ? err.message : 'The preview could not be loaded.')
      } finally {
        setPreviewLoading(false)
      }
    },
    [],
  )

  useEffect(() => {
    if (dataset) void loadPreview(dataset.dataset_id, page, pageSize, search)
  }, [dataset, page, pageSize, search, loadPreview])

  async function handleFile(file: File) {
    setBusy(true)
    setProgress(0)
    setError(null)
    setJustUploaded(false)
    try {
      const { dataset: summary } = await uploadDataset(file, setProgress)
      setDataset(summary)
      setPage(1)
      setSearchInput('')
      setSearch('')
      setJustUploaded(true)
      setSelectedFeature(null)
      setScatterX('')
      setScatterY('')
    } catch (err) {
      setError(err instanceof ApiRequestError ? err.message : 'The file could not be uploaded.')
    } finally {
      setBusy(false)
      setProgress(0)
    }
  }

  const intel = intelligence.data

  return (
    <div className="space-y-7">
      <PageHeader
        eyebrow="Step 1 of 5"
        title="Upload dataset"
        description="Bring in a CSV or Excel file of student records. It is validated and summarised as soon as it arrives, and nothing leaves this machine."
      />

      <FileDropzone onFile={handleFile} busy={busy} progress={progress} />

      {error && (
        <Alert tone="danger" title="Upload stopped">
          {error}
        </Alert>
      )}

      {justUploaded && dataset && (
        <Alert tone="brand" title={`${dataset.filename} is ready`}>
          {dataset.num_rows.toLocaleString()} rows across {dataset.num_cols} columns. Any previously
          trained model has been cleared, because it was fitted on different data.
        </Alert>
      )}

      {!dataset && !busy && !error && (
        <Card className="text-center py-10">
          <h3 className="font-display text-[16px] font-bold text-ink">No dataset loaded</h3>
          <p className="text-[13px] text-muted mt-1.5 max-w-[46ch] mx-auto">
            Drop a CSV or Excel file above to start. Everything after this point unlocks once a file
            is loaded.
          </p>
        </Card>
      )}

      {dataset && (
        <>
          {/* ── Dataset Intelligence ─────────────────────────────────── */}
          {intel && (
            <>
              {/* Story */}
              <div>
                <SectionHeading
                  title="Your dataset at a glance"
                  description="Factual summary generated from the uploaded data."
                />
                <Card className="space-y-1.5">
                  {intel.story.map((line, i) => (
                    <p key={i} className="text-[13px] text-ink leading-relaxed">
                      {line}
                    </p>
                  ))}
                </Card>
              </div>

              {/* Dataset Intelligence Metrics */}
              <div>
                <SectionHeading title="Dataset intelligence" />
                <MetricGrid columns={6}>
                  <MetricCard label="Records" value={intel.rows.toLocaleString()} />
                  <MetricCard label="Features" value={intel.columns} />
                  <MetricCard label="Numeric" value={intel.numeric_features} />
                  <MetricCard label="Categorical" value={intel.categorical_features} />
                  <MetricCard label="Text" value={intel.text_features || '0'} />
                  <MetricCard
                    label="Missing"
                    value={intel.missing_values.toLocaleString()}
                    tone={intel.missing_values > 0 ? 'warn' : 'default'}
                    hint={`${intel.missing_percentage}%`}
                  />
                </MetricGrid>
              </div>

              {/* Feature Composition + Target Overview */}
              <div className="grid gap-4 lg:grid-cols-2">
                {/* Feature Composition */}
                <Card>
                  <SectionHeading title="Feature composition" />
                  <FeatureCompositionChart intelligence={intel} />
                  <div className="mt-3 flex flex-wrap gap-3 text-[13px]">
                    <div className="flex items-center gap-1.5">
                      <span className="w-3 h-3 rounded-full" style={{ background: brandColors.brand }} />
                      <span className="text-ink">Numeric ({intel.numeric_features})</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className="w-3 h-3 rounded-full" style={{ background: brandColors.mid }} />
                      <span className="text-ink">Categorical ({intel.categorical_features})</span>
                    </div>
                    {intel.text_features > 0 && (
                      <div className="flex items-center gap-1.5">
                        <span className="w-3 h-3 rounded-full" style={{ background: brandColors.amber }} />
                        <span className="text-ink">Text ({intel.text_features})</span>
                      </div>
                    )}
                  </div>
                </Card>

                {/* Target Overview */}
                <Card>
                  <SectionHeading title="Target overview" />
                  {intel.target_overview ? (
                    <div className="space-y-4">
                      <div className="grid grid-cols-2 gap-3">
                        <MetricCard label="Target" value={intel.target_overview.column} tone="brand" />
                        <MetricCard label="Type" value={intel.target_overview.problem_type} />
                      </div>
                      <TargetDistribution intelligence={intel} />
                      {intel.target_overview.imbalance && (
                        <Alert tone={intel.target_overview.imbalance.detected ? 'warn' : 'brand'}>
                          {intel.target_overview.imbalance.description}
                        </Alert>
                      )}
                      {intel.target_overview.problem_type === 'regression' && (
                        <div className="grid grid-cols-3 gap-3">
                          {intel.target_overview.min != null && <MetricCard label="Min" value={formatNumber(intel.target_overview.min)} />}
                          {intel.target_overview.mean != null && <MetricCard label="Mean" value={formatNumber(intel.target_overview.mean)} />}
                          {intel.target_overview.max != null && <MetricCard label="Max" value={formatNumber(intel.target_overview.max)} />}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="py-6 text-center">
                      <p className="text-[13px] text-muted">No standard target column was automatically detected.</p>
                      <p className="text-[12px] text-muted mt-1">Choose the target column on the training page.</p>
                    </div>
                  )}
                </Card>
              </div>

              {/* Data Quality */}
              <div>
                <SectionHeading title="Dataset quality" />
                <Card>
                  <div className="flex items-center gap-4 mb-4">
                    <div className="relative w-16 h-16 shrink-0">
                      <svg viewBox="0 0 36 36" className="w-full h-full -rotate-90">
                        <circle cx="18" cy="18" r="15.915" fill="none" stroke="rgb(var(--color-line))" strokeWidth="3" />
                        <circle
                          cx="18" cy="18" r="15.915" fill="none"
                          stroke={intel.quality.score >= 80 ? 'rgb(var(--color-brand))' : intel.quality.score >= 50 ? 'rgb(var(--color-amber))' : 'rgb(var(--color-danger))'}
                          strokeWidth="3" strokeLinecap="round"
                          strokeDasharray={`${intel.quality.score} ${100 - intel.quality.score}`}
                        />
                      </svg>
                      <span className="absolute inset-0 flex items-center justify-center font-display font-bold text-[15px] text-ink">{intel.quality.score}</span>
                    </div>
                    <div className="min-w-0">
                      <p className="text-[15px] font-semibold text-ink">Quality score: {intel.quality.score}/100</p>
                      <p className="text-[12px] text-muted mt-0.5 max-w-[52ch]">{intel.quality.score_explanation}</p>
                    </div>
                  </div>

                  <div className="space-y-2">
                    {intel.quality.observations.map((obs, i) => (
                      <div key={i} className="flex items-start gap-2 text-[13px]">
                        <span className={`shrink-0 mt-0.5 w-1.5 h-1.5 rounded-full ${
                          obs.type === 'good' ? 'bg-brand' : obs.type === 'warn' ? 'bg-amber' : 'bg-muted/40'
                        }`} />
                        <span className="text-ink">{obs.text}</span>
                      </div>
                    ))}
                  </div>

                  {intel.quality.missing_columns.length > 0 && (
                    <div className="mt-4">
                      <p className="text-[12px] font-semibold text-muted uppercase tracking-wide mb-2">Missing values by column</p>
                      <div className="grid gap-1.5 sm:grid-cols-2 lg:grid-cols-3">
                        {intel.quality.missing_columns.slice(0, 9).map((mc) => (
                          <div key={mc.column} className="flex items-center justify-between gap-2 px-3 h-8 rounded-lg bg-canvas border border-line text-[12px]">
                            <span className="text-ink truncate" title={mc.column}>{mc.column}</span>
                            <span className="text-amber font-semibold tabular-nums shrink-0">{mc.count.toLocaleString()} ({mc.pct}%)</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </Card>
              </div>

              {/* Feature Explorer */}
              <div>
                <SectionHeading
                  title="Feature explorer"
                  description={`${intel.columns} features — click any card to see details, distribution, and statistics.`}
                />
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                  {intel.profiles.map((profile) => (
                    <button
                      type="button"
                      key={profile.name}
                      className="card card-hover p-4 text-left cursor-pointer w-full"
                      onClick={() => setSelectedFeature(profile)}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <p className="font-semibold text-[13px] text-ink truncate" title={profile.name}>
                          {profile.name}
                        </p>
                        <Badge tone={profile.kind === 'numeric' ? 'brand' : profile.kind === 'text' ? 'info' : 'neutral'}>
                          {profile.kind}
                        </Badge>
                      </div>
                      <p className="text-[11px] text-muted mt-1 line-clamp-2">{profile.meaning}</p>
                      <dl className="mt-3 space-y-1 text-[12px]">
                        <div className="flex justify-between gap-2">
                          <dt className="text-muted">Missing</dt>
                          <dd className={profile.missing > 0 ? 'text-amber font-semibold tabular-nums' : 'text-ink tabular-nums'}>
                            {profile.missing.toLocaleString()} ({profile.missing_pct}%)
                          </dd>
                        </div>
                        <div className="flex justify-between gap-2">
                          <dt className="text-muted">Distinct</dt>
                          <dd className="text-ink tabular-nums">{profile.distinct.toLocaleString()}</dd>
                        </div>
                        {profile.kind === 'numeric' && profile.mean != null && (
                          <div className="flex justify-between gap-2">
                            <dt className="text-muted">Mean</dt>
                            <dd className="text-ink tabular-nums">{formatNumber(profile.mean)}</dd>
                          </div>
                        )}
                        {profile.kind === 'numeric' && profile.min != null && profile.max != null && (
                          <div className="flex justify-between gap-2">
                            <dt className="text-muted">Range</dt>
                            <dd className="text-ink tabular-nums">{formatNumber(profile.min)} – {formatNumber(profile.max)}</dd>
                          </div>
                        )}
                        {profile.top_categories && profile.top_categories.length > 0 && (
                          <div className="flex justify-between gap-2">
                            <dt className="text-muted">Top</dt>
                            <dd className="text-ink truncate" title={profile.top_categories[0].value}>
                              {profile.top_categories[0].value} ({profile.top_categories[0].pct}%)
                            </dd>
                          </div>
                        )}
                      </dl>
                    </button>
                  ))}
                </div>
              </div>

              {/* Feature Relationships */}
              <div className="grid gap-4 xl:grid-cols-2">
                {/* Correlation Heatmap */}
                {corrData.data?.chart && (
                  <Card>
                    <SectionHeading
                      title="Feature relationships"
                      description="Pearson correlation between numeric columns. Observed relationships, not causation."
                    />
                    <CorrelationHeatmap chart={corrData.data.chart} height={380} />
                  </Card>
                )}

                {/* Scatter Plot */}
                {intel.composition.numeric.count >= 2 && (
                  <Card>
                    <SectionHeading title="Feature scatter" />
                    <div className="flex gap-3 mb-3">
                      <select
                        className="input flex-1"
                        value={scatterX}
                        onChange={(e) => setScatterX(e.target.value)}
                        aria-label="X axis feature"
                      >
                        <option value="">X axis</option>
                        {intel.composition.numeric.columns.map((c) => (
                          <option key={c} value={c}>{c}</option>
                        ))}
                      </select>
                      <select
                        className="input flex-1"
                        value={scatterY}
                        onChange={(e) => setScatterY(e.target.value)}
                        aria-label="Y axis feature"
                      >
                        <option value="">Y axis</option>
                        {intel.composition.numeric.columns.map((c) => (
                          <option key={c} value={c}>{c}</option>
                        ))}
                      </select>
                    </div>
                    {scatterData.loading && <Skeleton className="h-72 rounded-card" />}
                    {scatterData.data?.chart && <ScatterPlot chart={scatterData.data.chart} height={300} />}
                    {scatterData.data && !scatterData.data.chart && (
                      <p className="text-[13px] text-muted py-8 text-center">{scatterData.data.message}</p>
                    )}
                    {(!scatterX || !scatterY || scatterX === scatterY) && !scatterData.loading && (
                      <p className="text-[13px] text-muted py-8 text-center">Select two different numeric columns to compare.</p>
                    )}
                  </Card>
                )}
              </div>

              {/* Dataset Insights */}
              {intel.insights.length > 0 && (
                <div>
                  <SectionHeading title="Dataset insights" description="Factual observations derived from the uploaded data." />
                  <Card>
                    <div className="space-y-2">
                      {intel.insights.map((insight, i) => (
                        <div key={i} className="flex items-start gap-2.5 text-[13px]">
                          <Badge tone={
                            insight.category === 'imbalance' ? 'warn'
                              : insight.category === 'correlation' ? 'brand'
                                : insight.category === 'target' ? 'brand'
                                  : 'neutral'
                          }>
                            {insight.category}
                          </Badge>
                          <span className="text-ink">{insight.text}</span>
                        </div>
                      ))}
                    </div>
                  </Card>
                </div>
              )}
            </>
          )}

          {/* Loading skeleton for intelligence */}
          {intelligence.loading && !intel && (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
              {Array.from({ length: 6 }).map((_, i) => (
                <Skeleton key={i} className="h-20 rounded-card" />
              ))}
            </div>
          )}

          {/* Preview */}
          <div>
            <SectionHeading
              title="Preview"
              description="Rows are fetched a page at a time, so the browser never holds the whole file."
              action={
                <input
                  type="search"
                  className="input w-64"
                  placeholder="Search all columns"
                  aria-label="Search dataset rows"
                  value={searchInput}
                  onChange={(event) => setSearchInput(event.target.value)}
                />
              }
            />
            <DataTable
              columns={preview?.columns ?? dataset.column_names}
              rows={preview?.rows ?? []}
              page={preview?.page ?? 1}
              totalPages={preview?.total_pages ?? 1}
              totalRows={preview?.total_rows ?? dataset.num_rows}
              pageSize={pageSize}
              loading={previewLoading}
              onPageChange={setPage}
              onPageSizeChange={(size) => {
                setPageSize(size)
                setPage(1)
              }}
              emptyMessage={search ? `No rows contain "${search}".` : 'No rows to show.'}
            />
          </div>

          <div className="flex flex-wrap gap-3">
            <button type="button" className="btn-primary" onClick={() => navigate('/analysis')}>
              Analyse this dataset
            </button>
            <button type="button" className="btn-secondary" onClick={() => navigate('/dashboard')}>
              View dashboard
            </button>
            <button type="button" className="btn-secondary" onClick={() => navigate('/train')}>
              Skip to training
            </button>
          </div>
        </>
      )}

      {/* Feature Detail Drawer */}
      {selectedFeature && dataset && (
        <FeatureDetailDrawer
          profile={selectedFeature}
          datasetId={dataset.dataset_id}
          onClose={() => setSelectedFeature(null)}
        />
      )}
    </div>
  )
}
