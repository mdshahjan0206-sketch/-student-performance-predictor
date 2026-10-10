import { useEffect, useMemo, useState } from 'react'
import { ArrowRight, CircleCheck, TriangleAlert, X } from 'lucide-react'
import { useNavigate } from 'react-router-dom'

import {
  BoxPlot,
  CorrelationHeatmap,
  DistributionHistogram,
  MissingValuesChart,
  ScatterPlot,
} from '../charts'
import { DataTable } from '../components/DataTable'
import {
  Alert,
  Badge,
  Card,
  Checkbox,
  ChartSkeleton,
  EmptyState,
  MetricCard,
  MetricGrid,
  PageHeader,
  SectionHeading,
  SegmentedControl,
  Select,
  Spinner,
  formatNumber,
} from '../components/ui'
import { useAsync } from '../hooks/useAsync'
import { useWorkspace } from '../hooks/useWorkspace'
import {
  ApiRequestError,
  cleanedDatasetDownloadUrl,
  cleaningReportDownloadUrl,
  getBoxplot,
  getCorrelation,
  getFilterOptions,
  getHistogram,
  getMissing,
  getScatter,
  getStatistics,
  previewFilters,
  refineDataset,
  sessionId,
} from '../services/api'
import type { FilterState, MissingColumnDetail, RefinementResult } from '../types/api'

function ChartFrame({
  loading,
  error,
  message,
  height = 320,
  children,
}: {
  loading: boolean
  error: string | null
  message?: string
  height?: number
  children: React.ReactNode
}) {
  if (loading) return <ChartSkeleton height={height} />
  if (error) return <Alert tone="danger">{error}</Alert>
  if (message) return <Alert tone="info">{message}</Alert>
  return <>{children}</>
}

export function AnalysisPage() {
  const { dataset, setDataset, note } = useWorkspace()
  const navigate = useNavigate()
  const datasetId = dataset?.dataset_id ?? ''

  const [refinementOpen, setRefinementOpen] = useState(false)
  const [refining, setRefining] = useState(false)
  const [refineSuccess, setRefineSuccess] = useState<RefinementResult | null>(null)
  const [showDetails, setShowDetails] = useState(false)

  const overview = dataset?.overview
  const numericColumns = overview?.numeric_cols ?? []
  const allColumns = dataset?.column_names ?? []

  const [distColumn, setDistColumn] = useState('')
  const [chartType, setChartType] = useState<'histogram' | 'box'>('histogram')
  const [scatterX, setScatterX] = useState('')
  const [scatterY, setScatterY] = useState('')

  useEffect(() => {
    if (!dataset) return
    setDistColumn(numericColumns[0] ?? allColumns[0] ?? '')
    setScatterX(numericColumns[0] ?? '')
    setScatterY(numericColumns[1] ?? '')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId])

  async function handleRefine() {
    if (!datasetId) return
    setRefining(true)
    try {
      const res = await refineDataset(datasetId)
      setRefineSuccess(res)
      if (dataset) {
        setDataset({ ...dataset, refined: true })
      }
      note(`Data refined: ${res.total_missing.toLocaleString()} missing values handled`)
    } catch {
      // handled gracefully
    } finally {
      setRefining(false)
    }
  }

  const histogram = useAsync(
    () => getHistogram(datasetId, distColumn),
    [datasetId, distColumn],
    { enabled: Boolean(datasetId && distColumn && chartType === 'histogram') },
  )
  const boxplot = useAsync(
    () => getBoxplot(datasetId, distColumn),
    [datasetId, distColumn],
    { enabled: Boolean(datasetId && distColumn && chartType === 'box') },
  )
  const correlation = useAsync(() => getCorrelation(datasetId), [datasetId], {
    enabled: Boolean(datasetId),
  })
  const missing = useAsync(() => getMissing(datasetId), [datasetId], { enabled: Boolean(datasetId) })
  const scatter = useAsync(
    () => getScatter(datasetId, scatterX, scatterY),
    [datasetId, scatterX, scatterY],
    { enabled: Boolean(datasetId && scatterX && scatterY && scatterX !== scatterY) },
  )
  const statistics = useAsync(() => getStatistics(datasetId), [datasetId], {
    enabled: Boolean(datasetId),
  })
  const filterOptions = useAsync(() => getFilterOptions(datasetId), [datasetId], {
    enabled: Boolean(datasetId),
  })

  // --- exploratory filters ------------------------------------------
  const [filtersOpen, setFiltersOpen] = useState(false)
  const [numericFilter, setNumericFilter] = useState<{ column: string; min: number; max: number } | null>(null)
  const [categoryColumn, setCategoryColumn] = useState('')
  const [categoryValues, setCategoryValues] = useState<string[]>([])
  const [dropMissing, setDropMissing] = useState(false)
  const [filterResult, setFilterResult] = useState<Awaited<ReturnType<typeof previewFilters>> | null>(null)
  const [filterBusy, setFilterBusy] = useState(false)
  const [filterError, setFilterError] = useState<string | null>(null)

  const options = filterOptions.data

  useEffect(() => {
    if (!options) return
    const first = options.numeric[0]
    setNumericFilter(first ? { column: first.name, min: first.min, max: first.max } : null)
    setCategoryColumn(options.categorical[0]?.name ?? '')
    setCategoryValues(options.categorical[0]?.values ?? [])
  }, [options])

  const activeNumericRange = useMemo(
    () => options?.numeric.find((column) => column.name === numericFilter?.column) ?? null,
    [options, numericFilter?.column],
  )
  const activeCategoryValues = useMemo(
    () => options?.categorical.find((column) => column.name === categoryColumn)?.values ?? [],
    [options, categoryColumn],
  )

  const currentFilters: FilterState = useMemo(
    () => ({
      numeric: numericFilter ? { column: numericFilter.column, min: numericFilter.min, max: numericFilter.max } : null,
      categorical:
        categoryColumn && categoryValues.length && categoryValues.length !== activeCategoryValues.length
          ? { column: categoryColumn, values: categoryValues }
          : null,
      drop_missing: dropMissing,
    }),
    [numericFilter, categoryColumn, categoryValues, activeCategoryValues.length, dropMissing],
  )

  async function runFilters() {
    if (!datasetId) return
    setFilterBusy(true)
    setFilterError(null)
    try {
      const result = await previewFilters(datasetId, currentFilters, 50)
      setFilterResult(result)
      note(`Previewed ${result.matched_rows.toLocaleString()} filtered rows`)
    } catch (err) {
      setFilterError(err instanceof ApiRequestError ? err.message : 'The filter preview failed.')
    } finally {
      setFilterBusy(false)
    }
  }

  async function downloadFiltered() {
    if (!datasetId) return
    setFilterError(null)
    try {
      const response = await fetch(`/api/dataset/${datasetId}/filter/download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Session-Id': sessionId() },
        body: JSON.stringify(currentFilters),
      })
      if (!response.ok) {
        const body = await response.json().catch(() => null)
        throw new ApiRequestError('DOWNLOAD_FAILED', body?.error?.message ?? 'The download could not be prepared.')
      }
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = 'filtered_preview.csv'
      anchor.click()
      URL.revokeObjectURL(url)
      note('Downloaded filtered preview')
    } catch (err) {
      setFilterError(err instanceof ApiRequestError ? err.message : 'The download could not be prepared.')
    }
  }

  if (!dataset) {
    return (
      <div className="space-y-6">
        <PageHeader eyebrow="Step 2 of 5" title="Data analysis" />
        <EmptyState
          title="No dataset loaded"
          description="Upload a CSV or Excel file to unlock the analysis dashboard."
          action={
            <button type="button" className="btn-primary" onClick={() => navigate('/upload')}>
              Go to upload
            </button>
          }
        />
      </div>
    )
  }

  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="Step 2 of 5"
        title="Data analysis"
        description="Explore distributions, correlations and data quality before training. Charts are computed on the server and cached, so moving between them costs nothing."
      />

      <div>
        <SectionHeading title="Dataset overview" />
        <MetricGrid columns={6}>
          <MetricCard label="Total rows" value={overview?.total_rows.toLocaleString() ?? '—'} />
          <MetricCard label="Total columns" value={overview?.total_cols ?? '—'} />
          <MetricCard label="Numeric columns" value={numericColumns.length} />
          <MetricCard label="Categorical columns" value={overview?.categorical_cols.length ?? 0} />
          <MetricCard
            label="Missing values"
            value={overview?.total_missing.toLocaleString() ?? '—'}
            tone={(overview?.total_missing ?? 0) > 0 ? 'warn' : 'default'}
          />
          <MetricCard
            label="Duplicate rows"
            value={overview?.duplicate_rows.toLocaleString() ?? '—'}
            tone={(overview?.duplicate_rows ?? 0) > 0 ? 'warn' : 'default'}
          />
        </MetricGrid>
      </div>

      <Card>
        <SectionHeading title="Distribution" description="How the values in one column are spread." />
        <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_auto] md:items-end mb-5">
          <Select
            id="dist-column"
            label="Column"
            value={distColumn}
            onChange={setDistColumn}
            options={allColumns.map((name) => ({ value: name, label: name }))}
          />
          <SegmentedControl
            name="chart-type"
            label="Chart"
            value={chartType}
            onChange={(value) => setChartType(value as 'histogram' | 'box')}
            options={[
              { value: 'histogram', label: 'Histogram' },
              { value: 'box', label: 'Box plot' },
            ]}
          />
        </div>

        {chartType === 'histogram' ? (
          <ChartFrame
            loading={histogram.loading}
            error={histogram.error}
            message={histogram.data && !histogram.data.available ? histogram.data.message : undefined}
          >
            {histogram.data?.chart && (
              <>
                <DistributionHistogram chart={histogram.data.chart} />
                {histogram.data.chart.truncated && (
                  <p className="text-[12px] text-muted mt-2">
                    Showing the 40 most frequent of {histogram.data.chart.distinct?.toLocaleString()} distinct
                    values.
                  </p>
                )}
              </>
            )}
          </ChartFrame>
        ) : (
          <ChartFrame
            loading={boxplot.loading}
            error={boxplot.error}
            message={boxplot.data && !boxplot.data.available ? boxplot.data.message : undefined}
          >
            {boxplot.data?.chart && (
              <>
                <BoxPlot chart={boxplot.data.chart} />
                <dl className="grid grid-cols-2 sm:grid-cols-5 gap-3 mt-4 text-[12px]">
                  {(
                    [
                      ['Minimum', boxplot.data.chart.min],
                      ['Lower quartile', boxplot.data.chart.q1],
                      ['Median', boxplot.data.chart.median],
                      ['Upper quartile', boxplot.data.chart.q3],
                      ['Maximum', boxplot.data.chart.max],
                    ] as const
                  ).map(([label, value]) => (
                    <div key={label} className="rounded-lg bg-canvas border border-line px-3 py-2">
                      <dt className="text-muted">{label}</dt>
                      <dd className="font-semibold text-ink tabular-nums">{formatNumber(value)}</dd>
                    </div>
                  ))}
                </dl>
              </>
            )}
          </ChartFrame>
        )}
      </Card>

      <Card>
        <SectionHeading
          title="Correlation"
          description="How the numeric columns move together. Values run from -1 to 1."
        />
        <ChartFrame
          loading={correlation.loading}
          error={correlation.error}
          message={correlation.data && !correlation.data.available ? correlation.data.message : undefined}
          height={440}
        >
          {correlation.data?.chart && (
            <>
              <CorrelationHeatmap chart={correlation.data.chart} />
              {correlation.data.chart.truncated && (
                <p className="text-[12px] text-muted mt-2">
                  Showing the first 40 of {correlation.data.chart.total_numeric} numeric columns.
                </p>
              )}
            </>
          )}
        </ChartFrame>
      </Card>

      <Card>
        {missing.loading ? (
          <ChartSkeleton height={320} />
        ) : missing.error ? (
          <Alert tone="danger">{missing.error}</Alert>
        ) : (
          (() => {
            const analysis = missing.data?.analysis
            const isClean = !analysis || analysis.total_missing_cells === 0

            return (
              <div className="space-y-6">
                {/* Conditional Missing Values Action Banner (Section 6 & 7 & 16) */}
                {!isClean && (
                  !dataset?.refined ? (
                    <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-xl border border-amber/30 bg-amber-soft/25 backdrop-blur-sm">
                      <div className="flex items-center gap-3">
                        <TriangleAlert size={20} className="text-amber shrink-0" aria-hidden />
                        <div>
                          <p className="font-semibold text-ink text-[14px]">Missing values detected</p>
                          <p className="text-[12px] text-muted">
                            {analysis.total_missing_cells.toLocaleString()} cells · Refinement required
                          </p>
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={() => setRefinementOpen(true)}
                        className="btn-primary text-[13px] px-5 py-2 shadow-sm"
                      >
                        Handle Missing Values
                      </button>
                    </div>
                  ) : (
                    <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-xl border border-brand/30 bg-brand-soft/25 backdrop-blur-sm">
                      <div className="flex items-center gap-3">
                        <CircleCheck size={20} className="text-brand-mid shrink-0" aria-hidden />
                        <div>
                          <p className="font-semibold text-ink text-[14px]">Missing values handled</p>
                          <p className="text-[12px] text-muted">
                            {analysis.total_missing_cells.toLocaleString()} cells refined · Ready for training
                          </p>
                        </div>
                      </div>
                      <div className="flex flex-wrap items-center gap-2">
                        <a
                          href={cleanedDatasetDownloadUrl(datasetId)}
                          download
                          className="btn-secondary text-[12px] px-3.5 py-1.5 inline-flex items-center gap-1.5"
                        >
                          <svg className="w-3.5 h-3.5 text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                          </svg>
                          Download Cleaned Dataset
                        </a>
                        <button
                          type="button"
                          onClick={() => setRefinementOpen(true)}
                          className="btn-secondary text-[12px] px-3 py-1.5"
                        >
                          View Details
                        </button>
                        <button
                          type="button"
                          onClick={() => navigate('/train')}
                          className="btn-primary text-[13px] px-4 py-1.5"
                        >
                          Continue to Train <ArrowRight size={14} aria-hidden />
                        </button>
                      </div>
                    </div>
                  )
                )}

                <SectionHeading
                  title="Missing Value Analysis"
                  action={
                    <div className="flex flex-wrap items-center gap-2">
                      <Badge tone={isClean ? 'brand' : 'warn'}>
                        {isClean
                          ? <><CircleCheck size={12} aria-hidden /> Quality good</>
                          : <><TriangleAlert size={12} aria-hidden /> Needs cleaning</>}
                      </Badge>
                      {!isClean && (
                        <>
                          <span className="text-[12px] text-muted font-medium">
                            {analysis.affected_columns_count} columns · {analysis.total_missing_cells.toLocaleString()} missing
                          </span>
                          <button
                            type="button"
                            onClick={() => setShowDetails(!showDetails)}
                            className="text-[12px] font-semibold text-brand hover:underline px-1"
                          >
                            {showDetails ? 'Hide details' : '[ View details ]'}
                          </button>
                        </>
                      )}
                      <a
                        href={cleaningReportDownloadUrl(datasetId)}
                        download={`cleaning_report_${datasetId}.csv`}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-panel border border-line hover:border-brand/40 text-[12px] font-medium text-ink transition-colors shadow-sm"
                      >
                        <svg className="w-3.5 h-3.5 text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                        </svg>
                        Download Cleaning Report
                      </a>
                    </div>
                  }
                />

                {analysis && (
                  <>
                    <MetricGrid columns={5}>
                      <MetricCard
                        label="Total missing cells"
                        value={analysis.total_missing_cells.toLocaleString()}
                        tone={analysis.total_missing_cells > 0 ? 'warn' : 'default'}
                      />
                      <MetricCard
                        label="Columns with missing"
                        value={`${analysis.affected_columns_count} / ${analysis.total_cols}`}
                        tone={analysis.affected_columns_count > 0 ? 'warn' : 'default'}
                      />
                      <MetricCard
                        label="Rows containing missing"
                        value={analysis.rows_with_missing.toLocaleString()}
                        hint={`${analysis.rows_with_missing_pct}% of rows`}
                        tone={analysis.rows_with_missing > 0 ? 'warn' : 'default'}
                      />
                      <MetricCard
                        label="Highest missing column"
                        value={analysis.highest_missing_column ? analysis.highest_missing_column.column : 'None'}
                        hint={
                          analysis.highest_missing_column
                            ? `${analysis.highest_missing_column.count.toLocaleString()} cells (${analysis.highest_missing_column.percentage}%)`
                            : 'All columns complete'
                        }
                      />
                      <MetricCard
                        label="Overall missing rate"
                        value={`${analysis.overall_missing_pct}%`}
                        hint={`${analysis.total_missing_cells.toLocaleString()} of ${analysis.total_cells.toLocaleString()} cells`}
                      />
                    </MetricGrid>

                    {/* Progressive Disclosure: Explanatory Message behind View Details */}
                    {showDetails && (
                      <div className="rounded-xl border border-brand/20 bg-brand-soft/40 p-4 text-[13px] leading-relaxed text-ink space-y-2 animate-fade-in">
                        <div className="flex items-start gap-2.5">
                          <span className="text-brand font-bold text-[15px] leading-none mt-0.5">ℹ</span>
                          <div className="space-y-1">
                            <p className="font-medium text-ink">{analysis.explanation_message}</p>
                            <p className="text-muted text-[12px]">{analysis.dynamic_explanation}</p>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Compact Before -> After Cleaning Card */}
                    <div className="grid sm:grid-cols-[1fr_auto_1fr] items-center gap-3 p-4 rounded-xl border border-line bg-canvas/60">
                      <div className="space-y-1 p-3 rounded-lg bg-panel border border-line">
                        <p className="text-[11px] font-bold uppercase tracking-wider text-muted">Before Cleaning</p>
                        <p className="text-[18px] font-extrabold text-ink tabular-nums">
                          {analysis.before_after.before_missing_cells.toLocaleString()} <span className="text-[13px] font-normal text-muted">missing cells</span>
                        </p>
                        <p className="text-[11px] text-muted">
                          {analysis.affected_columns_count} incomplete columns in original upload
                        </p>
                      </div>

                      <div className="flex flex-col items-center justify-center px-2 py-1 text-center">
                        <span className="text-[11px] font-semibold text-brand px-2.5 py-1 rounded-full bg-brand-soft border border-brand/20">
                          Data Cleaning Pipeline
                        </span>
                        <span className="text-[11px] text-muted mt-1">Median (Num) · Most-Frequent (Cat)</span>
                        <ArrowRight size={14} className="text-muted mt-0.5" aria-hidden />
                      </div>

                      <div className="space-y-1 p-3 rounded-lg bg-panel border border-brand/30">
                        <p className="text-[11px] font-bold uppercase tracking-wider text-brand">After Cleaning</p>
                        <p className="text-[18px] font-extrabold text-brand-deep tabular-nums">
                          {analysis.before_after.after_projected_feature_missing}{' '}
                          <span className="text-[13px] font-normal text-muted">missing feature values</span>
                        </p>
                        <p className="text-[11px] text-brand-deep font-medium">
                          <CircleCheck size={12} className="inline -mt-0.5 mr-1" aria-hidden />Ready for training
                        </p>
                      </div>
                    </div>

                    {/* Columns Handling Table */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <h3 className="text-[14px] font-semibold text-ink">Feature Quality &amp; Imputation Strategy</h3>
                        <span className="text-[12px] text-muted">{analysis.details.length} columns inspected</span>
                      </div>
                      <div className="table-scroll border border-line rounded-lg">
                        <table className="w-full text-[13px] border-collapse">
                          <thead>
                            <tr className="bg-canvas">
                              <th scope="col" className="px-3 py-2.5 border-b border-line font-semibold text-left text-ink">Column</th>
                              <th scope="col" className="px-3 py-2.5 border-b border-line font-semibold text-left text-ink">Type</th>
                              <th scope="col" className="px-3 py-2.5 border-b border-line font-semibold text-right text-ink">Missing</th>
                              <th scope="col" className="px-3 py-2.5 border-b border-line font-semibold text-left text-ink min-w-[130px]">Missing %</th>
                              <th scope="col" className="px-3 py-2.5 border-b border-line font-semibold text-left text-ink">Severity</th>
                              <th scope="col" className="px-3 py-2.5 border-b border-line font-semibold text-left text-ink">Handling</th>
                              <th scope="col" className="px-3 py-2.5 border-b border-line font-semibold text-center text-ink">Status</th>
                            </tr>
                          </thead>
                          <tbody>
                            {analysis.details.map((detail: MissingColumnDetail) => {
                              const sevTone: 'neutral' | 'info' | 'warn' | 'danger' =
                                detail.severity === 'None'
                                  ? 'neutral'
                                  : detail.severity === 'Low'
                                  ? 'info'
                                  : detail.severity === 'Moderate'
                                  ? 'warn'
                                  : 'danger'

                              return (
                                <tr key={detail.column} className="hover:bg-canvas/70 transition-colors">
                                  <td className="px-3 py-2.5 border-b border-line font-medium text-ink">
                                    {detail.column}
                                  </td>
                                  <td className="px-3 py-2.5 border-b border-line text-muted">
                                    <span className="inline-block px-2 py-0.5 rounded text-[11px] font-mono bg-canvas border border-line">
                                      {detail.type}
                                    </span>
                                  </td>
                                  <td className="px-3 py-2.5 border-b border-line text-right tabular-nums font-semibold text-ink">
                                    {detail.missing_count.toLocaleString()}
                                  </td>
                                  <td className="px-3 py-2.5 border-b border-line text-left">
                                    <div className="flex items-center gap-2">
                                      <div className="w-16 h-2 rounded-full bg-line overflow-hidden shrink-0">
                                        <div
                                          className={`h-full ${detail.missing_pct > 20 ? 'bg-danger' : detail.missing_pct > 5 ? 'bg-amber' : 'bg-brand'}`}
                                          style={{ width: `${Math.min(100, Math.max(detail.missing_pct > 0 ? 5 : 0, detail.missing_pct))}%` }}
                                        />
                                      </div>
                                      <span className="text-[12px] tabular-nums font-medium text-ink">{detail.missing_pct.toFixed(2)}%</span>
                                    </div>
                                  </td>
                                  <td className="px-3 py-2.5 border-b border-line">
                                    <Badge tone={sevTone}>{detail.severity}</Badge>
                                  </td>
                                  <td className="px-3 py-2.5 border-b border-line text-muted font-medium">
                                    {detail.handling}
                                  </td>
                                  <td className="px-3 py-2.5 border-b border-line text-center">
                                    {detail.final_status === 'Clean' ? (
                                      <span className="inline-flex items-center gap-1 text-ok font-semibold text-[13px]" title="No missing values"><CircleCheck size={13} aria-hidden />Clean</span>
                                    ) : detail.final_status === 'Excluded' ? (
                                      <span className="inline-flex items-center gap-1 text-danger font-semibold text-[13px]" title="Excluded from model features"><X size={13} aria-hidden />Excluded</span>
                                    ) : (
                                      <span className="inline-flex items-center gap-1 text-brand-mid font-semibold text-[13px]" title="Automatically imputed"><CircleCheck size={13} aria-hidden />Handled</span>
                                    )}
                                  </td>
                                </tr>
                              )
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    {/* Distribution chart */}
                    {missing.data?.chart && (
                      <div className="pt-4 border-t border-line">
                        <h3 className="text-[14px] font-semibold text-ink mb-3">Missing Values Distribution Chart</h3>
                        <MissingValuesChart chart={missing.data.chart} />
                      </div>
                    )}
                  </>
                )}
              </div>
            )
          })()
        )}
      </Card>

      <Card>
        <SectionHeading title="Scatter comparison" description="Compare two numeric columns directly." />
        {numericColumns.length < 2 ? (
          <Alert tone="info">A scatter plot needs at least two numeric columns.</Alert>
        ) : (
          <>
            <div className="grid gap-4 sm:grid-cols-2 mb-5">
              <Select
                id="scatter-x"
                label="Horizontal axis"
                value={scatterX}
                onChange={(value) => {
                  setScatterX(value)
                  if (value === scatterY) {
                    setScatterY(numericColumns.find((column) => column !== value) ?? '')
                  }
                }}
                options={numericColumns.map((name) => ({ value: name, label: name }))}
              />
              <Select
                id="scatter-y"
                label="Vertical axis"
                value={scatterY}
                onChange={setScatterY}
                options={numericColumns
                  .filter((name) => name !== scatterX)
                  .map((name) => ({ value: name, label: name }))}
              />
            </div>
            <ChartFrame
              loading={scatter.loading}
              error={scatter.error}
              message={scatter.data && !scatter.data.available ? scatter.data.message : undefined}
              height={380}
            >
              {scatter.data?.chart && (
                <>
                  <ScatterPlot chart={scatter.data.chart} />
                  {scatter.data.chart.sampled && (
                    <p className="text-[12px] text-muted mt-2">
                      Plotting a fixed sample of {scatter.data.chart.plotted_points.toLocaleString()} of{' '}
                      {scatter.data.chart.total_points.toLocaleString()} points. The sample is the same every
                      time, and training always uses every row.
                    </p>
                  )}
                </>
              )}
            </ChartFrame>
          </>
        )}
      </Card>

      <Card>
        <SectionHeading title="Numeric summary" description="Descriptive statistics for every numeric column." />
        {statistics.loading && <ChartSkeleton height={160} />}
        {statistics.error && <Alert tone="danger">{statistics.error}</Alert>}
        {statistics.data &&
          (statistics.data.rows.length === 0 ? (
            <Alert tone="info">This dataset has no numeric columns to summarise.</Alert>
          ) : (
            <div className="table-scroll border border-line rounded-lg">
              <table className="w-full text-[13px] border-collapse">
                <thead>
                  <tr className="bg-canvas">
                    {['Column', 'Count', 'Mean', 'Std dev', 'Min', '25%', 'Median', '75%', 'Max'].map((head) => (
                      <th
                        key={head}
                        scope="col"
                        className={`px-3 py-2.5 border-b border-line font-semibold text-ink ${head === 'Column' ? 'text-left' : 'text-right'}`}
                      >
                        {head}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {statistics.data.rows.map((row) => (
                    <tr key={row.column} className="hover:bg-canvas/70">
                      <td className="px-3 py-2 border-b border-line font-medium text-ink">{row.column}</td>
                      {([row.count, row.mean, row.std, row.min, row.p25, row.p50, row.p75, row.max] as const).map(
                        (value, index) => (
                          <td key={index} className="px-3 py-2 border-b border-line text-right tabular-nums text-ink">
                            {formatNumber(value, 2)}
                          </td>
                        ),
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
      </Card>

      <Card>
        <SectionHeading
          title="Filters"
          description="Try a subset of the data and download it. Training and prediction always use the full dataset."
          action={
            <button
              type="button"
              className="btn-secondary"
              aria-expanded={filtersOpen}
              onClick={() => setFiltersOpen((open) => !open)}
            >
              {filtersOpen ? 'Hide filters' : 'Show filters'}
            </button>
          }
        />

        {filtersOpen && (
          <div className="animate-rise">
            <Alert tone="info">
              These filters preview a subset only. The stored dataset is never replaced, so the model
              you train still sees all {dataset.num_rows.toLocaleString()} rows.
            </Alert>

            <div className="grid gap-5 lg:grid-cols-3 mt-5">
              <div className="space-y-3">
                <Select
                  id="filter-numeric"
                  label="Numeric column"
                  value={numericFilter?.column ?? ''}
                  onChange={(value) => {
                    const column = options?.numeric.find((item) => item.name === value)
                    if (column) setNumericFilter({ column: column.name, min: column.min, max: column.max })
                  }}
                  options={(options?.numeric ?? []).map((column) => ({ value: column.name, label: column.name }))}
                  hint={options?.numeric.length ? undefined : 'No numeric columns available'}
                />
                {numericFilter && activeNumericRange && (
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label htmlFor="filter-min" className="field-label">
                        From
                      </label>
                      <input
                        id="filter-min"
                        type="number"
                        className="input tabular-nums"
                        value={numericFilter.min}
                        min={activeNumericRange.min}
                        max={activeNumericRange.max}
                        onChange={(event) =>
                          setNumericFilter({ ...numericFilter, min: Number(event.target.value) })
                        }
                      />
                    </div>
                    <div>
                      <label htmlFor="filter-max" className="field-label">
                        To
                      </label>
                      <input
                        id="filter-max"
                        type="number"
                        className="input tabular-nums"
                        value={numericFilter.max}
                        min={activeNumericRange.min}
                        max={activeNumericRange.max}
                        onChange={(event) =>
                          setNumericFilter({ ...numericFilter, max: Number(event.target.value) })
                        }
                      />
                    </div>
                  </div>
                )}
              </div>

              <div className="space-y-3">
                <Select
                  id="filter-category"
                  label="Category column"
                  value={categoryColumn}
                  onChange={(value) => {
                    setCategoryColumn(value)
                    setCategoryValues(
                      options?.categorical.find((column) => column.name === value)?.values ?? [],
                    )
                  }}
                  options={(options?.categorical ?? []).map((column) => ({
                    value: column.name,
                    label: column.name,
                  }))}
                  hint={options?.categorical.length ? undefined : 'No categorical columns available'}
                />
                {activeCategoryValues.length > 0 && (
                  <div className="max-h-44 overflow-y-auto rounded-lg border border-line p-3 space-y-1.5">
                    {activeCategoryValues.map((value) => (
                      <Checkbox
                        key={value}
                        id={`cat-${value}`}
                        label={value}
                        checked={categoryValues.includes(value)}
                        onChange={(checked) =>
                          setCategoryValues((current) =>
                            checked ? [...current, value] : current.filter((item) => item !== value),
                          )
                        }
                      />
                    ))}
                  </div>
                )}
              </div>

              <div className="space-y-4">
                <Checkbox
                  id="drop-missing"
                  label="Remove rows with missing values"
                  checked={dropMissing}
                  onChange={setDropMissing}
                />
                <div className="flex flex-wrap gap-2">
                  <button type="button" className="btn-primary" onClick={runFilters} disabled={filterBusy}>
                    {filterBusy ? 'Filtering' : 'Apply filters'}
                  </button>
                  <button
                    type="button"
                    className="btn-secondary"
                    onClick={downloadFiltered}
                    disabled={!filterResult || filterResult.matched_rows === 0}
                  >
                    Download CSV
                  </button>
                </div>
                {filterBusy && <Spinner label="Evaluating on the server" />}
              </div>
            </div>

            {filterError && (
              <div className="mt-4">
                <Alert tone="danger">{filterError}</Alert>
              </div>
            )}

            {filterResult && (
              <div className="mt-6">
                <div className="flex items-center gap-3 mb-3">
                  <Badge tone={filterResult.changed ? 'brand' : 'neutral'}>
                    {filterResult.matched_rows.toLocaleString()} of{' '}
                    {filterResult.total_rows.toLocaleString()} rows
                  </Badge>
                  {!filterResult.changed && <span className="text-[12px] text-muted">No filters are narrowing this view.</span>}
                </div>
                <DataTable
                  columns={filterResult.columns}
                  rows={filterResult.rows}
                  page={1}
                  totalPages={1}
                  totalRows={filterResult.preview_rows}
                  pageSize={filterResult.preview_rows || 1}
                  onPageChange={() => undefined}
                  emptyMessage="No rows match these filters."
                />
                <p className="text-[12px] text-muted mt-2">
                  Showing the first {filterResult.preview_rows.toLocaleString()} matching rows. The CSV
                  download contains every matching row.
                </p>
              </div>
            )}
          </div>
        )}
      </Card>

      <div className="flex flex-wrap gap-3">
        <button type="button" className="btn-secondary" onClick={() => navigate('/upload')}>
          Back to upload
        </button>
        <button type="button" className="btn-primary" onClick={() => navigate('/train')}>
          Train a model
        </button>
      </div>

      {/* Refinement Modal / Drawer (Section 8, 9, 12, 13, 16, 17, 18) */}
      {refinementOpen && missing.data?.analysis && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-ink/40 backdrop-blur-sm animate-fade-in">
          <div className="card max-w-2xl w-full max-h-[90vh] overflow-y-auto p-6 space-y-5 bg-panel border border-line shadow-2xl rounded-2xl">
            <div className="flex items-center justify-between pb-3 border-b border-line">
              <div>
                <h2 className="text-[18px] font-bold text-ink">Handling Missing Values</h2>
                <p className="text-[12px] text-muted mt-0.5">Automated type-aware imputation pipeline</p>
              </div>
              <button
                type="button"
                onClick={() => setRefinementOpen(false)}
                className="text-muted hover:text-ink text-[18px] p-1.5 rounded-lg hover:bg-canvas" aria-label="Close"
              >
                <X size={16} aria-hidden />
              </button>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 rounded-xl bg-canvas border border-line">
                <span className="text-[11px] font-bold uppercase text-muted tracking-wider">Missing cells</span>
                <p className="text-[20px] font-extrabold text-ink tabular-nums mt-0.5">
                  {missing.data.analysis.total_missing_cells.toLocaleString()}
                </p>
              </div>
              <div className="p-3 rounded-xl bg-canvas border border-line">
                <span className="text-[11px] font-bold uppercase text-muted tracking-wider">Columns affected</span>
                <p className="text-[20px] font-extrabold text-ink tabular-nums mt-0.5">
                  {missing.data.analysis.affected_columns_count}
                </p>
              </div>
            </div>

            {/* Target missing rows notice */}
            {dataset?.detected_targets?.some((t) => (dataset.missing_summary[t] ?? 0) > 0) && (
              <div className="p-3 rounded-lg border border-amber/30 bg-amber-soft/20 text-[12px] text-amber-deep space-y-0.5">
                <p className="font-semibold">
                  <TriangleAlert size={12} className="inline -mt-0.5 mr-1" aria-hidden />Missing target: {dataset.detected_targets.reduce((acc, t) => acc + (dataset.missing_summary[t] ?? 0), 0).toLocaleString()} rows affected
                </p>
                <p className="text-muted text-[11px]">Rows with missing target values are excluded from supervised training.</p>
              </div>
            )}

            {/* Compact table: Column | Type | Missing | Method */}
            <div className="space-y-1.5">
              <p className="text-[12px] font-semibold text-muted uppercase tracking-wider">Refinement Strategy</p>
              <div className="border border-line rounded-lg overflow-hidden max-h-56 overflow-y-auto">
                <table className="w-full text-[12px] border-collapse">
                  <thead>
                    <tr className="bg-canvas sticky top-0 border-b border-line text-muted">
                      <th className="px-3 py-2 text-left font-semibold text-ink">Column</th>
                      <th className="px-3 py-2 text-left font-semibold text-ink">Type</th>
                      <th className="px-3 py-2 text-right font-semibold text-ink">Missing</th>
                      <th className="px-3 py-2 text-left font-semibold text-ink">Method</th>
                    </tr>
                  </thead>
                  <tbody>
                    {missing.data.analysis.details
                      .filter((d) => d.missing_count > 0)
                      .map((d) => {
                        const method =
                          d.missing_count === missing.data?.analysis?.total_rows
                            ? 'Excluded (no usable data)'
                            : d.type === 'Numeric'
                            ? 'Median'
                            : 'Most frequent'
                        return (
                          <tr key={d.column} className="border-b border-line/50 hover:bg-canvas/40">
                            <td className="px-3 py-2 font-medium text-ink font-mono">{d.column}</td>
                            <td className="px-3 py-2 text-muted">{d.type === 'Numeric' ? 'Numeric' : 'Category'}</td>
                            <td className="px-3 py-2 text-right tabular-nums font-semibold text-ink">
                              {d.missing_count.toLocaleString()}
                            </td>
                            <td className="px-3 py-2 font-medium text-brand">{method}</td>
                          </tr>
                        )
                      })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Before / After */}
            <div className="flex items-center justify-between p-3 rounded-xl bg-canvas border border-line text-[12px]">
              <div className="space-y-0.5">
                <span className="text-[10px] uppercase font-bold text-muted">Before</span>
                <p className="font-extrabold text-ink tabular-nums">{missing.data.analysis.total_missing_cells.toLocaleString()} missing</p>
              </div>
              <div className="flex flex-col items-center">
                <span className="text-muted text-[10px]">Clean</span>
                <ArrowRight size={14} className="text-muted" aria-hidden />
              </div>
              <div className="space-y-0.5 text-right">
                <span className="text-[10px] uppercase font-bold text-brand">After</span>
                <p className="font-extrabold text-brand tabular-nums">0 feature missing</p>
              </div>
            </div>

            {/* Cleaning Summary if Refined */}
            {(dataset?.refined || refineSuccess) && (
              <div className="p-4 rounded-xl border border-brand/30 bg-brand-soft/20 space-y-3 animate-fade-in">
                <div className="flex items-center gap-2">
                  <CircleCheck size={16} className="text-brand-mid shrink-0" aria-hidden />
                  <div>
                    <p className="font-semibold text-ink text-[13px]">Data refined</p>
                    <p className="text-[11px] text-muted">Transformed features validated and ready for training</p>
                  </div>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px]">
                  <div className="p-2 rounded bg-panel border border-line">
                    <span className="text-muted block text-[10px]">Rows</span>
                    <span className="font-bold text-ink">{dataset?.num_rows.toLocaleString()}</span>
                  </div>
                  <div className="p-2 rounded bg-panel border border-line">
                    <span className="text-muted block text-[10px]">Columns</span>
                    <span className="font-bold text-ink">{dataset?.num_cols}</span>
                  </div>
                  <div className="p-2 rounded bg-panel border border-line">
                    <span className="text-muted block text-[10px]">Missing cells</span>
                    <span className="font-bold text-brand font-mono">{missing.data.analysis.total_missing_cells.toLocaleString()} handled</span>
                  </div>
                  <div className="p-2 rounded bg-panel border border-line">
                    <span className="text-muted block text-[10px]">Ready</span>
                    <span className="inline-flex items-center gap-1 font-bold text-brand-mid"><CircleCheck size={12} aria-hidden />Yes</span>
                  </div>
                </div>
              </div>
            )}

            {/* Action Footer */}
            <div className="flex items-center justify-between gap-3 pt-3 border-t border-line">
              <button
                type="button"
                onClick={() => setRefinementOpen(false)}
                className="btn-secondary text-[12px] px-4 py-2"
              >
                Close
              </button>

              <div className="flex items-center gap-2">
                {!dataset?.refined && !refineSuccess ? (
                  <button
                    type="button"
                    disabled={refining}
                    onClick={handleRefine}
                    className="btn-primary text-[13px] px-5 py-2 shadow-sm"
                  >
                    {refining ? 'Refining dataset...' : 'Handle Automatically'}
                  </button>
                ) : (
                  <>
                    <a
                      href={cleanedDatasetDownloadUrl(datasetId)}
                      download
                      className="btn-secondary text-[12px] px-3.5 py-2 inline-flex items-center gap-1.5"
                    >
                      <svg className="w-3.5 h-3.5 text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                      </svg>
                      Download Cleaned Dataset
                    </a>
                    <button
                      type="button"
                      onClick={() => {
                        setRefinementOpen(false)
                        navigate('/train')
                      }}
                      className="btn-primary text-[13px] px-5 py-2 shadow-sm"
                    >
                      Continue to Train <ArrowRight size={14} aria-hidden />
                    </button>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
