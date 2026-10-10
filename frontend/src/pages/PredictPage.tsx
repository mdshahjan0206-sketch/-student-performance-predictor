import { useEffect, useState } from 'react'
import { Info } from 'lucide-react'
import { useNavigate } from 'react-router-dom'

import { PredictionDistribution } from '../charts'
import { DataTable } from '../components/DataTable'
import { FeatureInput } from '../components/FeatureInput'
import { FileDropzone } from '../components/FileDropzone'
import {
  Alert,
  Badge,
  Card,
  ChartSkeleton,
  EmptyState,
  ProgressRing,
  MetricCard,
  MetricGrid,
  PageHeader,
  SectionHeading,
  Spinner,
  formatNumber,
} from '../components/ui'
import { useAsync } from '../hooks/useAsync'
import { useWorkspace } from '../hooks/useWorkspace'
import {
  ApiRequestError,
  getFeatureSchema,
  getPredictionPage,
  predictBatch,
  predictSingle,
  predictionDownloadUrl,
} from '../services/api'
import type { BatchSummary, PredictionPage as ResultsPage, SingleResult } from '../types/api'

export function PredictPage() {
  const { model, note } = useWorkspace()
  const navigate = useNavigate()
  const modelId = model?.model_id ?? ''

  const [tab, setTab] = useState<'single' | 'batch'>('single')

  // --- single ---------------------------------------------------------
  const schema = useAsync(() => getFeatureSchema(modelId), [modelId], { enabled: Boolean(modelId) })
  const [inputs, setInputs] = useState<Record<string, string | number>>({})
  const [singleResult, setSingleResult] = useState<SingleResult | null>(null)
  const [singleBusy, setSingleBusy] = useState(false)
  const [singleError, setSingleError] = useState<string | null>(null)

  useEffect(() => {
    if (!schema.data) return
    const defaults: Record<string, string | number> = {}
    for (const field of schema.data.fields) defaults[field.name] = field.default
    setInputs(defaults)
    setSingleResult(null)
  }, [schema.data])

  async function handleSingle() {
    if (!modelId) return
    setSingleBusy(true)
    setSingleError(null)
    try {
      const result = await predictSingle(modelId, inputs)
      setSingleResult(result)
      note(`Predicted ${result.target_column} for one student`)
    } catch (err) {
      setSingleError(err instanceof ApiRequestError ? err.message : 'The prediction could not be completed.')
    } finally {
      setSingleBusy(false)
    }
  }

  function resetInputs() {
    if (!schema.data) return
    const defaults: Record<string, string | number> = {}
    for (const field of schema.data.fields) defaults[field.name] = field.default
    setInputs(defaults)
    setSingleResult(null)
    setSingleError(null)
  }

  // --- batch ----------------------------------------------------------
  const [batch, setBatch] = useState<BatchSummary | null>(null)
  const [results, setResults] = useState<ResultsPage | null>(null)
  const [batchBusy, setBatchBusy] = useState(false)
  const [batchError, setBatchError] = useState<{ message: string; errors: string[] } | null>(null)
  const [pageSize, setPageSize] = useState(25)
  const [pageLoading, setPageLoading] = useState(false)

  async function handleBatchFile(file: File) {
    if (!modelId) return
    setBatchBusy(true)
    setBatchError(null)
    setBatch(null)
    setResults(null)
    try {
      const response = await predictBatch(modelId, file)
      setBatch(response.batch)
      setResults(response.results)
      setPageSize(response.results.page_size)
      note(`Scored ${response.batch.row_count.toLocaleString()} rows from ${response.batch.source_filename}`)
    } catch (err) {
      if (err instanceof ApiRequestError) {
        const details = (err.details?.errors as string[] | undefined) ?? []
        setBatchError({ message: err.message, errors: details })
      } else {
        setBatchError({ message: 'The batch prediction could not be completed.', errors: [] })
      }
    } finally {
      setBatchBusy(false)
    }
  }

  async function loadPage(page: number, size: number) {
    if (!results) return
    setPageLoading(true)
    try {
      setResults(await getPredictionPage(results.prediction_id, page, size))
    } catch (err) {
      setBatchError({
        message: err instanceof ApiRequestError ? err.message : 'That page of results could not be loaded.',
        errors: [],
      })
    } finally {
      setPageLoading(false)
    }
  }

  if (!model) {
    return (
      <div className="space-y-6">
        <PageHeader eyebrow="Step 4 of 5" title="Prediction" />
        <EmptyState
          title="No trained model"
          description="Train a model before making predictions. Uploading a new dataset clears the previous model, because it was fitted on different data."
          action={
            <button type="button" className="btn-primary" onClick={() => navigate('/train')}>
              Go to training
            </button>
          }
        />
      </div>
    )
  }

  const predictionColumns = ['Prediction', 'Confidence', 'Predicted Class']

  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="Step 4 of 5"
        title="Prediction"
        description="Score one student from a form built out of the model’s own features, or upload a file and score every row."
      />

      <div>
        <SectionHeading title="Model in use" />
        <MetricGrid columns={5}>
          <MetricCard label="Model" value={model.model_choice} hint={model.estimator_name} tone="brand" />
          <MetricCard label="Problem type" value={model.problem_type} />
          <MetricCard label="Predicting" value={model.target_column} />
          <MetricCard label="Features" value={model.feature_count} />
          <MetricCard label="Trained on" value={model.dataset_filename} />
        </MetricGrid>
      </div>

      <div role="tablist" aria-label="Prediction mode" className="inline-flex p-0.5 bg-canvas border border-line rounded-lg">
        {(
          [
            ['single', 'Single student'],
            ['batch', 'Batch file'],
          ] as const
        ).map(([key, label]) => (
          <button
            key={key}
            type="button"
            role="tab"
            aria-selected={tab === key}
            onClick={() => setTab(key)}
            className={`px-4 h-9 rounded-[7px] text-[13px] font-semibold transition-colors
              ${tab === key ? 'bg-surface text-ink shadow-card' : 'text-muted hover:text-ink'}`}
          >
            {label}
          </button>
        ))}
      </div>

      {tab === 'single' && (
        <Card>
          <SectionHeading
            title="Enter student data"
            description={`One field per feature the model was trained on. Ranges come from the dataset, so the defaults are a realistic starting point.`}
          />

          {schema.loading && <ChartSkeleton height={200} />}
          {schema.error && <Alert tone="danger">{schema.error}</Alert>}

          {schema.data && (
            <>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {schema.data.fields.map((field) => (
                  <FeatureInput
                    key={field.name}
                    field={field}
                    value={inputs[field.name] ?? field.default}
                    onChange={(value) => setInputs((current) => ({ ...current, [field.name]: value }))}
                  />
                ))}
              </div>

              <div className="flex flex-wrap items-center gap-3 mt-6">
                <button type="button" className="btn-primary px-6" onClick={handleSingle} disabled={singleBusy}>
                  {singleBusy ? 'Predicting' : 'Predict'}
                </button>
                <button type="button" className="btn-secondary" onClick={resetInputs} disabled={singleBusy}>
                  Reset to defaults
                </button>
                {singleBusy && <Spinner />}
              </div>
            </>
          )}

          {singleError && (
            <div className="mt-5">
              <Alert tone="danger" title="Prediction stopped">
                {singleError}
              </Alert>
            </div>
          )}

          {singleResult && (
            <div className="mt-6 card p-6 animate-rise border-brand/30 shadow-glow-brand">
              <div className="flex flex-wrap items-center justify-between gap-6">
                <div className="min-w-0">
                  <p className="text-[12.5px] font-medium text-muted">Prediction · {singleResult.target_column}</p>
                  <p className="font-display text-[44px] leading-none font-semibold text-brand-mid mt-2 tabular-nums break-words">
                    {singleResult.problem_type === 'classification'
                      ? String(singleResult.predicted_class ?? singleResult.prediction)
                      : formatNumber(Number(singleResult.prediction))}
                  </p>
                  <p className="text-[12px] text-muted mt-2">Prediction ready</p>
                </div>
                {singleResult.confidence !== null && singleResult.confidence !== undefined && (
                  <ProgressRing
                    value={singleResult.confidence}
                    label={`${(singleResult.confidence * 100).toFixed(0)}%`}
                    caption="Confidence"
                  />
                )}
              </div>
              {singleResult.notice && (
                <div className="mt-5 p-3 rounded-lg bg-canvas/60 border border-line text-[12px] text-ink flex items-center gap-2">
                  <Info size={14} className="text-brand-mid shrink-0" aria-hidden />
                  <span>{singleResult.notice}</span>
                </div>
              )}
            </div>
          )}
        </Card>
      )}

      {tab === 'batch' && (
        <div className="space-y-6">
          <Card>
            <SectionHeading
              title="Upload a file to score"
              description="The file must contain every feature column the model was trained on. Extra columns are kept in the output and ignored by the model."
            />

            <div className="flex flex-wrap gap-1.5 mb-5">
              {model.feature_columns.map((name) => (
                <code
                  key={name}
                  className="px-2.5 h-7 inline-flex items-center rounded-md bg-canvas border border-line text-[12px] font-mono text-ink"
                >
                  {name}
                </code>
              ))}
            </div>

            <FileDropzone
              onFile={handleBatchFile}
              busy={batchBusy}
              label="Drop a file to score"
              hint="CSV or Excel, containing the feature columns above."
            />
          </Card>

          {batchError && (
            <Alert tone="danger" title="Batch prediction stopped">
              <p>{batchError.message}</p>
              {batchError.errors.length > 1 && (
                <ul className="mt-2 space-y-0.5">
                  {batchError.errors.slice(1).map((item) => (
                    <li key={item}>• {item}</li>
                  ))}
                </ul>
              )}
            </Alert>
          )}

          {batch && batch.warnings.length > 0 && (
            <div className="space-y-2">
              {batch.warnings.map((warning) => (
                <Alert key={warning} tone="warn">
                  {warning}
                </Alert>
              ))}
            </div>
          )}

          {batch && results && (
            <>
              <Alert tone="brand" title={`Scored ${batch.row_count.toLocaleString()} rows`}>
                From {batch.source_filename}. A Prediction column has been added
                {batch.has_confidence ? ', along with Confidence and Predicted Class' : ''}.
              </Alert>

              <Card>
                <SectionHeading
                  title="Results"
                  description="Rows are paged from the server, so a large run does not slow the page down."
                  action={
                    <a className="btn-primary" href={predictionDownloadUrl(batch.prediction_id)} download>
                      Download CSV
                    </a>
                  }
                />
                <DataTable
                  columns={results.columns}
                  rows={results.rows}
                  page={results.page}
                  totalPages={results.total_pages}
                  totalRows={results.total_rows}
                  pageSize={pageSize}
                  loading={pageLoading}
                  highlightColumns={predictionColumns}
                  onPageChange={(page) => void loadPage(page, pageSize)}
                  onPageSizeChange={(size) => {
                    setPageSize(size)
                    void loadPage(1, size)
                  }}
                />
              </Card>

              <Card>
                <SectionHeading
                  title="Prediction distribution"
                  description={
                    batch.problem_type === 'classification'
                      ? 'How many rows fell into each predicted class.'
                      : 'How the predicted values are spread.'
                  }
                  action={<Badge tone="neutral">{batch.row_count.toLocaleString()} rows</Badge>}
                />
                <PredictionDistribution distribution={batch.distribution} />
              </Card>
            </>
          )}
        </div>
      )}

      <div className="flex flex-wrap gap-3">
        <button type="button" className="btn-secondary" onClick={() => navigate('/train')}>
          Back to training
        </button>
      </div>
    </div>
  )
}
