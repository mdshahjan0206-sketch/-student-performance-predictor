import { useCallback, useEffect, useRef, useState } from 'react'
import { ArrowRight, Check, CircleCheck, TriangleAlert, X } from 'lucide-react'
import { useNavigate } from 'react-router-dom'

import { ConfusionMatrix, FeatureImportanceChart, ResidualPlot, RocCurve } from '../charts'
import {
  Alert,
  Badge,
  Card,
  Checkbox,
  EmptyState,
  MetricCard,
  MetricGrid,
  PageHeader,
  SectionHeading,
  SegmentedControl,
  Select,
  formatNumber,
} from '../components/ui'
import { useWorkspace } from '../hooks/useWorkspace'
import {
  ApiRequestError,
  cleanedDatasetDownloadUrl,
  getJob,
  getMissing,
  getModel,
  getTargetAnalysis,
  getTrainingConfig,
  refineDataset,
  runPreflight,
  startTraining,
} from '../services/api'
import type { JobStatus, ModelInfo, PreflightResult, ProblemType, TargetAnalysis, TrainingConfig } from '../types/api'

const STATUS_COPY: Record<JobStatus, string> = {
  queued: 'Waiting for a worker',
  processing: 'Checking the dataset',
  training: 'Fitting the model',
  evaluating: 'Scoring on the held-out split',
  completed: 'Finished',
  failed: 'Stopped',
}

const STATUS_ORDER: JobStatus[] = ['queued', 'processing', 'training', 'evaluating', 'completed']

function ModelResults({ model }: { model: ModelInfo }) {
  const classification = model.problem_type === 'classification'
  const [fiSlice, setFiSlice] = useState<'top5' | 'top10' | 'all'>('top10')

  const displayedImportance = model.charts.feature_importance
    ? {
        features: model.charts.feature_importance.features.slice(
          0,
          fiSlice === 'top5' ? 5 : fiSlice === 'top10' ? 10 : undefined,
        ),
        importance: model.charts.feature_importance.importance.slice(
          0,
          fiSlice === 'top5' ? 5 : fiSlice === 'top10' ? 10 : undefined,
        ),
      }
    : null

  return (
    <div className="space-y-7">
      <div>
        <SectionHeading title="Training summary" />
        <MetricGrid columns={5}>
          <MetricCard label="Model" value={model.model_choice} hint={model.estimator_name} tone="brand" />
          <MetricCard label="Predicting" value={model.target_column} hint={model.problem_type} />
          <MetricCard label="Training rows" value={model.n_train.toLocaleString()} />
          <MetricCard label="Testing rows" value={model.n_test.toLocaleString()} />
          <MetricCard
            label="Training time"
            value={`${model.training_time.toFixed(2)}s`}
            hint={`${model.feature_count} features`}
          />
        </MetricGrid>
        {model.target_missing_excluded != null && model.target_missing_excluded > 0 && (
          <div className="mt-3">
            <Alert tone="warn">
              {model.target_missing_excluded.toLocaleString()} rows have missing target values and were excluded from supervised training.
            </Alert>
          </div>
        )}
      </div>

      <div>
        <SectionHeading
          title="Model performance"
          description={
            classification
              ? 'Measured on the 20% of rows held back from training. Precision, recall and F1 are weighted across classes.'
              : 'Measured on the 20% of rows held back from training.'
          }
        />
        <MetricGrid columns={4}>
          {classification ? (
            <>
              <MetricCard label="Accuracy" value={formatNumber(model.metrics.accuracy)} tone="brand" />
              <MetricCard label="Precision" value={formatNumber(model.metrics.precision)} />
              <MetricCard label="Recall" value={formatNumber(model.metrics.recall)} />
              <MetricCard label="F1 score" value={formatNumber(model.metrics.f1)} />
            </>
          ) : (
            <>
              <MetricCard label="Mean absolute error" value={formatNumber(model.metrics.mae)} />
              <MetricCard label="Mean squared error" value={formatNumber(model.metrics.mse)} />
              <MetricCard label="Root mean squared error" value={formatNumber(model.metrics.rmse)} />
              <MetricCard label="R²" value={formatNumber(model.metrics.r2)} tone="brand" />
            </>
          )}
        </MetricGrid>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        {model.charts.confusion_matrix && (
          <Card>
            <SectionHeading
              title="Confusion matrix"
              description="Rows are the true class, columns are what the model predicted."
            />
            <ConfusionMatrix chart={model.charts.confusion_matrix} />
          </Card>
        )}

        {model.charts.roc_curve && (
          <Card>
            <SectionHeading
              title="ROC curve"
              description={`Area under the curve ${model.charts.roc_curve.auc.toFixed(3)}, with “${model.charts.roc_curve.positive_class}” as the positive class.`}
            />
            <RocCurve chart={model.charts.roc_curve} />
          </Card>
        )}

        {model.charts.residuals && (
          <Card>
            <SectionHeading
              title="Residual plot"
              description="Prediction error against predicted value. A flat band around zero means the model is not systematically over- or under-shooting."
            />
            <ResidualPlot chart={model.charts.residuals} />
            {model.charts.residuals.sampled && (
              <p className="text-[12px] text-muted mt-2">
                Plotting a fixed sample of {model.charts.residuals.plotted_points.toLocaleString()} of{' '}
                {model.charts.residuals.total_points.toLocaleString()} test points.
              </p>
            )}
          </Card>
        )}

        {model.charts.feature_importance && displayedImportance && (
          <Card>
            <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
              <div>
                <SectionHeading
                  title="Feature importance"
                  description="Features ranked by actual relative importance to the model. Note: relative importance does not imply causation."
                />
              </div>
              <div className="inline-flex p-0.5 bg-canvas border border-line rounded-lg text-[12px]">
                {(['top5', 'top10', 'all'] as const).map((k) => (
                  <button
                    key={k}
                    type="button"
                    onClick={() => setFiSlice(k)}
                    className={`px-2.5 py-1 rounded-md font-medium transition-colors ${
                      fiSlice === k ? 'bg-panel text-ink shadow-sm font-semibold' : 'text-muted hover:text-ink'
                    }`}
                  >
                    {k === 'top5' ? 'Top 5' : k === 'top10' ? 'Top 10' : `All (${model.charts.feature_importance?.features.length})`}
                  </button>
                ))}
              </div>
            </div>
            <FeatureImportanceChart chart={displayedImportance} height={Math.min(500, Math.max(280, displayedImportance.features.length * 28))} />
          </Card>
        )}
      </div>

      {classification && !model.charts.roc_curve && (
        <Alert tone="info">
          A ROC curve is drawn for two-class targets scored with probabilities. This model does not
          meet both conditions, so it is not shown.
        </Alert>
      )}

      {!model.charts.feature_importance && (
        <Alert tone="info">
          {model.estimator_name} does not expose feature importances, so that chart is not shown.
          Tree-based models do.
        </Alert>
      )}
    </div>
  )
}

export function TrainPage() {
  const { dataset, setDataset, model, setModel } = useWorkspace()
  const navigate = useNavigate()
  const datasetId = dataset?.dataset_id ?? ''

  const [config, setConfig] = useState<TrainingConfig | null>(null)
  const [configError, setConfigError] = useState<string | null>(null)

  const [target, setTarget] = useState('')
  const [manualTarget, setManualTarget] = useState(false)
  const [problemOverride, setProblemOverride] = useState<'auto' | ProblemType>('auto')
  const [analysis, setAnalysis] = useState<TargetAnalysis | null>(null)
  const [modelChoice, setModelChoice] = useState('')

  const [preflight, setPreflight] = useState<PreflightResult | null>(null)
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null)
  const [jobError, setJobError] = useState<{ message: string; recommended: string[] } | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const [showWhy, setShowWhy] = useState(false)
  const [showImbalanceDetails, setShowImbalanceDetails] = useState(false)
  const [showTargetDetails, setShowTargetDetails] = useState(false)
  const [refinementModalOpen, setRefinementModalOpen] = useState(false)
  const [refining, setRefining] = useState(false)
  const [missingAnalysis, setMissingAnalysis] = useState<any>(null)
  const pollRef = useRef<number | null>(null)

  const handleOpenRefinement = async () => {
    setRefinementModalOpen(true)
    if (!missingAnalysis && datasetId) {
      try {
        const res = await getMissing(datasetId)
        setMissingAnalysis(res.analysis)
      } catch {
        // fallback
      }
    }
  }

  const handleRefine = async () => {
    if (!datasetId) return
    setRefining(true)
    try {
      const res = await refineDataset(datasetId)
      if (dataset) {
        setDataset({
          ...dataset,
          refined: true,
          missing_values_handled: res.total_missing,
          missing_values_remaining: 0,
        })
      }
      setRefinementModalOpen(false)
      if (target && analysis && modelChoice) {
        runPreflight(datasetId, target, analysis.problem_type, modelChoice)
          .then(setPreflight)
          .catch(() => undefined)
      }
    } catch {
      // ignore
    } finally {
      setRefining(false)
    }
  }

  // load the training configuration once per dataset
  useEffect(() => {
    if (!datasetId) return
    let cancelled = false
    getTrainingConfig(datasetId)
      .then((result) => {
        if (cancelled) return
        setConfig(result)
        setTarget(result.default_target)
        setManualTarget(!result.auto_detected)
        setAnalysis(result.analysis)
        setModelChoice(result.analysis.recommended_model)
        setConfigError(null)
      })
      .catch((err) =>
        setConfigError(err instanceof ApiRequestError ? err.message : 'The training options could not be loaded.'),
      )
    return () => {
      cancelled = true
    }
  }, [datasetId])

  // re-analyse whenever the target or the problem-type override changes
  useEffect(() => {
    if (!datasetId || !target) return
    let cancelled = false
    getTargetAnalysis(datasetId, target, problemOverride)
      .then((result) => {
        if (cancelled) return
        setAnalysis(result)
        setModelChoice((current) =>
          result.model_options.includes(current) ? current : result.recommended_model,
        )
      })
      .catch(() => undefined)
    return () => {
      cancelled = true
    }
  }, [datasetId, target, problemOverride])

  // pre-flight validation follows the current configuration
  useEffect(() => {
    if (!datasetId || !target || !analysis || !modelChoice) return
    let cancelled = false
    runPreflight(datasetId, target, analysis.problem_type, modelChoice)
      .then((result) => !cancelled && setPreflight(result))
      .catch(() => !cancelled && setPreflight(null))
    return () => {
      cancelled = true
    }
  }, [datasetId, target, analysis, modelChoice, dataset?.refined])

  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  useEffect(() => stopPolling, [stopPolling])

  async function handleTrain() {
    if (!datasetId || !analysis) return
    setJobError(null)
    setJobStatus('queued')
    setElapsed(0)

    try {
      const { job_id } = await startTraining(datasetId, target, analysis.problem_type, modelChoice)

      stopPolling()
      pollRef.current = window.setInterval(async () => {
        try {
          const job = await getJob(job_id)
          setJobStatus(job.status)
          setElapsed(job.elapsed)

          if (job.status === 'completed' && job.model_id) {
            stopPolling()
            const trained = await getModel(job.model_id)
            setModel(trained)
          } else if (job.status === 'failed') {
            stopPolling()
            setJobError({
              message: job.error?.message ?? 'Training stopped before it finished.',
              recommended: job.error?.recommended_models ?? [],
            })
          }
        } catch (err) {
          stopPolling()
          setJobStatus('failed')
          setJobError({
            message: err instanceof ApiRequestError ? err.message : 'Lost contact with the training job.',
            recommended: [],
          })
        }
      }, 400)
    } catch (err) {
      setJobStatus('failed')
      setJobError({
        message: err instanceof ApiRequestError ? err.message : 'Training could not be started.',
        recommended: [],
      })
    }
  }

  if (!dataset) {
    return (
      <div className="space-y-6">
        <PageHeader eyebrow="Step 3 of 5" title="Train model" />
        <EmptyState
          title="No dataset loaded"
          description="Upload a dataset before training a model."
          action={
            <button type="button" className="btn-primary" onClick={() => navigate('/upload')}>
              Go to upload
            </button>
          }
        />
      </div>
    )
  }

  const running = jobStatus !== null && !['completed', 'failed'].includes(jobStatus)
  const isRefinementNeeded = Boolean((dataset.total_missing ?? 0) > 0 && !dataset.refined)
  const blocked = isRefinementNeeded || (preflight ? !preflight.ok : false)
  const offRecommendation = Boolean(analysis && modelChoice && modelChoice !== analysis.recommended_model)

  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="Step 3 of 5"
        title="Train model"
        description="Choose what to predict. The problem type is detected from the target column, a model is recommended for the dataset size, and the data is checked before any fitting starts."
      />

      {configError && <Alert tone="danger">{configError}</Alert>}

      <Card>
        <SectionHeading title="Dataset" />
        <MetricGrid columns={5}>
          <MetricCard label="Rows" value={dataset.num_rows.toLocaleString()} />
          <MetricCard label="Columns" value={dataset.num_cols} />
          <MetricCard label="Numeric" value={dataset.overview?.numeric_cols.length ?? '—'} />
          <MetricCard label="Categorical" value={dataset.overview?.categorical_cols.length ?? '—'} />
          <MetricCard
            label="Missing values"
            value={(dataset.total_missing ?? 0).toLocaleString()}
            hint={dataset.refined ? 'Refined' : (dataset.total_missing ?? 0) > 0 ? 'Refinement required' : 'None'}
            tone={dataset.refined ? 'brand' : (dataset.total_missing ?? 0) > 0 ? 'warn' : 'default'}
          />
        </MetricGrid>
        {(dataset.total_missing ?? 0) > 0 && (
          <div
            className={`mt-3 p-3 rounded-lg border ${
              dataset.refined ? 'border-brand/30 bg-brand-soft/20' : 'border-amber/30 bg-amber-soft/20'
            } flex items-center justify-between gap-3 flex-wrap`}
          >
            <div className="flex items-center gap-2">
              <span className={dataset.refined ? 'text-brand font-bold text-base' : 'text-amber font-bold text-base'}>
                {dataset.refined ? <CircleCheck size={18} aria-hidden /> : <TriangleAlert size={18} aria-hidden />}
              </span>
              <div>
                <p className="text-[13px] font-semibold text-ink">
                  {dataset.refined ? 'Missing values handled' : 'Missing values detected'}
                </p>
                <p className="text-[12px] text-muted">
                  {dataset.refined
                    ? `${dataset.total_missing.toLocaleString()} cells refined · Ready for training`
                    : `${dataset.total_missing.toLocaleString()} cells · Refinement required`}
                </p>
              </div>
            </div>
            {dataset.refined ? (
              <div className="flex flex-wrap items-center gap-2">
                <a
                  href={cleanedDatasetDownloadUrl(datasetId)}
                  download
                  className="btn-secondary text-[12px] px-3.5 py-1.5 inline-flex items-center gap-1.5 shadow-sm"
                >
                  <svg className="w-3.5 h-3.5 text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Download Cleaned Dataset
                </a>
                <Badge tone="brand">Ready for training</Badge>
              </div>
            ) : (
              <button
                type="button"
                onClick={handleOpenRefinement}
                className="btn-primary text-[12px] px-3.5 py-1.5 shadow-sm"
              >
                Handle Missing Values
              </button>
            )}
          </div>
        )}
      </Card>

      <Card>
        {/* SECTION 26: TRAIN PAGE PRIORITY BAR */}
        <div className="p-3.5 bg-canvas/70 rounded-xl border border-line grid grid-cols-2 sm:grid-cols-5 gap-3 text-[12px] mb-4">
          <div>
            <span className="text-[10px] font-bold uppercase text-muted tracking-wider block">Target</span>
            <span className="font-mono text-[13px] font-bold text-ink truncate block">{target || '—'}</span>
          </div>
          <div>
            <span className="text-[10px] font-bold uppercase text-muted tracking-wider block">Problem</span>
            <span className="text-[13px] font-semibold text-ink capitalize block">{analysis?.problem_type || '—'}</span>
          </div>
          <div>
            <span className="text-[10px] font-bold uppercase text-muted tracking-wider block">Model</span>
            <span className="text-[13px] font-semibold text-ink truncate block">
              {modelChoice || '—'}{' '}
              {analysis && modelChoice === analysis.recommended_model && (
                <span className="text-[11px] text-brand font-normal">· Rec</span>
              )}
            </span>
          </div>
          <div>
            <span className="text-[10px] font-bold uppercase text-muted tracking-wider block">Dataset</span>
            <span className="text-[13px] font-mono font-semibold text-ink block">
              {dataset.num_rows >= 1000 ? `${Math.round(dataset.num_rows / 1000)}K` : dataset.num_rows} × {dataset.num_cols}
            </span>
          </div>
          <div>
            <span className="text-[10px] font-bold uppercase text-muted tracking-wider block">Missing Values</span>
            <span className="text-[13px] font-semibold block">
              {dataset.refined ? (
                <span className="inline-flex items-center gap-1 text-ok"><CircleCheck size={13} aria-hidden />Handled</span>
              ) : (dataset.total_missing ?? 0) > 0 ? (
                <span className="inline-flex items-center gap-1 text-amber"><TriangleAlert size={13} aria-hidden />Required</span>
              ) : (
                <span className="inline-flex items-center gap-1 text-ok"><CircleCheck size={13} aria-hidden />None</span>
              )}
            </span>
          </div>
        </div>

        <SectionHeading
          title="Configure training"
          description="Target columns are detected by name, so this stays a student-performance tool rather than a general model builder."
        />

        {config && !config.auto_detected && (
          <div className="mb-5">
            <Alert tone="warn" title="No standard target column detected">
              Nothing matched the known performance column names. Choose the column you want to
              predict below.
            </Alert>
          </div>
        )}

        <div className="grid gap-5 lg:grid-cols-2">
          <div className="space-y-4">
            {config?.auto_detected && !manualTarget ? (
              <Select
                id="target-detected"
                label="Target column"
                value={target}
                onChange={setTarget}
                options={config.detected_targets.map((name) => ({ value: name, label: name }))}
                hint={`Detected: ${config.detected_targets.join(', ')}`}
              />
            ) : (
              <Select
                id="target-manual"
                label="Target column"
                value={target}
                onChange={setTarget}
                options={(config?.columns ?? dataset.column_names).map((name) => ({
                  value: name,
                  label: name,
                }))}
                hint="Every column in the dataset is available."
              />
            )}

            {config?.auto_detected && (
              <Checkbox
                id="manual-target"
                label="Choose any column manually"
                checked={manualTarget}
                onChange={(checked) => {
                  setManualTarget(checked)
                  if (!checked) setTarget(config.detected_targets[0] ?? target)
                }}
              />
            )}

            <SegmentedControl
              name="problem-type"
              label="Problem type"
              value={problemOverride}
              onChange={(value) => setProblemOverride(value as 'auto' | ProblemType)}
              options={[
                { value: 'auto', label: 'Detect' },
                { value: 'classification', label: 'Classification' },
                { value: 'regression', label: 'Regression' },
              ]}
            />

            {analysis && (
              <div className="space-y-1.5 pt-0.5">
                <div className="flex items-center justify-between text-[13px] flex-wrap gap-2">
                  <div className="flex items-center gap-2">
                    <CircleCheck size={16} className="text-brand-mid shrink-0" aria-hidden />
                    <span className="font-semibold text-ink capitalize">
                      {analysis.problem_type === 'regression' ? 'Regression' : 'Classification'}
                    </span>
                    <span className="text-muted">·</span>
                    <span className="text-muted">
                      {analysis.problem_type === 'regression' ? 'Numeric target: ' : 'Target: '}
                      <code className="font-mono text-ink text-[12px]">{analysis.target_column}</code>
                    </span>
                    {analysis.overridden && (
                      <span className="text-amber font-semibold text-[12px]">
                        (overridden)
                      </span>
                    )}
                  </div>
                  <button
                    type="button"
                    onClick={() => setShowTargetDetails(!showTargetDetails)}
                    className="text-[12px] font-semibold text-brand hover:underline"
                  >
                    {showTargetDetails ? 'Hide details' : '[ View details ]'}
                  </button>
                </div>
                {showTargetDetails && (
                  <p className="text-[12px] text-muted pl-5 animate-fade-in">
                    Detected <strong className="text-ink capitalize">{analysis.auto_problem_type}</strong> from{' '}
                    {analysis.distinct_values.toLocaleString()} distinct values in{' '}
                    <code className="font-mono text-ink">{analysis.target_column}</code> ({analysis.target_dtype}).
                  </p>
                )}
              </div>
            )}
          </div>

          <div className="space-y-4">
            <Select
              id="model-choice"
              label="Model"
              value={modelChoice}
              onChange={setModelChoice}
              options={(analysis?.model_options ?? []).map((name) => ({ value: name, label: name }))}
            />

            {analysis && (
              <div className="card p-4 space-y-2 bg-canvas/60">
                <div className="flex items-center justify-between gap-2 flex-wrap">
                  <span className="text-[12px] font-semibold text-muted uppercase tracking-wider">
                    Model recommendation
                  </span>
                  <Badge tone="brand">{analysis.recommended_model} · Recommended</Badge>
                </div>

                <div className="flex items-center justify-between gap-2 flex-wrap pt-1">
                  <div className="text-[13px] text-ink font-medium">
                    <span className="text-muted font-normal">Why? </span>
                    {dataset.num_rows >= 1000 ? `${Math.round(dataset.num_rows / 1000)}K` : dataset.num_rows} rows · {analysis.problem_type === 'regression' ? 'Numeric target' : 'Mixed features'} · <span className="capitalize">{analysis.problem_type}</span>
                  </div>
                  <button
                    type="button"
                    onClick={() => setShowWhy(!showWhy)}
                    className="text-[12px] font-semibold text-brand hover:underline"
                  >
                    {showWhy ? 'Hide details' : '[ View details ]'}
                  </button>
                </div>

                {showWhy && (
                  <div className="pt-2 border-t border-line text-[12px] text-muted space-y-2 animate-fade-in">
                    <p className="leading-relaxed">
                      {analysis.recommendation_reason ||
                        `Selected by dataset heuristics for ${analysis.problem_type} with ${dataset.num_rows.toLocaleString()} rows.`}
                    </p>
                    {analysis.workflow_notes && analysis.workflow_notes.length > 0 && (
                      <ul className="text-[12px] text-muted space-y-1 list-disc pl-4">
                        {analysis.workflow_notes.map((note, idx) => (
                          <li key={idx}>{note}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                )}

                {modelChoice !== analysis.recommended_model && (
                  <button
                    type="button"
                    onClick={() => setModelChoice(analysis.recommended_model)}
                    className="btn-secondary text-[12px] h-7 px-2.5 mt-1"
                  >
                    Use recommended model ({analysis.recommended_model})
                  </button>
                )}
              </div>
            )}

            {offRecommendation && analysis && (
              <div className="p-3 rounded-lg border border-line bg-canvas/40 text-[12px] text-ink flex items-center justify-between gap-2 flex-wrap">
                <span>Selected: <strong>{modelChoice}</strong> · Recommended: <strong>{analysis.recommended_model}</strong></span>
                <button
                  type="button"
                  onClick={() => setModelChoice(analysis.recommended_model)}
                  className="text-brand font-semibold hover:underline"
                >
                  Use recommended
                </button>
              </div>
            )}

            {analysis && analysis.missing_values > 0 && (
              <div className="p-3 rounded-lg border border-amber/30 bg-amber-soft/20 text-[12px] text-amber-deep">
                <span className="font-semibold"><TriangleAlert size={12} className="inline -mt-0.5 mr-1" aria-hidden />Missing target: </span>
                {analysis.missing_values.toLocaleString()} rows affected. (Rows with missing target values are excluded.)
              </div>
            )}
          </div>
        </div>

        {/* Dedicated Missing Values Status & Refinement Gate */}
        {(dataset.total_missing ?? 0) > 0 && (
          <div
            className={`mt-6 p-4 rounded-xl border ${
              dataset.refined ? 'border-brand/30 bg-brand-soft/20' : 'border-amber/30 bg-amber-soft/20'
            } flex items-center justify-between gap-4 flex-wrap`}
          >
            <div className="flex items-center gap-3">
              <span className={dataset.refined ? 'text-brand font-bold text-lg' : 'text-amber font-bold text-lg'}>
                {dataset.refined ? <CircleCheck size={18} aria-hidden /> : <TriangleAlert size={18} aria-hidden />}
              </span>
              <div>
                <p className="text-[14px] font-semibold text-ink">
                  {dataset.refined ? 'Missing values handled' : 'Missing values'}
                </p>
                <p className="text-[12px] text-muted">
                  {dataset.refined
                    ? `${dataset.total_missing.toLocaleString()} cells refined · Ready for training`
                    : `${dataset.total_missing.toLocaleString()} cells · Refinement required`}
                </p>
              </div>
            </div>
            {!dataset.refined ? (
              <button
                type="button"
                onClick={handleOpenRefinement}
                className="btn-primary text-[13px] px-4 py-2 shadow-sm"
              >
                Handle Missing Values
              </button>
            ) : (
              <Badge tone="brand">Ready for training</Badge>
            )}
          </div>
        )}

        {preflight && !preflight.ok && (
          <div className="mt-5">
            <Alert tone="danger" title="This combination cannot be trained">
              <p>{preflight.reason}</p>
              {preflight.recommended_models.length > 0 && (
                <>
                  <p className="mt-2 font-semibold">Try one of these instead:</p>
                  <ul className="mt-1 space-y-0.5">
                    {preflight.recommended_models.map((name) => (
                      <li key={name}>• {name}</li>
                    ))}
                  </ul>
                </>
              )}
            </Alert>
          </div>
        )}

        {preflight?.ok && preflight.warnings.length > 0 && (
          <div className="mt-5 space-y-2">
            {preflight.warnings.map((warning) => {
              const lower = warning.toLowerCase()
              // Issue 3: Class imbalance must only appear for classification tasks, never regression
              if (lower.includes('imbalance')) {
                if (analysis?.problem_type === 'regression') return null
                return (
                  <div key={warning} className="p-3 rounded-lg border border-amber/30 bg-amber-soft/20 flex items-center justify-between gap-3 flex-wrap">
                    <div className="flex items-center gap-2">
                      <TriangleAlert size={14} className="text-amber shrink-0" aria-hidden />
                      <span className="text-[13px] font-semibold text-ink">Class imbalance: Uneven classes</span>
                    </div>
                    <button
                      type="button"
                      onClick={() => setShowImbalanceDetails(!showImbalanceDetails)}
                      className="text-[12px] font-semibold text-brand hover:underline"
                    >
                      {showImbalanceDetails ? 'Hide details' : '[ View metrics ]'}
                    </button>
                    {showImbalanceDetails && (
                      <p className="w-full text-[12px] text-muted pt-2 border-t border-line/40">{warning}</p>
                    )}
                  </div>
                )
              }
              // Issue 1: Do not show contradictory or duplicate missing feature / imputation warning
              if (lower.includes('missing feature') || lower.includes('automatically imputed') || lower.includes('unresolved missing data')) {
                return null
              }
              return (
                <Alert key={warning} tone="warn">
                  {warning}
                </Alert>
              )
            })}
          </div>
        )}

        {/* Train Model Action with Mandatory Training Gate */}
        <div className="mt-6 flex flex-wrap items-center gap-3">
          <button
            type="button"
            className="btn-primary px-6"
            onClick={handleTrain}
            disabled={running || blocked || isRefinementNeeded || !modelChoice || !target}
          >
            {running ? 'Training in progress' : 'Train model'}
          </button>
          {isRefinementNeeded ? (
            <div className="flex items-center gap-2 text-amber text-[13px] font-medium">
              <span className="inline-flex items-center gap-1.5"><TriangleAlert size={14} aria-hidden />Missing values: {dataset.total_missing.toLocaleString()} cells</span>
              <button
                type="button"
                onClick={handleOpenRefinement}
                className="btn-secondary text-[12px] px-3 py-1 text-ink"
              >
                Handle Missing Values
              </button>
            </div>
          ) : blocked ? (
            <span className="text-[13px] text-muted">
              {preflight?.reason ? preflight.reason : 'Fix the issue above to continue.'}
            </span>
          ) : (dataset.total_missing ?? 0) > 0 ? (
            <span className="text-[13px] text-brand font-medium flex items-center gap-1.5">
              <span className="inline-flex items-center gap-1.5"><CircleCheck size={14} aria-hidden />Data refined</span>
              <span className="text-muted">· Ready for training</span>
            </span>
          ) : null}
        </div>
      </Card>

      {jobStatus && (
        <Card>
          <SectionHeading
            title="Training job"
            description="Status comes from the backend as each stage actually completes."
          />
          <div className="flex flex-wrap items-center gap-2">
            {STATUS_ORDER.map((status) => {
              const currentIndex = STATUS_ORDER.indexOf(jobStatus as JobStatus)
              const index = STATUS_ORDER.indexOf(status)
              const done = jobStatus === 'completed' || (currentIndex > index && currentIndex !== -1)
              const active = jobStatus === status
              const failedHere = jobStatus === 'failed'

              return (
                <Badge key={status} tone={failedHere && index === 0 ? 'danger' : active ? 'brand' : done ? 'brand' : 'neutral'}>
                  {done && !active ? <Check size={12} aria-hidden /> : null}
                  {STATUS_COPY[status]}
                </Badge>
              )
            })}
            {jobStatus === 'failed' && <Badge tone="danger">Stopped</Badge>}
            <span className="text-[12px] text-muted tabular-nums ml-1">{elapsed.toFixed(1)}s</span>
          </div>

          {jobError && (
            <div className="mt-4">
              <Alert tone="danger" title="Training stopped">
                <p>{jobError.message}</p>
                {jobError.recommended.length > 0 && (
                  <>
                    <p className="mt-2 font-semibold">Recommended models for this dataset:</p>
                    <ul className="mt-1 space-y-0.5">
                      {jobError.recommended.map((name) => (
                        <li key={name}>• {name}</li>
                      ))}
                    </ul>
                  </>
                )}
              </Alert>
            </div>
          )}
        </Card>
      )}

      {model && model.dataset_id === datasetId && (
        <>
          {jobStatus === 'completed' && (
            <Alert tone="brand" title="Model trained">
              {model.model_choice} finished in {model.training_time.toFixed(2)} seconds and is ready for
              predictions.
            </Alert>
          )}

          {model.warnings.length > 0 && (
            <div className="space-y-2">
              {model.warnings.map((warning) => (
                <Alert key={warning} tone="warn">
                  {warning}
                </Alert>
              ))}
            </div>
          )}

          <ModelResults model={model} />

          <div className="flex flex-wrap gap-3">
            <button type="button" className="btn-secondary" onClick={() => navigate('/analysis')}>
              Back to analysis
            </button>
            <button type="button" className="btn-primary" onClick={() => navigate('/predict')}>
              Make predictions
            </button>
          </div>
        </>
      )}

      {/* Refinement Modal (State B: Handling Missing Values) */}
      {refinementModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-ink/40 backdrop-blur-sm animate-fade-in">
          <div className="card max-w-2xl w-full max-h-[90vh] overflow-y-auto p-6 space-y-5 bg-panel border border-line shadow-2xl rounded-2xl">
            <div className="flex items-center justify-between pb-3 border-b border-line">
              <div>
                <h2 className="text-[18px] font-bold text-ink">Handling Missing Values</h2>
                <p className="text-[12px] text-muted mt-0.5">Automated type-aware refinement pipeline</p>
              </div>
              <button
                type="button"
                onClick={() => setRefinementModalOpen(false)}
                className="text-muted hover:text-ink text-[18px] p-1.5 rounded-lg hover:bg-canvas" aria-label="Close"
              >
                <X size={16} aria-hidden />
              </button>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 rounded-xl bg-canvas border border-line">
                <span className="text-[11px] font-bold uppercase text-muted tracking-wider">Missing cells</span>
                <p className="text-[20px] font-extrabold text-ink tabular-nums mt-0.5">
                  {(dataset.total_missing ?? 0).toLocaleString()}
                </p>
              </div>
              <div className="p-3 rounded-xl bg-canvas border border-line">
                <span className="text-[11px] font-bold uppercase text-muted tracking-wider">Columns affected</span>
                <p className="text-[20px] font-extrabold text-ink tabular-nums mt-0.5">
                  {missingAnalysis?.affected_columns_count ?? Object.values(dataset.missing_summary ?? {}).filter((v) => v > 0).length}
                </p>
              </div>
            </div>

            {/* Target missing rows notice if applicable */}
            {dataset?.detected_targets?.some((t) => (dataset.missing_summary?.[t] ?? 0) > 0) && (
              <div className="p-3 rounded-lg border border-amber/30 bg-amber-soft/20 text-[12px] text-amber-deep space-y-0.5">
                <p className="font-semibold">
                  <TriangleAlert size={12} className="inline -mt-0.5 mr-1" aria-hidden />Missing target: {dataset.detected_targets.reduce((acc, t) => acc + (dataset.missing_summary?.[t] ?? 0), 0).toLocaleString()} rows affected
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
                    {(missingAnalysis?.details ?? Object.entries(dataset.missing_summary ?? {})
                      .filter(([, count]) => count > 0)
                      .map(([col, count]) => {
                        const dtype = dataset.dtypes?.[col] ?? 'string'
                        const isNum = dtype.includes('int') || dtype.includes('float')
                        return {
                          column_name: col,
                          dtype: dtype,
                          missing_count: count,
                          missing_percentage: (count / (dataset.num_rows || 1)) * 100,
                          recommended_method: isNum ? 'median' : 'mode',
                        }
                      }))
                      .filter((d: any) => d.missing_count > 0)
                      .map((d: any) => {
                        const method =
                          d.recommended_method === 'median'
                            ? 'Median imputation'
                            : d.recommended_method === 'mode'
                            ? 'Mode (most frequent)'
                            : d.recommended_method === 'mean'
                            ? 'Mean imputation'
                            : 'Drop / Exclude'
                        return (
                          <tr key={d.column_name} className="border-b border-line/50 hover:bg-canvas/40">
                            <td className="px-3 py-2 font-mono text-[11px] text-ink">{d.column_name}</td>
                            <td className="px-3 py-2 text-muted">{d.dtype}</td>
                            <td className="px-3 py-2 text-right tabular-nums text-amber font-medium">
                              {d.missing_count.toLocaleString()} ({d.missing_percentage.toFixed(1)}%)
                            </td>
                            <td className="px-3 py-2 text-ink font-medium">{method}</td>
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
                <p className="font-extrabold text-ink tabular-nums">{(dataset.total_missing ?? 0).toLocaleString()} missing</p>
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

            {/* Action Footer */}
            <div className="flex items-center justify-between gap-3 pt-3 border-t border-line">
              <button
                type="button"
                onClick={() => setRefinementModalOpen(false)}
                className="btn-secondary text-[12px] px-4 py-2"
              >
                Close
              </button>

              <button
                type="button"
                disabled={refining}
                onClick={handleRefine}
                className="btn-primary text-[13px] px-5 py-2 shadow-sm"
              >
                {refining ? 'Refining dataset...' : 'Handle Automatically'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
