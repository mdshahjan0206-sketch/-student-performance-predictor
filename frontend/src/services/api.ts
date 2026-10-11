// One place that talks to the backend.
//
// Every response is unwrapped here, and any non-2xx reply is converted into an
// ApiRequestError carrying the backend's own code and message — so pages show
// the message the server wrote, never a raw status line or stack trace.

import type {
  BatchSummary,
  BoxChart,
  ChartResponse,
  CorrelationChart,
  DashboardSummary,
  DatasetIntelligence,
  DatasetSummary,
  FeatureSchema,
  FilterOptions,
  FilterPreview,
  FilterState,
  HistogramChart,
  JobState,
  MissingResponse,
  ModelInfo,
  ModelRecommendation,
  OverviewStats,
  PerformanceEntry,
  PredictionHistoryEntry,
  PredictionPage,
  PreflightResult,
  PreviewPage,
  ProblemType,
  RefinementResult,
  ScatterChart,
  SessionStatus,
  SingleResult,
  StatisticsRow,
  StudentCapability,
  StudentExplorerFilters,
  StudentExplorerPage,
  ConfidenceSummary,
  StudentProfile,
  TargetAnalysis,
  TrainingConfig,
  ExplainCapability,
  ExplainInstanceResult,
  GlobalImportanceResult,
} from '../types/api'

export class ApiRequestError extends Error {
  code: string
  details?: Record<string, unknown>

  constructor(code: string, message: string, details?: Record<string, unknown>) {
    super(message)
    this.name = 'ApiRequestError'
    this.code = code
    this.details = details
  }
}

const SESSION_KEY = 'edupredict.session'

/** A per-browser id that keeps one tab's active dataset and model apart from
 *  another's. It is not a login: it carries no identity and the server only
 *  uses it as a dictionary key. */
export function sessionId(): string {
  let id = localStorage.getItem(SESSION_KEY)
  if (!id) {
    id = crypto.randomUUID()
    localStorage.setItem(SESSION_KEY, id)
  }
  return id
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, {
      ...init,
      headers: {
        'X-Session-Id': sessionId(),
        ...(init.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
        ...(init.headers ?? {}),
      },
    })
  } catch {
    throw new ApiRequestError(
      'NETWORK_ERROR',
      'Cannot reach the EduPredict server. Check that the backend is running, then try again.',
    )
  }

  const text = await response.text()
  let body: unknown = null
  if (text) {
    try {
      body = JSON.parse(text)
    } catch {
      body = null
    }
  }

  if (!response.ok) {
    const error = (body as { error?: { code: string; message: string; details?: Record<string, unknown> } })?.error
    throw new ApiRequestError(
      error?.code ?? 'REQUEST_FAILED',
      error?.message ?? 'The request could not be completed. Try again.',
      error?.details,
    )
  }

  return body as T
}

// --- small GET cache ---------------------------------------------------
// Chart and summary reads are pure and keyed by dataset, so repeating one
// while moving between tabs should not hit the network again. Cleared
// whenever a new dataset is uploaded.

const cache = new Map<string, Promise<unknown>>()

function cachedGet<T>(path: string): Promise<T> {
  const hit = cache.get(path)
  if (hit) return hit as Promise<T>
  const pending = request<T>(path).catch((error) => {
    cache.delete(path)
    throw error
  })
  cache.set(path, pending)
  return pending as Promise<T>
}

export function clearCache(prefix?: string) {
  if (!prefix) {
    cache.clear()
    return
  }
  for (const key of Array.from(cache.keys())) {
    if (key.includes(prefix)) cache.delete(key)
  }
}

// --- dataset -----------------------------------------------------------

const UPLOAD_TIMEOUT_MS = 180_000

interface UploadReply {
  dataset?: DatasetSummary
  session?: SessionStatus
  error?: { code?: string; message?: string; details?: Record<string, unknown> }
}

/** Build the error for a failed upload. The backend's own validation message
 *  always wins. Only when the reply is not the backend's JSON envelope (an
 *  HTML error page from a proxy, a gateway timeout, a wrong route) is a
 *  status-based explanation used, so the person is never told "try again"
 *  about a problem that retrying cannot fix. */
function uploadFailure(status: number, body: UploadReply, raw: string): ApiRequestError {
  if (body.error?.message) {
    return new ApiRequestError(body.error.code ?? 'UPLOAD_FAILED', body.error.message, body.error.details)
  }
  if (status >= 200 && status < 300) {
    return new ApiRequestError(
      'UNEXPECTED_RESPONSE',
      'The server accepted the file but sent back an unexpected reply, so the dataset could not be opened. Try again, and report it if it keeps happening.',
    )
  }
  if (status === 0 || status === 502 || status === 503 || status === 504) {
    return new ApiRequestError(
      'SERVER_UNAVAILABLE',
      `The server is not responding right now (HTTP ${status}). If it was idle it may still be starting up, so wait a minute and try again. Your file was not rejected.`,
    )
  }
  if (status === 404) {
    return new ApiRequestError(
      'UPLOAD_ROUTE_NOT_FOUND',
      'The upload address was not found on the server (HTTP 404). This is a deployment problem, not a problem with your file.',
    )
  }
  if (status === 413) {
    return new ApiRequestError('FILE_TOO_LARGE', 'The file is larger than the server accepts (HTTP 413).')
  }
  const looksLikeHtml = /^\s*<(!doctype|html)/i.test(raw)
  return new ApiRequestError(
    'UPLOAD_FAILED',
    `The server returned an unexpected ${looksLikeHtml ? 'web page' : 'reply'} (HTTP ${status}) instead of a validation result. This is a server problem, not a problem with your file.`,
  )
}

export async function uploadDataset(
  file: File,
  onProgress?: (percent: number) => void,
): Promise<{ dataset: DatasetSummary; session: SessionStatus }> {
  // XMLHttpRequest rather than fetch, because it reports real upload progress.
  return new Promise((resolve, reject) => {
    const form = new FormData()
    form.append('file', file)

    const xhr = new XMLHttpRequest()
    xhr.open('POST', '/api/dataset/upload')
    xhr.setRequestHeader('X-Session-Id', sessionId())

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) {
        onProgress(Math.round((event.loaded / event.total) * 100))
      }
    }

    // A cold or overloaded server can take a while to answer, and the browser
    // would otherwise wait indefinitely behind a spinner.
    xhr.timeout = UPLOAD_TIMEOUT_MS

    xhr.onload = () => {
      let body: UploadReply = {}
      try {
        const parsed: unknown = JSON.parse(xhr.responseText)
        if (parsed && typeof parsed === 'object') body = parsed as UploadReply
      } catch {
        /* not JSON: described by uploadFailure() below */
      }
      if (xhr.status >= 200 && xhr.status < 300 && body.dataset) {
        clearCache()
        resolve({ dataset: body.dataset, session: body.session as SessionStatus })
      } else {
        reject(uploadFailure(xhr.status, body, xhr.responseText))
      }
    }

    xhr.onerror = () =>
      reject(
        new ApiRequestError(
          'NETWORK_ERROR',
          'Cannot reach the EduPredict server. Check your connection, then try again. If the server was idle it may still be starting up.',
        ),
      )

    xhr.ontimeout = () =>
      reject(
        new ApiRequestError(
          'UPLOAD_TIMEOUT',
          'The server took too long to respond. If it was idle it may still be waking up, so wait a minute and try again.',
        ),
      )

    xhr.onabort = () => reject(new ApiRequestError('UPLOAD_ABORTED', 'The upload was cancelled before it finished.'))

    xhr.send(form)
  })
}

export const getDatasetSummary = (id: string) =>
  cachedGet<{ dataset: DatasetSummary }>(`/api/dataset/${id}/summary`).then((r) => r.dataset)

export const refineDataset = (id: string) =>
  request<{ success: boolean; refinement: RefinementResult }>(`/api/dataset/${id}/refine`, {
    method: 'POST',
  }).then((r) => {
    clearCache()
    return r.refinement
  })

export const getPreview = (id: string, page: number, pageSize: number, search: string) =>
  request<PreviewPage>(
    `/api/dataset/${id}/preview?page=${page}&page_size=${pageSize}&search=${encodeURIComponent(search)}`,
  )

export const getStatistics = (id: string) =>
  cachedGet<{ columns: string[]; rows: StatisticsRow[] }>(`/api/dataset/${id}/statistics`)

export const getFilterOptions = (id: string) =>
  cachedGet<FilterOptions>(`/api/dataset/${id}/filter-options`)

export const previewFilters = (id: string, filters: FilterState, limit = 50) =>
  request<FilterPreview>(`/api/dataset/${id}/filter`, {
    method: 'POST',
    body: JSON.stringify({ ...filters, limit }),
  })

export function filterDownloadUrl(id: string) {
  return `/api/dataset/${id}/filter/download`
}

export function cleanedDatasetDownloadUrl(id: string) {
  return `/api/dataset/${id}/download-cleaned`
}

// --- analysis ----------------------------------------------------------

export const getOverview = (id: string) =>
  cachedGet<{ overview: OverviewStats }>(`/api/analysis/${id}/overview`).then((r) => r.overview)

export const getHistogram = (id: string, column: string) =>
  cachedGet<ChartResponse<HistogramChart>>(`/api/analysis/${id}/histogram?column=${encodeURIComponent(column)}`)

export const getBoxplot = (id: string, column: string) =>
  cachedGet<ChartResponse<BoxChart>>(`/api/analysis/${id}/boxplot?column=${encodeURIComponent(column)}`)

export const getCorrelation = (id: string) =>
  cachedGet<ChartResponse<CorrelationChart>>(`/api/analysis/${id}/correlation`)

export const getMissing = (id: string) =>
  cachedGet<MissingResponse>(`/api/analysis/${id}/missing`)

export function cleaningReportDownloadUrl(id: string) {
  return `/api/analysis/${id}/cleaning-report`
}

export const getScatter = (id: string, x: string, y: string) =>
  cachedGet<ChartResponse<ScatterChart>>(
    `/api/analysis/${id}/scatter?x=${encodeURIComponent(x)}&y=${encodeURIComponent(y)}`,
  )

// --- model -------------------------------------------------------------

export const getTrainingConfig = (id: string) =>
  request<TrainingConfig>(`/api/model/config/${id}`)

export const getTargetAnalysis = (datasetId: string, target: string, override: 'auto' | ProblemType) =>
  request<{ analysis: TargetAnalysis }>('/api/model/target-analysis', {
    method: 'POST',
    body: JSON.stringify({ dataset_id: datasetId, target_column: target, problem_override: override }),
  }).then((r) => r.analysis)

export const runPreflight = (
  datasetId: string,
  target: string,
  problemType: ProblemType,
  modelChoice: string,
) =>
  request<{ validation: PreflightResult }>('/api/model/preflight', {
    method: 'POST',
    body: JSON.stringify({
      dataset_id: datasetId,
      target_column: target,
      problem_type: problemType,
      model_choice: modelChoice,
    }),
  }).then((r) => r.validation)

export const startTraining = (
  datasetId: string,
  target: string,
  problemType: ProblemType,
  modelChoice: string,
) =>
  request<{ job_id: string; status: string }>('/api/model/train', {
    method: 'POST',
    body: JSON.stringify({
      dataset_id: datasetId,
      target_column: target,
      problem_type: problemType,
      model_choice: modelChoice,
    }),
  })

export const getJob = (jobId: string) => request<JobState>(`/api/jobs/${jobId}`)

export const getModel = (modelId: string) =>
  request<{ model: ModelInfo }>(`/api/model/${modelId}`).then((r) => r.model)

export const getTrainingHistory = (datasetId: string) =>
  request<{ models: ModelInfo[] }>(`/api/model/history/${datasetId}`).then((r) => r.models)

export const getFeatureSchema = (modelId: string) =>
  cachedGet<FeatureSchema>(`/api/model/${modelId}/feature-schema`)

// --- prediction --------------------------------------------------------

export const predictSingle = (modelId: string, inputs: Record<string, unknown>) =>
  request<{ result: SingleResult }>('/api/predict/single', {
    method: 'POST',
    body: JSON.stringify({ model_id: modelId, inputs }),
  }).then((r) => r.result)

export async function predictBatch(
  modelId: string,
  file: File,
): Promise<{ batch: BatchSummary; results: PredictionPage }> {
  const form = new FormData()
  form.append('model_id', modelId)
  form.append('file', file)
  return request<{ batch: BatchSummary; results: PredictionPage }>('/api/predict/batch', {
    method: 'POST',
    body: form,
  })
}

export const getPredictionPage = (predictionId: string, page: number, pageSize: number) =>
  request<PredictionPage>(`/api/predictions/${predictionId}?page=${page}&page_size=${pageSize}`)

export function predictionDownloadUrl(predictionId: string) {
  return `/api/predictions/${predictionId}/download`
}

// --- session -----------------------------------------------------------

export const getSessionStatus = () => request<SessionStatus>('/api/session/status')

// --- dashboard (feature 1) ----------------------------------------------

export const getDashboard = (datasetId: string) =>
  request<{ dashboard: DashboardSummary }>(`/api/dashboard/${datasetId}`).then((r) => r.dashboard)

// --- prediction history (feature 7) -------------------------------------

export const getPredictionHistory = (datasetId: string) =>
  request<{ history: PredictionHistoryEntry[] }>(`/api/predictions/history/${datasetId}`).then((r) => r.history)

// --- performance monitor (feature 16) ------------------------------------

export const getPerformanceLog = (limit = 50) =>
  request<{ entries: PerformanceEntry[] }>(`/api/performance?limit=${limit}`).then((r) => r.entries)

// --- students: at-risk / profile / explorer / compare (features 18-20, 34, 37, 52) --

function explorerQuery(filters: StudentExplorerFilters): string {
  const params = new URLSearchParams()
  if (filters.page) params.set('page', String(filters.page))
  if (filters.page_size) params.set('page_size', String(filters.page_size))
  if (filters.search) params.set('search', filters.search)
  if (filters.risk && filters.risk.length) params.set('risk', filters.risk.join(','))
  if (filters.prediction) params.set('prediction', filters.prediction)
  if (filters.confidence_min !== undefined) params.set('confidence_min', String(filters.confidence_min))
  if (filters.confidence_max !== undefined) params.set('confidence_max', String(filters.confidence_max))
  if (filters.sort_by) params.set('sort_by', filters.sort_by)
  if (filters.sort_dir) params.set('sort_dir', filters.sort_dir)
  return params.toString()
}

export const getStudentCapability = (predictionId: string) =>
  request<{ capability: StudentCapability }>(`/api/students/${predictionId}/capability`).then((r) => r.capability)

export const getStudentExplorer = (predictionId: string, filters: StudentExplorerFilters = {}) =>
  request<StudentExplorerPage>(`/api/students/${predictionId}?${explorerQuery(filters)}`)

export const getStudentProfile = (predictionId: string, rowIndex: number) =>
  request<{ profile: StudentProfile }>(`/api/students/${predictionId}/profile/${rowIndex}`).then((r) => r.profile)

export const compareStudents = (predictionId: string, rowIndices: number[]) =>
  request<{ comparison: { prediction_id: string; students: StudentProfile[] } }>(
    `/api/students/${predictionId}/compare`,
    { method: 'POST', body: JSON.stringify({ row_indices: rowIndices }) },
  ).then((r) => r.comparison)

export const getConfidenceSummary = (predictionId: string) =>
  request<{ summary: ConfidenceSummary }>(`/api/students/${predictionId}/confidence-summary`).then((r) => r.summary)

export function studentExplorerDownloadUrl(predictionId: string, filters: StudentExplorerFilters = {}) {
  return `/api/students/${predictionId}/download?${explorerQuery(filters)}`
}

// --- explainability (feature 25) ------------------------------------------

export const getExplainCapability = (modelId: string) =>
  request<{ capability: ExplainCapability }>(`/api/explain/${modelId}/capability`).then((r) => r.capability)

export const getGlobalImportance = (modelId: string) =>
  request<GlobalImportanceResult>(`/api/explain/${modelId}/global`)

export const explainInstance = (modelId: string, inputs: Record<string, string | number>) =>
  request<ExplainInstanceResult>(`/api/explain/${modelId}/instance`, {
    method: 'POST',
    body: JSON.stringify({ inputs }),
  })

export const explainForPrediction = (predictionId: string, rowIndex: number) =>
  request<ExplainInstanceResult>(`/api/explain/prediction/${predictionId}/${rowIndex}`)

// --- dataset intelligence (features 1-12) ----------------------------------

export const getDatasetIntelligence = (datasetId: string) =>
  cachedGet<{ intelligence: DatasetIntelligence }>(`/api/intelligence/${datasetId}`).then((r) => r.intelligence)

export const getModelRecommendation = (datasetId: string, target: string, problemType: string) =>
  cachedGet<{ recommendation: ModelRecommendation }>(
    `/api/intelligence/${datasetId}/recommendation?target=${encodeURIComponent(target)}&problem_type=${encodeURIComponent(problemType)}`,
  ).then((r) => r.recommendation)
