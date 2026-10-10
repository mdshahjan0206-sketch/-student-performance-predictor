// Shapes returned by the FastAPI backend. Kept in one file so a change to a
// payload surfaces as a type error rather than a runtime surprise.

export interface ApiError {
  code: string
  message: string
  details?: Record<string, unknown>
}

export interface ColumnProfile {
  name: string
  dtype: string
  kind: 'numeric' | 'categorical'
  missing: number
  missing_pct?: number
  distinct: number
  example: string | number | boolean | null
  min?: number | null
  max?: number | null
  mean?: number | null
  median?: number | null
  std?: number | null
  top_categories?: { value: string; count: number }[]
}

export interface OverviewStats {
  total_rows: number
  total_cols: number
  numeric_cols: string[]
  categorical_cols: string[]
  total_missing: number
  duplicate_rows: number
}

export interface DatasetSummary {
  dataset_id: string
  filename: string
  file_size: number
  num_rows: number
  num_cols: number
  column_names: string[]
  dtypes: Record<string, string>
  missing_summary: Record<string, number>
  duplicate_count: number
  detected_targets: string[]
  uploaded_at: number
  column_profiles: ColumnProfile[]
  overview: OverviewStats
  total_missing: number
  refined?: boolean
  missing_values_detected?: number
  missing_values_handled?: number
  missing_values_remaining?: number
  models_trained: number
}

export interface RefinementResult {
  status: string
  dataset_id: string
  refined: boolean
  rows: number
  cols: number
  total_missing: number
  missing_values_detected?: number
  missing_values_handled?: number
  missing_values_remaining?: number
  columns_affected: number
  numeric_cols_count: number
  categorical_cols_count: number
  target_rows_excluded: number
  ready: boolean
  message: string
  download_url?: string
}

export type Cell = string | number | boolean | null

export interface PreviewPage {
  columns: string[]
  rows: Record<string, Cell>[]
  page: number
  page_size: number
  total_rows: number
  total_pages: number
  dataset_rows: number
  search: string
}

export interface StatisticsRow {
  column: string
  count: number | null
  mean: number | null
  std: number | null
  min: number | null
  p25: number | null
  p50: number | null
  p75: number | null
  max: number | null
}

export interface FilterState {
  numeric: { column: string; min: number | null; max: number | null } | null
  categorical: { column: string; values: string[] } | null
  drop_missing: boolean
}

export interface FilterPreview {
  dataset_id: string
  matched_rows: number
  total_rows: number
  changed: boolean
  columns: string[]
  rows: Record<string, Cell>[]
  preview_rows: number
}

export interface FilterOptions {
  numeric: { name: string; min: number; max: number }[]
  categorical: { name: string; values: string[]; truncated: boolean }[]
}

// --- charts ------------------------------------------------------------

export interface HistogramChart {
  kind: 'numeric' | 'categorical'
  column: string
  bins: string[]
  counts: number[]
  bin_edges?: number[]
  truncated?: boolean
  distinct?: number
}

export interface BoxChart {
  column: string
  q1: number
  median: number
  q3: number
  lower_fence: number
  upper_fence: number
  min: number
  max: number
  mean: number
  count: number
  outliers: number[]
}

export interface CorrelationChart {
  columns: string[]
  matrix: (number | null)[][]
  truncated: boolean
  total_numeric: number
}

export interface MissingChart {
  columns: string[]
  counts: number[]
  percentages: number[]
  total_missing: number
}

export interface MissingColumnDetail {
  column: string
  type: 'Numeric' | 'Categorical'
  dtype: string
  missing_count: number
  missing_pct: number
  severity: 'None' | 'Low' | 'Moderate' | 'High' | 'Very High' | 'Unusable (100% Missing)'
  handling: string
  final_status: string
}

export interface MissingAnalysis {
  total_rows: number
  total_cols: number
  total_cells: number
  total_missing_cells: number
  overall_missing_pct: number
  rows_with_missing: number
  rows_with_missing_pct: number
  affected_columns_count: number
  highest_missing_column: {
    column: string
    count: number
    percentage: number
  } | null
  quality_status: 'Valid' | 'Needs Cleaning'
  quality_label: string
  numeric_columns_imputed_count: number
  categorical_columns_imputed_count: number
  columns_100_pct_missing: string[]
  details: MissingColumnDetail[]
  before_after: {
    before_missing_cells: number
    after_projected_feature_missing: number
  }
  explanation_message: string
  dynamic_explanation: string
}

export interface MissingResponse {
  available: boolean
  message?: string
  chart?: MissingChart
  analysis?: MissingAnalysis
}

export interface ScatterChart {
  x_column: string
  y_column: string
  x: number[]
  y: number[]
  sampled: boolean
  plotted_points: number
  total_points: number
}

export interface ChartResponse<T> {
  available: boolean
  message?: string
  chart?: T
}

// --- model -------------------------------------------------------------

export type ProblemType = 'classification' | 'regression'

export interface TargetAnalysis {
  target_column: string
  auto_problem_type: ProblemType
  problem_type: ProblemType
  overridden: boolean
  model_options: string[]
  recommended_model: string
  distinct_values: number
  missing_values: number
  target_dtype: string
  recommendation_reason?: string
  recommendation_evidence?: {
    rows: number
    problem_type: ProblemType
    numeric_features: number
    categorical_features: number
    text_features?: number
    target_column?: string
  }
  workflow_notes?: string[]
}

export interface TrainingConfig {
  dataset_id: string
  filename: string
  num_rows: number
  num_cols: number
  columns: string[]
  detected_targets: string[]
  auto_detected: boolean
  default_target: string
  model_options: Record<ProblemType, string[]>
  analysis: TargetAnalysis
}

export interface PreflightResult {
  ok: boolean
  reason: string | null
  recommended_models: string[]
  warnings: string[]
}

export interface ModelCharts {
  confusion_matrix?: { labels: (string | number)[]; matrix: number[][] }
  roc_curve?: { fpr: number[]; tpr: number[]; auc: number; positive_class: string | number }
  residuals?: {
    predicted: number[]
    residual: number[]
    sampled: boolean
    plotted_points: number
    total_points: number
  }
  feature_importance?: { features: string[]; importance: number[] }
}

export interface ModelInfo {
  model_id: string
  dataset_id: string
  dataset_filename: string
  target_column: string
  problem_type: ProblemType
  model_choice: string
  estimator_name: string
  feature_columns: string[]
  feature_count: number
  training_time: number
  n_train: number
  n_test: number
  metrics: Record<string, number>
  warnings: string[]
  target_missing_excluded?: number
  trained_at: number
  charts: ModelCharts
}

export type JobStatus = 'queued' | 'processing' | 'training' | 'evaluating' | 'completed' | 'failed'

export interface JobState {
  job_id: string
  kind: string
  status: JobStatus
  dataset_id: string
  model_id?: string
  warnings?: string[]
  elapsed: number
  error?: { code: string; message: string; recommended_models?: string[] }
}

// --- prediction --------------------------------------------------------

export interface FeatureField {
  name: string
  kind: 'numeric' | 'integer' | 'categorical' | 'text'
  min?: number
  max?: number
  step?: number
  default: number | string
  options?: (string | number)[]
  observed_min?: number
  observed_max?: number
}

export interface FeatureSchema {
  model_id: string
  target_column: string
  problem_type: ProblemType
  feature_columns: string[]
  fields: FeatureField[]
}

export interface SingleResult {
  prediction_id: string
  model_id: string
  problem_type: ProblemType
  target_column: string
  prediction: number | string
  confidence: number | null
  predicted_class: string | number | null
  had_missing?: boolean
  notice?: string
  inputs: Record<string, Cell>
}

export type Distribution =
  | { kind: 'classification'; labels: string[]; counts: number[] }
  | { kind: 'regression'; bins: string[]; counts: number[] }
  | { kind: 'empty' }

export interface BatchSummary {
  prediction_id: string
  model_id: string
  source_filename: string
  row_count: number
  columns: string[]
  has_confidence: boolean
  problem_type: ProblemType
  warnings: string[]
  distribution: Distribution
}

export interface PredictionPage {
  prediction_id: string
  model_id: string
  columns: string[]
  rows: Record<string, Cell>[]
  page: number
  page_size: number
  total_rows: number
  total_pages: number
  has_confidence: boolean
  problem_type: ProblemType
  warnings: string[]
  distribution: Distribution
}

// --- session -----------------------------------------------------------

export type Stage = 'upload' | 'analyze' | 'train' | 'predict' | 'export'

export interface SessionStatus {
  session_id: string
  active_dataset_id: string
  active_model_id: string
  stage: Stage
  last_operation: string
  last_operation_at: number | null
  dataset: { dataset_id: string; filename: string; num_rows: number; num_cols: number } | null
  model: {
    model_id: string
    model_choice: string
    problem_type: ProblemType
    target_column: string
    dataset_id: string
  } | null
}

// --- dashboard (feature 1) ----------------------------------------------

export type TargetBreakdown =
  | { kind: 'regression'; average: number; highest: number; lowest: number }
  | {
      kind: 'classification'
      labels: string[]
      counts: number[]
      pass_percentage?: number
      fail_percentage?: number
    }
  | Record<string, never>

export interface DashboardModelStatus {
  model_id: string
  model_choice: string
  target_column: string
  problem_type: ProblemType
  metrics: Record<string, number>
  is_active: boolean
}

export interface DashboardSummary {
  dataset_id: string
  filename: string
  total_records: number
  total_columns: number
  missing_values: number
  duplicate_records: number
  models_trained: number
  prediction_runs: number
  records_scored: number
  model: DashboardModelStatus | null
  target_breakdown: TargetBreakdown
}

// --- prediction history (feature 7) -------------------------------------

export interface PredictionHistoryEntry {
  prediction_id: string
  model_id: string
  dataset_id: string
  kind: string
  row_count: number
  problem_type: ProblemType
  has_confidence: boolean
  created_at: number
  warnings: string[]
  model_choice?: string
  target_column?: string
  distribution?: Distribution
}

// --- performance monitor (feature 16) -----------------------------------

export interface PerformanceEntry {
  operation: string
  duration: number
  status: 'success' | 'failed'
  timestamp: number
  meta?: Record<string, unknown>
}

// --- students: at-risk / profile / explorer / compare (features 18-20, 34, 37, 52) --

export type RiskLevel = 'high' | 'medium' | 'low'

export interface RiskCapability {
  supported: boolean
  reason: string | null
  basis: string | null
}

export interface StudentCapability {
  prediction_id: string
  problem_type: ProblemType
  has_confidence: boolean
  identifier_column: string | null
  row_count: number
  risk: RiskCapability
  prediction_labels: string[]
  model_id: string | null
  target_column: string | null
  model_choice: string | null
}

export interface StudentRow {
  [column: string]: Cell
  _row_index: number
  _risk: RiskLevel | null
  _confidence_tier: RiskLevel | null
}

export interface StudentExplorerPage {
  prediction_id: string
  columns: string[]
  identifier_column: string | null
  rows: StudentRow[]
  page: number
  page_size: number
  total_rows: number
  total_pages: number
  problem_type: ProblemType
  has_confidence: boolean
  risk: RiskCapability
}

export interface StudentExplorerFilters {
  page?: number
  page_size?: number
  search?: string
  risk?: RiskLevel[]
  prediction?: string
  confidence_min?: number
  confidence_max?: number
  sort_by?: string
  sort_dir?: 'asc' | 'desc'
}

export interface ProfileField {
  name: string
  value: Cell
}

export interface StudentProfile {
  prediction_id: string
  row_index: number
  identifier_column: string | null
  identifier_value: Cell
  display_label: string
  sections: {
    academic: ProfileField[]
    behavioral: ProfileField[]
    demographic: ProfileField[]
    other: ProfileField[]
  }
  prediction: { value: Cell; target_column: string | null; problem_type: ProblemType }
  confidence: { value: number | null; tier: RiskLevel | null }
  risk: { value: RiskLevel | null; capability: RiskCapability }
}

export interface ConfidenceSummary {
  prediction_id: string
  available: boolean
  reason?: string
  total?: number
  mean?: number | null
  min?: number | null
  max?: number | null
  high?: number
  medium?: number
  low?: number
}

// --- explainability (feature 25) ------------------------------------------

export type ExplainMethod = 'tree' | 'linear' | null

export interface ExplainCapability {
  supported: boolean
  method: ExplainMethod
  model_choice: string
  description: string
}

export interface Contribution {
  feature: string
  contribution: number
  direction: 'positive' | 'negative'
}

export interface InstanceExplanation {
  base_value: number
  predicted_value: number
  predicted_label: Cell
  top_contributions: Contribution[]
}

export interface ExplainInstanceResult {
  model_id: string
  capability: ExplainCapability
  explanation: InstanceExplanation | null
}

export interface GlobalImportanceResult {
  model_id: string
  capability: ExplainCapability
  importance: { features: string[]; importance: number[]; sample_size: number } | null
}

// --- dataset intelligence (features 1-12) ---------------------------------

export interface IntelligenceProfile {
  name: string
  dtype: string
  kind: 'numeric' | 'categorical' | 'text'
  missing: number
  missing_pct: number
  distinct: number
  meaning: string
  examples: (string | number | boolean | null)[]
  min?: number | null
  max?: number | null
  mean?: number | null
  median?: number | null
  std?: number | null
  range?: number | null
  top_categories?: { value: string; count: number; pct: number }[]
}

export interface FeatureComposition {
  numeric: { count: number; columns: string[] }
  categorical: { count: number; columns: string[] }
  text: { count: number; columns: string[] }
  total: number
}

export interface TargetOverview {
  column: string
  dtype: string
  problem_type: ProblemType
  total: number
  missing: number
  class_count?: number
  distribution?: { labels: string[]; counts: number[]; percentages: number[] }
  imbalance?: { detected: boolean; ratio: number; description: string }
  min?: number
  max?: number
  mean?: number
  median?: number
  std?: number
  spread?: number
  histogram?: { bins: string[]; counts: number[] }
}

export interface QualityObservation {
  type: 'good' | 'warn' | 'info'
  text: string
}

export interface DataQuality {
  score: number
  score_explanation: string
  total_missing: number
  missing_columns: { column: string; count: number; pct: number }[]
  duplicate_rows: number
  constant_features: string[]
  high_cardinality: { column: string; unique_values: number; ratio: number }[]
  observations: QualityObservation[]
}

export interface DatasetInsight {
  category: string
  text: string
}

export interface DatasetIntelligence {
  dataset_id: string
  filename: string
  rows: number
  columns: number
  numeric_features: number
  categorical_features: number
  text_features: number
  missing_values: number
  missing_percentage: number
  duplicate_rows: number
  target_column: string | null
  problem_type: ProblemType | null
  file_size: number
  composition: FeatureComposition
  story: string[]
  profiles: IntelligenceProfile[]
  target_overview: TargetOverview | null
  quality: DataQuality
  insights: DatasetInsight[]
}

export interface ModelRecommendation {
  recommended_model: string
  recommendation_reason: string
  recommendation_evidence: {
    rows: number
    problem_type: ProblemType
    numeric_features: number
    categorical_features: number
    text_features: number
    target_column: string
  }
  workflow_notes: string[]
  model_options: string[]
}

// Enhanced DashboardSummary with new fields
export interface EnhancedDashboardSummary extends DashboardSummary {
  numeric_features?: number
  categorical_features?: number
  numeric_columns?: string[]
  categorical_columns?: string[]
  detected_target?: string
  detected_problem_type?: ProblemType
  detected_target_breakdown?: TargetBreakdown
  at_risk_count?: number | null
}

