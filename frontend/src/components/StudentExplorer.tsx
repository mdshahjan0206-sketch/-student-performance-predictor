// Shared table for every student-level view: At-Risk Detection (18),
// Prediction Explorer (20), Confidence Analysis (37) and the Low-Confidence
// Review Queue (52) are all this same component with different starting
// filters — the backend already agrees on what "risk" and "confidence tier"
// mean (see app/ml/risk.py), so the UI only needs one table.
//
// All filtering, sorting and pagination happen on the backend: a page here
// only ever holds the current page's rows, so this stays fast even when the
// underlying prediction has 100,000+ rows.

import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { Badge, Checkbox, Pagination, Select, Skeleton, formatCell } from './ui'
import { useAsync } from '../hooks/useAsync'
import * as api from '../services/api'
import type { RiskLevel, StudentExplorerFilters, StudentRow } from '../types/api'

const RISK_TONE: Record<RiskLevel, 'danger' | 'warn' | 'brand'> = {
  high: 'danger',
  medium: 'warn',
  low: 'brand',
}

function RiskBadge({ risk }: { risk: RiskLevel | null }) {
  if (!risk) return <span className="text-muted">—</span>
  return <Badge tone={RISK_TONE[risk]}>{risk[0].toUpperCase() + risk.slice(1)}</Badge>
}

function ConfidenceBadge({ value, tier }: { value: number | null; tier: RiskLevel | null }) {
  if (value === null) return <span className="text-muted">—</span>
  const tone = tier === 'high' ? 'brand' : tier === 'medium' ? 'warn' : 'danger'
  return <Badge tone={tone as 'brand' | 'warn' | 'danger'}>{(value * 100).toFixed(0)}%</Badge>
}

interface Props {
  predictionId: string
  initialFilters?: StudentExplorerFilters
  /** Fixed risk filter the user cannot clear from this view (used by the
   *  At-Risk page); undefined means the risk filter is fully user-editable. */
  lockRisk?: RiskLevel[]
  enableCompare?: boolean
}

export function StudentExplorer({ predictionId, initialFilters, lockRisk, enableCompare = true }: Props) {
  const navigate = useNavigate()
  const [page, setPage] = useState(initialFilters?.page ?? 1)
  const [pageSize, setPageSize] = useState(initialFilters?.page_size ?? 25)
  const [search, setSearch] = useState(initialFilters?.search ?? '')
  const [riskFilter, setRiskFilter] = useState<RiskLevel[]>(lockRisk ?? initialFilters?.risk ?? [])
  const [predictionFilter, setPredictionFilter] = useState(initialFilters?.prediction ?? '')
  const [confidenceMax, setConfidenceMax] = useState<string>(
    initialFilters?.confidence_max !== undefined ? String(initialFilters.confidence_max) : '',
  )
  const [sortBy, setSortBy] = useState(initialFilters?.sort_by ?? '')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>(initialFilters?.sort_dir ?? 'asc')
  const [visibleColumns, setVisibleColumns] = useState<Set<string> | null>(null)
  const [showColumnPicker, setShowColumnPicker] = useState(false)
  const [selected, setSelected] = useState<number[]>([])

  const { data: capability } = useAsync(() => api.getStudentCapability(predictionId), [predictionId])

  const filters: StudentExplorerFilters = useMemo(
    () => ({
      page,
      page_size: pageSize,
      search: search || undefined,
      risk: riskFilter.length ? riskFilter : undefined,
      prediction: predictionFilter || undefined,
      confidence_max: confidenceMax ? Number(confidenceMax) : undefined,
      sort_by: sortBy || undefined,
      sort_dir: sortDir,
    }),
    [page, pageSize, search, riskFilter, predictionFilter, confidenceMax, sortBy, sortDir],
  )

  const { data, loading, error } = useAsync(() => api.getStudentExplorer(predictionId, filters), [
    predictionId,
    JSON.stringify(filters),
  ])

  // Any filter change other than the page itself should return to page 1.
  useEffect(() => {
    setPage(1)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [search, JSON.stringify(riskFilter), predictionFilter, confidenceMax, sortBy, sortDir])

  const featureColumns = useMemo(
    () => (data ? data.columns.filter((c) => !['Prediction', 'Confidence', 'Predicted Class'].includes(c)) : []),
    [data],
  )
  const columnsToShow = visibleColumns ? featureColumns.filter((c) => visibleColumns.has(c)) : featureColumns

  function toggleRisk(level: RiskLevel) {
    if (lockRisk) return
    setRiskFilter((prev) => (prev.includes(level) ? prev.filter((r) => r !== level) : [...prev, level]))
  }

  function toggleSort(column: string) {
    if (sortBy === column) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortBy(column)
      setSortDir('asc')
    }
  }

  function toggleSelect(rowIndex: number) {
    setSelected((prev) =>
      prev.includes(rowIndex) ? prev.filter((r) => r !== rowIndex) : prev.length >= 4 ? prev : [...prev, rowIndex],
    )
  }

  const riskSupported = capability?.risk.supported ?? true

  return (
    <div className="space-y-4">
      {capability && !capability.risk.supported && (
        <div className="rounded-lg border border-line bg-canvas px-4 py-3 text-[13px] text-muted">
          <span className="font-semibold text-ink">Risk analysis is not available for this run: </span>
          {capability.risk.reason}
        </div>
      )}
      {capability?.risk.supported && capability.risk.basis && (
        <p className="text-[12px] text-muted leading-relaxed">{capability.risk.basis}</p>
      )}

      <div className="flex flex-wrap items-end gap-3">
        <div className="min-w-[220px] flex-1">
          <label htmlFor="student-search" className="field-label">
            Search
          </label>
          <input
            id="student-search"
            className="input"
            placeholder="Search student"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        {riskSupported && (
          <div>
            <p className="field-label">Risk</p>
            <div className="flex gap-1.5">
              {(['high', 'medium', 'low'] as RiskLevel[]).map((level) => (
                <button
                  key={level}
                  type="button"
                  disabled={Boolean(lockRisk)}
                  onClick={() => toggleRisk(level)}
                  className={`h-9 px-3 rounded-lg border text-[12px] font-semibold transition-colors
                    ${riskFilter.includes(level) ? 'border-brand bg-brand text-onbrand' : 'border-line bg-surface text-muted hover:text-ink'}
                    ${lockRisk ? 'cursor-default' : ''}`}
                >
                  {level[0].toUpperCase() + level.slice(1)}
                </button>
              ))}
            </div>
          </div>
        )}

        {capability && capability.prediction_labels.length > 0 && capability.prediction_labels.length <= 12 && (
          <div className="w-[160px]">
            <Select
              id="prediction-filter"
              label="Prediction"
              value={predictionFilter}
              onChange={setPredictionFilter}
              options={[{ value: '', label: 'Any' }, ...capability.prediction_labels.map((l) => ({ value: l, label: l }))]}
            />
          </div>
        )}

        {capability?.has_confidence && (
          <div className="w-[160px]">
            <label htmlFor="confidence-max" className="field-label">
              Max confidence
            </label>
            <input
              id="confidence-max"
              className="input"
              type="number"
              min={0}
              max={1}
              step={0.05}
              placeholder="e.g. 0.5"
              value={confidenceMax}
              onChange={(e) => setConfidenceMax(e.target.value)}
            />
          </div>
        )}

        <div className="flex items-center gap-2 ml-auto">
          <button type="button" className="btn-secondary h-9 px-3 text-[12px]" onClick={() => setShowColumnPicker((v) => !v)}>
            Columns
          </button>
          <a
            className="btn-secondary h-9 px-3 text-[12px]"
            href={api.studentExplorerDownloadUrl(predictionId, filters)}
            download
          >
            Export CSV
          </a>
          {enableCompare && selected.length >= 2 && (
            <button
              type="button"
              className="btn-primary h-9 px-3 text-[12px]"
              onClick={() => navigate(`/insights/compare/${predictionId}?rows=${selected.join(',')}`)}
            >
              Compare ({selected.length})
            </button>
          )}
        </div>
      </div>

      {showColumnPicker && (
        <div className="card p-3 flex flex-wrap gap-x-4 gap-y-2">
          {featureColumns.map((col) => (
            <Checkbox
              key={col}
              id={`col-${col}`}
              label={col}
              checked={visibleColumns ? visibleColumns.has(col) : true}
              onChange={(checked) => {
                setVisibleColumns((prev) => {
                  const next = new Set(prev ?? featureColumns)
                  if (checked) next.add(col)
                  else next.delete(col)
                  return next
                })
              }}
            />
          ))}
        </div>
      )}

      {error && <div className="rounded-lg border border-danger/25 bg-danger-soft px-4 py-3 text-[13px] text-danger">{error}</div>}

      <div className="table-scroll border border-line rounded-lg bg-surface">
        <table className="w-full border-collapse text-[13px]">
          <thead>
            <tr className="bg-canvas">
              {enableCompare && <th className="w-9 px-3 py-2.5 border-b border-line" />}
              <th className="text-left font-semibold px-3 py-2.5 border-b border-line whitespace-nowrap">Student</th>
              <th
                className="text-left font-semibold px-3 py-2.5 border-b border-line whitespace-nowrap cursor-pointer"
                onClick={() => toggleSort('Prediction')}
              >
                Prediction {sortBy === 'Prediction' && (sortDir === 'asc' ? '▲' : '▼')}
              </th>
              {data?.has_confidence && (
                <th
                  className="text-left font-semibold px-3 py-2.5 border-b border-line whitespace-nowrap cursor-pointer"
                  onClick={() => toggleSort('Confidence')}
                >
                  Confidence {sortBy === 'Confidence' && (sortDir === 'asc' ? '▲' : '▼')}
                </th>
              )}
              {riskSupported && (
                <th
                  className="text-left font-semibold px-3 py-2.5 border-b border-line whitespace-nowrap cursor-pointer"
                  onClick={() => toggleSort('_risk')}
                >
                  Risk {sortBy === '_risk' && (sortDir === 'asc' ? '▲' : '▼')}
                </th>
              )}
              {columnsToShow.map((col) => (
                <th
                  key={col}
                  className="text-left font-medium text-muted px-3 py-2.5 border-b border-line whitespace-nowrap cursor-pointer"
                  onClick={() => toggleSort(col)}
                >
                  {col} {sortBy === col && (sortDir === 'asc' ? '▲' : '▼')}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loading &&
              Array.from({ length: 8 }).map((_, i) => (
                <tr key={i}>
                  <td colSpan={columnsToShow.length + 4} className="px-3 py-2.5 border-b border-line">
                    <Skeleton className="h-3 w-full" />
                  </td>
                </tr>
              ))}

            {!loading &&
              data?.rows.map((row: StudentRow) => {
                const label = data.identifier_column ? formatCell(row[data.identifier_column]) : `Row ${row._row_index + 1}`
                return (
                  <tr key={row._row_index} className="hover:bg-canvas/70">
                    {enableCompare && (
                      <td className="px-3 py-2 border-b border-line">
                        <input
                          type="checkbox"
                          className="w-4 h-4 rounded border-line text-brand"
                          checked={selected.includes(row._row_index)}
                          onChange={() => toggleSelect(row._row_index)}
                          aria-label={`Select ${label}`}
                        />
                      </td>
                    )}
                    <td className="px-3 py-2 border-b border-line font-medium">
                      <button
                        type="button"
                        className="text-brand-deep hover:underline"
                        onClick={() => navigate(`/insights/profile/${predictionId}/${row._row_index}`)}
                      >
                        {label}
                      </button>
                    </td>
                    <td className="px-3 py-2 border-b border-line">{formatCell(row.Prediction)}</td>
                    {data.has_confidence && (
                      <td className="px-3 py-2 border-b border-line">
                        <ConfidenceBadge value={row.Confidence as number | null} tier={row._confidence_tier} />
                      </td>
                    )}
                    {riskSupported && (
                      <td className="px-3 py-2 border-b border-line">
                        <RiskBadge risk={row._risk} />
                      </td>
                    )}
                    {columnsToShow.map((col) => (
                      <td key={col} className="px-3 py-2 border-b border-line whitespace-nowrap max-w-[18rem] truncate text-muted" title={formatCell(row[col])}>
                        {formatCell(row[col])}
                      </td>
                    ))}
                  </tr>
                )
              })}

            {!loading && data && data.rows.length === 0 && (
              <tr>
                <td colSpan={columnsToShow.length + 4} className="px-3 py-10 text-center text-muted">
                  No students match this view.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {data && (
        <Pagination
          page={data.page}
          totalPages={data.total_pages}
          totalRows={data.total_rows}
          pageSize={data.page_size}
          onPageChange={setPage}
          onPageSizeChange={setPageSize}
          rowNoun="students"
        />
      )}
    </div>
  )
}
