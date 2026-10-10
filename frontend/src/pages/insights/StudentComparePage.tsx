// Feature 34 (Student Comparison). Compares 2-4 students selected from the
// Prediction Explorer / At-Risk table side by side.

import { useMemo } from 'react'
import { Link, useParams, useSearchParams } from 'react-router-dom'

import { Alert, Badge, PageHeader, Skeleton, formatCell } from '../../components/ui'
import { useAsync } from '../../hooks/useAsync'
import * as api from '../../services/api'
import type { RiskLevel } from '../../types/api'

const RISK_TONE: Record<RiskLevel, 'danger' | 'warn' | 'brand'> = {
  high: 'danger',
  medium: 'warn',
  low: 'brand',
}

const SECTIONS = ['academic', 'behavioral', 'demographic', 'other'] as const

export function StudentComparePage() {
  const { predictionId } = useParams()
  const [searchParams] = useSearchParams()
  const rowIndices = useMemo(
    () => (searchParams.get('rows') ?? '').split(',').filter(Boolean).map(Number),
    [searchParams],
  )

  const { data: comparison, loading, error } = useAsync(
    () => api.compareStudents(predictionId!, rowIndices),
    [predictionId, rowIndices.join(',')],
    { enabled: Boolean(predictionId) && rowIndices.length >= 2 },
  )

  const allFieldNames = useMemo(() => {
    if (!comparison) return { academic: [] as string[], behavioral: [] as string[], demographic: [] as string[], other: [] as string[] }
    const names: Record<(typeof SECTIONS)[number], Set<string>> = {
      academic: new Set(), behavioral: new Set(), demographic: new Set(), other: new Set(),
    }
    for (const student of comparison.students) {
      for (const section of SECTIONS) {
        for (const field of student.sections[section]) names[section].add(field.name)
      }
    }
    return {
      academic: [...names.academic], behavioral: [...names.behavioral],
      demographic: [...names.demographic], other: [...names.other],
    }
  }, [comparison])

  function valueFor(studentIndex: number, section: (typeof SECTIONS)[number], name: string) {
    const fields = comparison!.students[studentIndex].sections[section]
    return fields.find((f) => f.name === name)?.value ?? null
  }

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Insights"
        title="Compare students"
        description="Side-by-side comparison of performance, prediction, confidence and risk for the selected students."
      />

      {rowIndices.length < 2 && (
        <Alert tone="warn" title="Select students to compare">
          Open the Prediction Explorer, tick 2-4 students, then choose Compare.
        </Alert>
      )}

      {error && <Alert tone="danger" title="Could not load comparison">{error}</Alert>}
      {loading && <Skeleton className="h-72 rounded-card" />}

      {comparison && (
        <div className="table-scroll border border-line rounded-lg bg-surface">
          <table className="w-full border-collapse text-[13px]">
            <thead>
              <tr className="bg-canvas">
                <th className="text-left font-semibold px-3 py-2.5 border-b border-line sticky left-0 bg-canvas">Field</th>
                {comparison.students.map((s) => (
                  <th key={s.row_index} className="text-left font-semibold px-3 py-2.5 border-b border-line whitespace-nowrap">
                    {s.display_label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr className="bg-brand-soft/40">
                <td className="px-3 py-2 border-b border-line font-semibold sticky left-0 bg-brand-soft/40">Prediction</td>
                {comparison.students.map((s) => (
                  <td key={s.row_index} className="px-3 py-2 border-b border-line font-semibold">{formatCell(s.prediction.value)}</td>
                ))}
              </tr>
              <tr>
                <td className="px-3 py-2 border-b border-line font-semibold sticky left-0 bg-surface">Confidence</td>
                {comparison.students.map((s) => (
                  <td key={s.row_index} className="px-3 py-2 border-b border-line">
                    {s.confidence.value !== null ? `${(s.confidence.value * 100).toFixed(0)}%` : '—'}
                  </td>
                ))}
              </tr>
              <tr>
                <td className="px-3 py-2 border-b border-line font-semibold sticky left-0 bg-surface">Risk</td>
                {comparison.students.map((s) => (
                  <td key={s.row_index} className="px-3 py-2 border-b border-line">
                    {s.risk.value ? <Badge tone={RISK_TONE[s.risk.value]}>{s.risk.value}</Badge> : '—'}
                  </td>
                ))}
              </tr>

              {SECTIONS.map((section) =>
                allFieldNames[section].map((name) => (
                  <tr key={`${section}-${name}`} className="hover:bg-canvas/70">
                    <td className="px-3 py-2 border-b border-line text-muted sticky left-0 bg-surface" title={section}>
                      {name}
                    </td>
                    {comparison.students.map((s, i) => (
                      <td key={s.row_index} className="px-3 py-2 border-b border-line">{formatCell(valueFor(i, section, name))}</td>
                    ))}
                  </tr>
                )),
              )}
            </tbody>
          </table>
        </div>
      )}

      {predictionId && (
        <Link className="text-[12px] text-brand-deep hover:underline" to={`/insights/explorer/${predictionId}`}>
          ← Back to prediction explorer
        </Link>
      )}
    </div>
  )
}
