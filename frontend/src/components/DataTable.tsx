// A paged table. Only the rows for the current page are ever in the DOM —
// the backend holds the rest — so this stays responsive on a 120,000-row
// dataset or prediction run.

import type { ReactNode } from 'react'

import { Pagination, Skeleton, formatCell } from './ui'
import type { Cell } from '../types/api'

interface Props {
  columns: string[]
  rows: Record<string, Cell>[]
  page: number
  totalPages: number
  totalRows: number
  pageSize: number
  loading?: boolean
  onPageChange: (page: number) => void
  onPageSizeChange?: (size: number) => void
  highlightColumns?: string[]
  caption?: ReactNode
  emptyMessage?: string
}

export function DataTable({
  columns,
  rows,
  page,
  totalPages,
  totalRows,
  pageSize,
  loading = false,
  onPageChange,
  onPageSizeChange,
  highlightColumns = [],
  caption,
  emptyMessage = 'No rows match this view.',
}: Props) {
  const firstRow = totalRows === 0 ? 0 : (page - 1) * pageSize + 1

  return (
    <div>
      <div className="table-scroll border border-line rounded-lg bg-surface">
        <table className="w-full border-collapse text-[13px]">
          {caption && <caption className="sr-only">{caption}</caption>}
          <thead>
            <tr className="bg-canvas">
              <th
                scope="col"
                className="text-right font-medium text-muted px-3 py-2.5 border-b border-line w-14 sticky left-0 bg-canvas"
              >
                #
              </th>
              {columns.map((column) => (
                <th
                  key={column}
                  scope="col"
                  className={`text-left font-semibold px-3 py-2.5 border-b border-line whitespace-nowrap
                    ${highlightColumns.includes(column) ? 'text-brand-deep bg-brand-soft' : 'text-ink'}`}
                >
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loading &&
              Array.from({ length: Math.min(pageSize, 8) }).map((_, index) => (
                <tr key={`skeleton-${index}`}>
                  <td className="px-3 py-2.5 border-b border-line">
                    <Skeleton className="h-3 w-6 ml-auto" />
                  </td>
                  {columns.map((column) => (
                    <td key={column} className="px-3 py-2.5 border-b border-line">
                      <Skeleton className="h-3 w-20" />
                    </td>
                  ))}
                </tr>
              ))}

            {!loading &&
              rows.map((row, index) => (
                <tr key={index} className="hover:bg-canvas/70">
                  <td className="px-3 py-2 border-b border-line text-right text-muted tabular-nums sticky left-0 bg-surface">
                    {firstRow + index}
                  </td>
                  {columns.map((column) => {
                    const value = row[column]
                    const numeric = typeof value === 'number'
                    return (
                      <td
                        key={column}
                        className={`px-3 py-2 border-b border-line whitespace-nowrap max-w-[22rem] truncate
                          ${numeric ? 'tabular-nums' : ''}
                          ${highlightColumns.includes(column) ? 'font-semibold text-brand-deep' : 'text-ink'}`}
                        title={formatCell(value)}
                      >
                        {formatCell(value)}
                      </td>
                    )
                  })}
                </tr>
              ))}

            {!loading && rows.length === 0 && (
              <tr>
                <td colSpan={columns.length + 1} className="px-3 py-10 text-center text-muted">
                  {emptyMessage}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <Pagination
        page={page}
        totalPages={totalPages}
        totalRows={totalRows}
        pageSize={pageSize}
        onPageChange={onPageChange}
        onPageSizeChange={onPageSizeChange}
      />
    </div>
  )
}
