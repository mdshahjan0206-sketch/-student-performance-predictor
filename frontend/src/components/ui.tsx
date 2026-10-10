// Shared building blocks. Everything here is presentational — no component in
// this file fetches data or touches workspace state.

import type { CSSProperties, ReactNode } from 'react'

// --- layout ------------------------------------------------------------

export function Card({
  children,
  className = '',
  padded = true,
  hoverable = false,
}: {
  children: ReactNode
  className?: string
  padded?: boolean
  hoverable?: boolean
}) {
  return <section className={`card ${padded ? 'p-5' : ''} ${hoverable ? 'card-hover' : ''} ${className}`}>{children}</section>
}

export function SectionHeading({
  title,
  description,
  action,
}: {
  title: string
  description?: string
  action?: ReactNode
}) {
  return (
    <div className="flex items-start justify-between gap-4 mb-3">
      <div className="min-w-0">
        <h2 className="font-display text-[17px] font-bold text-ink tracking-[-0.01em]">{title}</h2>
        {description && <p className="text-[13px] text-muted mt-0.5 max-w-[68ch]">{description}</p>}
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </div>
  )
}

export function PageHeader({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow?: string
  title: string
  description?: string
  action?: ReactNode
}) {
  return (
    <header className="flex flex-wrap items-end justify-between gap-4 mb-6">
      <div className="min-w-0">
        {eyebrow && <p className="text-[13px] font-semibold text-brand mb-1">{eyebrow}</p>}
        <h1 className="font-display text-[26px] leading-tight font-bold text-ink tracking-[-0.02em]">
          {title}
        </h1>
        {description && <p className="text-sm text-muted mt-1.5 max-w-[72ch]">{description}</p>}
      </div>
      {action}
    </header>
  )
}

// --- metrics -----------------------------------------------------------

export function MetricCard({
  label,
  value,
  hint,
  tone = 'default',
}: {
  label: string
  value: ReactNode
  hint?: string
  tone?: 'default' | 'warn' | 'danger' | 'brand'
}) {
  const valueTone = {
    default: 'text-ink',
    brand: 'text-brand-mid',
    warn: 'text-amber',
    danger: 'text-danger',
  }[tone]

  return (
    <div className={`card p-4 ${tone === 'brand' ? 'border-brand/35' : ''}`}>
      <p className="text-[12px] font-medium text-muted">{label}</p>
      <p className={`font-display text-[22px] font-semibold mt-1 tabular-nums truncate ${valueTone}`} title={String(value)}>
        {value}
      </p>
      {hint && <p className="text-[11px] text-muted mt-0.5 truncate">{hint}</p>}
    </div>
  )
}

export function MetricGrid({ children, columns = 4 }: { children: ReactNode; columns?: number }) {
  const cols: Record<number, string> = {
    3: 'sm:grid-cols-3',
    4: 'sm:grid-cols-2 lg:grid-cols-4',
    5: 'sm:grid-cols-3 lg:grid-cols-5',
    6: 'sm:grid-cols-3 lg:grid-cols-6',
  }
  return <div className={`grid grid-cols-2 gap-3 ${cols[columns] ?? cols[4]}`}>{children}</div>
}


/** Green progress ring. `value` is a 0-1 fraction of something real (for
 *  example model confidence); the caller decides what the centre text says. */
export function ProgressRing({
  value,
  label,
  caption,
  size = 132,
}: {
  value: number
  label: string
  caption: string
  size?: number
}) {
  const clamped = Math.min(1, Math.max(0, value))
  const stroke = 9
  const r = (size - stroke) / 2
  const circumference = 2 * Math.PI * r
  const gradientId = `progress-ring-${size}`
  return (
    <div className="relative shrink-0" style={{ width: size, height: size }} role="img" aria-label={`${caption} ${label}`}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="-rotate-90">
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="rgb(var(--color-brand-mid))" />
            <stop offset="100%" stopColor="rgb(var(--color-brand))" />
          </linearGradient>
        </defs>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgb(var(--color-line))" strokeWidth={stroke} />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={`url(#${gradientId})`}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - clamped)}
          style={{ filter: 'drop-shadow(0 0 6px rgb(var(--color-brand) / 0.55))', transition: 'stroke-dashoffset 600ms cubic-bezier(.16,1,.3,1)' }}
        />
      </svg>
      <div className="absolute inset-0 grid place-content-center text-center">
        <p className="font-display text-[26px] font-semibold text-ink tabular-nums leading-none">{label}</p>
        <p className="text-[11px] text-muted mt-1">{caption}</p>
      </div>
    </div>
  )
}

// --- status ------------------------------------------------------------

type Tone = 'brand' | 'neutral' | 'warn' | 'danger' | 'info'

const toneStyles: Record<Tone, string> = {
  brand: 'bg-brand-soft text-brand-deep border-brand/20',
  neutral: 'bg-canvas text-muted border-line',
  warn: 'bg-amber-soft text-amber border-amber/20',
  danger: 'bg-danger-soft text-danger border-danger/20',
  info: 'glass text-ink',
}

export function Badge({ children, tone = 'neutral' }: { children: ReactNode; tone?: Tone }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 h-6 px-2.5 rounded-full border text-[12px] font-semibold ${toneStyles[tone]}`}
    >
      {children}
    </span>
  )
}

export function Dot({ tone = 'neutral' }: { tone?: Tone }) {
  const colors: Record<Tone, string> = {
    brand: 'bg-brand',
    neutral: 'bg-muted/40',
    warn: 'bg-amber',
    danger: 'bg-danger',
    info: 'bg-ink',
  }
  return <span className={`inline-block w-1.5 h-1.5 rounded-full ${colors[tone]}`} aria-hidden />
}

/** Errors state what happened and what to do; they do not apologise. */
export function Alert({
  tone = 'info',
  title,
  children,
}: {
  tone?: 'info' | 'warn' | 'danger' | 'brand'
  title?: string
  children?: ReactNode
}) {
  const styles = {
    info: 'glass text-ink',
    brand: 'bg-brand-soft border-brand/20 text-brand-deep',
    warn: 'bg-amber-soft border-amber/25 text-amber',
    danger: 'bg-danger-soft border-danger/25 text-danger',
  }[tone]

  return (
    <div className={`rounded-lg border px-4 py-3 text-[13px] leading-relaxed ${styles}`} role={tone === 'danger' ? 'alert' : undefined}>
      {title && <p className="font-semibold mb-0.5">{title}</p>}
      {children}
    </div>
  )
}

export function EmptyState({
  title,
  description,
  action,
}: {
  title: string
  description: string
  action?: ReactNode
}) {
  return (
    <div className="card px-6 py-12 text-center">
      <h3 className="font-display text-[17px] font-bold text-ink">{title}</h3>
      <p className="text-[13px] text-muted mt-1.5 max-w-[46ch] mx-auto">{description}</p>
      {action && <div className="mt-5 flex justify-center">{action}</div>}
    </div>
  )
}

// --- loading -----------------------------------------------------------

export function Skeleton({
  className = 'h-4 w-full',
  style,
}: {
  className?: string
  style?: CSSProperties
}) {
  return (
    <div className={`relative overflow-hidden rounded bg-line/60 ${className}`} style={style} aria-hidden>
      <div className="absolute inset-y-0 -left-1/3 w-1/3 bg-gradient-to-r from-transparent via-ink/10 to-transparent animate-sweep" />
    </div>
  )
}

export function ChartSkeleton({ height = 280 }: { height?: number }) {
  return (
    <div className="flex items-end gap-2 px-2" style={{ height }} aria-label="Loading chart">
      {[48, 72, 40, 88, 64, 96, 56, 80].map((barHeight, index) => (
        <Skeleton key={index} className="flex-1 rounded-t" style={{ height: `${barHeight}%` }} />
      ))}
    </div>
  )
}

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-[13px] text-muted">
      <span
        className="w-3.5 h-3.5 rounded-full border-2 border-line border-t-brand animate-spin motion-reduce:animate-none"
        aria-hidden
      />
      {label}
    </span>
  )
}

// --- form controls -----------------------------------------------------

export function Select({
  label,
  value,
  onChange,
  options,
  hint,
  id,
  disabled,
}: {
  label: string
  value: string
  onChange: (value: string) => void
  options: { value: string; label: string }[]
  hint?: string
  id: string
  disabled?: boolean
}) {
  return (
    <div>
      <label htmlFor={id} className="field-label">
        {label}
      </label>
      <select
        id={id}
        className="input"
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      {hint && <p className="text-[12px] text-muted mt-1">{hint}</p>}
    </div>
  )
}

export function SegmentedControl({
  label,
  value,
  onChange,
  options,
  name,
}: {
  label: string
  value: string
  onChange: (value: string) => void
  options: { value: string; label: string }[]
  name: string
}) {
  return (
    <fieldset>
      <legend className="field-label">{label}</legend>
      <div className="inline-flex p-0.5 bg-canvas/60 border border-line rounded-lg" role="radiogroup">
        {options.map((option) => {
          const active = option.value === value
          return (
            <label
              key={option.value}
              className={`px-3 h-9 inline-flex items-center rounded-[7px] text-[13px] font-semibold cursor-pointer transition-colors
                ${active ? 'bg-brand/15 text-brand-deep border border-brand/40' : 'text-muted hover:text-ink border border-transparent'}`}
            >
              <input
                type="radio"
                name={name}
                value={option.value}
                checked={active}
                onChange={() => onChange(option.value)}
                className="sr-only"
              />
              {option.label}
            </label>
          )
        })}
      </div>
    </fieldset>
  )
}

export function Checkbox({
  label,
  checked,
  onChange,
  id,
}: {
  label: string
  checked: boolean
  onChange: (checked: boolean) => void
  id: string
}) {
  return (
    <label htmlFor={id} className="inline-flex items-center gap-2.5 text-[13px] text-ink cursor-pointer select-none">
      <input
        id={id}
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        className="w-4 h-4 rounded border-line text-brand focus:ring-brand/30"
      />
      {label}
    </label>
  )
}

const PAGE_SIZES = [10, 25, 50, 100, 200]

/** Shared page-size + prev/next footer, used by DataTable and StudentTable
 *  so both paginated views behave identically. */
export function Pagination({
  page,
  totalPages,
  totalRows,
  pageSize,
  onPageChange,
  onPageSizeChange,
  rowNoun = '',
}: {
  page: number
  totalPages: number
  totalRows: number
  pageSize: number
  onPageChange: (page: number) => void
  onPageSizeChange?: (size: number) => void
  rowNoun?: string
}) {
  const firstRow = totalRows === 0 ? 0 : (page - 1) * pageSize + 1
  const lastRow = Math.min(page * pageSize, totalRows)
  const suffix = rowNoun ? ` ${rowNoun}` : ''

  return (
    <div className="flex flex-wrap items-center justify-between gap-3 mt-3">
      <p className="text-[12px] text-muted tabular-nums">
        {totalRows === 0
          ? `No${suffix ? suffix : ' rows'}`
          : `Showing ${firstRow.toLocaleString()}–${lastRow.toLocaleString()} of ${totalRows.toLocaleString()}${suffix}`}
      </p>

      <div className="flex items-center gap-2">
        {onPageSizeChange && (
          <label className="flex items-center gap-2 text-[12px] text-muted">
            Rows
            <select
              className="h-8 px-2 bg-surface border border-line rounded-md text-[12px] text-ink"
              value={pageSize}
              onChange={(event) => onPageSizeChange(Number(event.target.value))}
            >
              {PAGE_SIZES.map((size) => (
                <option key={size} value={size}>
                  {size}
                </option>
              ))}
            </select>
          </label>
        )}

        <nav className="flex items-center gap-1" aria-label="Pagination">
          <button type="button" className="btn-secondary h-8 px-2.5 text-[12px]" onClick={() => onPageChange(1)} disabled={page <= 1}>
            First
          </button>
          <button type="button" className="btn-secondary h-8 px-2.5 text-[12px]" onClick={() => onPageChange(page - 1)} disabled={page <= 1}>
            Previous
          </button>
          <span className="px-2 text-[12px] text-muted tabular-nums">
            Page {page.toLocaleString()} of {Math.max(totalPages, 1).toLocaleString()}
          </span>
          <button
            type="button"
            className="btn-secondary h-8 px-2.5 text-[12px]"
            onClick={() => onPageChange(page + 1)}
            disabled={page >= totalPages}
          >
            Next
          </button>
          <button
            type="button"
            className="btn-secondary h-8 px-2.5 text-[12px]"
            onClick={() => onPageChange(totalPages)}
            disabled={page >= totalPages}
          >
            Last
          </button>
        </nav>
      </div>
    </div>
  )
}

// --- misc --------------------------------------------------------------

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export function formatNumber(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  if (Number.isInteger(value) && Math.abs(value) < 1e6) return value.toLocaleString()
  if (Math.abs(value) >= 1e6 || (Math.abs(value) < 1e-3 && value !== 0)) return value.toExponential(2)
  return value.toFixed(digits)
}

export function formatCell(value: unknown): string {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'number') return Number.isInteger(value) ? value.toLocaleString() : formatNumber(value, 4)
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  return String(value)
}
