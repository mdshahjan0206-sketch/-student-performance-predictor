// Thin wrapper around Plotly.js.
//
// Plotly is imported dynamically the first time a chart mounts, so the initial
// page load does not carry it. The module promise is memoised, so the second
// chart on the page reuses the already-loaded library.
//
// The cartesian build is used rather than the full distribution: it carries the
// bar, box, heatmap and scatter traces this app draws and nothing else, and it
// renders as SVG. That keeps charts working on machines without WebGL, which is
// the same guard the original application applied with render_mode="svg".

import { useEffect, useRef, useState } from 'react'

import { ChartSkeleton } from '../components/ui'
import { useTheme } from '../hooks/useTheme'

type PlotlyModule = typeof import('plotly.js-cartesian-dist-min')

let plotlyPromise: Promise<PlotlyModule> | null = null

function loadPlotly(): Promise<PlotlyModule> {
  if (!plotlyPromise) {
    plotlyPromise = import('plotly.js-cartesian-dist-min')
  }
  return plotlyPromise
}

/** Chart palettes are keyed by theme rather than read from CSS variables at
 *  render time: the <html> attribute is only updated after React has rendered,
 *  so a CSS read would lag one toggle behind. Values mirror index.css. Green is
 *  the lead series; blue, amber and violet are used only where a chart needs
 *  distinguishable categories, so no chart is entirely green. */
export interface ChartPalette {
  brand: string
  brandMid: string
  brandDeep: string
  blue: string
  amber: string
  violet: string
  danger: string
  ink: string
  muted: string
  grid: string
  tooltip: string
  brandFill: string
  /** Sequential scale low end for heatmaps (near the card colour). */
  low: string
  /** Text colour that reads on top of the saturated end of a heatmap. */
  onBrand: string
}

const PALETTES: Record<'dark' | 'light', ChartPalette> = {
  dark: {
    brand: '#28BE76',
    brandMid: '#50D796',
    brandDeep: '#1B9A5E',
    blue: '#6BA7DB',
    amber: '#E0A040',
    violet: '#9C8CE0',
    danger: '#E06565',
    ink: '#E8F4EC',
    muted: '#96AA9C',
    grid: 'rgba(232,244,236,0.09)',
    tooltip: '#122119',
    brandFill: 'rgba(40,190,118,0.18)',
    low: '#16291F',
    onBrand: '#05160D',
  },
  light: {
    brand: '#16A765',
    brandMid: '#5CC992',
    brandDeep: '#128650',
    blue: '#3478BE',
    amber: '#C47A0C',
    violet: '#7060BA',
    danger: '#C94242',
    ink: '#172B1F',
    muted: '#647568',
    grid: 'rgba(23,43,31,0.09)',
    tooltip: '#FFFFFF',
    brandFill: 'rgba(22,167,101,0.16)',
    low: '#EAF6EE',
    onBrand: '#FFFFFF',
  },
}

export function useChartPalette(): ChartPalette {
  const { theme } = useTheme()
  return PALETTES[theme]
}

function buildBaseLayout(c: ChartPalette) {
  return {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'Inter, system-ui, sans-serif', size: 12, color: c.muted },
    margin: { l: 56, r: 20, t: 16, b: 48 },
    hoverlabel: {
      bgcolor: c.tooltip,
      bordercolor: c.brandDeep,
      font: { color: c.ink, family: 'Inter, system-ui, sans-serif', size: 12 },
    },
    xaxis: { gridcolor: c.grid, zerolinecolor: c.grid, linecolor: c.grid, tickfont: { color: c.muted }, automargin: true },
    yaxis: { gridcolor: c.grid, zerolinecolor: c.grid, linecolor: c.grid, tickfont: { color: c.muted }, automargin: true },
    colorway: [c.brand, c.blue, c.amber, c.violet, c.brandMid],
    legend: { font: { color: c.muted } },
    showlegend: false,
  } as const
}

interface PlotProps {
  data: Record<string, unknown>[]
  layout?: Record<string, unknown>
  height?: number
  title: string
}

export function Plot({ data, layout = {}, height = 300, title }: PlotProps) {
  const node = useRef<HTMLDivElement>(null)
  const [ready, setReady] = useState(false)
  const [failed, setFailed] = useState(false)
  const palette = useChartPalette()
  const { theme } = useTheme()

  useEffect(() => {
    let cancelled = false
    let element: HTMLDivElement | null = null

    loadPlotly()
      .then((Plotly) => {
        if (cancelled || !node.current) return
        element = node.current
        setReady(true)
        return Plotly.react(
          element,
          data as never,
          { ...buildBaseLayout(palette), height, ...layout } as never,
          {
            displayModeBar: false,
            responsive: true,
          } as never,
        )
      })
      .catch(() => {
        if (!cancelled) setFailed(true)
      })

    return () => {
      cancelled = true
      if (element) {
        void loadPlotly().then((Plotly) => Plotly.purge(element as HTMLDivElement))
      }
    }
    // `theme` / `palette` are dependencies so a toggle re-draws the chart
    // with the new palette.
  }, [data, layout, height, theme, palette])

  if (failed) {
    return (
      <p className="text-[13px] text-muted py-10 text-center">
        The charting library could not be loaded. Reload the page to try again.
      </p>
    )
  }

  return (
    <div className="relative" style={{ minHeight: height }}>
      {!ready && (
        <div className="absolute inset-0">
          <ChartSkeleton height={height} />
        </div>
      )}
      <div ref={node} role="img" aria-label={title} />
    </div>
  )
}
