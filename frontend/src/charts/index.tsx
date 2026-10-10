// Every chart the application draws. Each takes the backend's chart payload
// and nothing else, so a chart can never disagree with the data behind it.

import { Plot, useChartPalette } from './Plot'
import type {
  BoxChart,
  CorrelationChart,
  Distribution,
  HistogramChart,
  MissingChart,
  ModelCharts,
  ScatterChart,
} from '../types/api'

export function DistributionHistogram({ chart, height = 320 }: { chart: HistogramChart; height?: number }) {
  const c = useChartPalette()
  return (
    <Plot
      title={`Distribution of ${chart.column}`}
      height={height}
      data={[
        {
          type: 'bar',
          x: chart.bins,
          y: chart.counts,
          marker: { color: c.brand, line: { width: 0 } },
          hovertemplate: '%{x}<br>%{y:,} rows<extra></extra>',
        },
      ]}
      layout={{
        xaxis: { title: { text: chart.column }, tickangle: chart.kind === 'categorical' ? -30 : 0 },
        yaxis: { title: { text: 'Rows' } },
        bargap: chart.kind === 'numeric' ? 0.02 : 0.3,
      }}
    />
  )
}

export function BoxPlot({ chart, height = 320 }: { chart: BoxChart; height?: number }) {
  const c = useChartPalette()
  return (
    <Plot
      title={`Box plot of ${chart.column}`}
      height={height}
      data={[
        {
          type: 'box',
          name: chart.column,
          q1: [chart.q1],
          median: [chart.median],
          q3: [chart.q3],
          lowerfence: [chart.lower_fence],
          upperfence: [chart.upper_fence],
          mean: [chart.mean],
          y: chart.outliers.length ? [chart.outliers] : undefined,
          boxpoints: chart.outliers.length ? 'outliers' : false,
          marker: { color: c.brand, outliercolor: c.amber, size: 4 },
          line: { color: c.brandMid },
          fillcolor: c.brandFill,
        },
      ]}
      layout={{ yaxis: { title: { text: chart.column } }, xaxis: { showticklabels: false } }}
    />
  )
}

export function CorrelationHeatmap({ chart, height = 460 }: { chart: CorrelationChart; height?: number }) {
  const c = useChartPalette()
  return (
    <Plot
      title="Correlation between numeric columns"
      height={height}
      data={[
        {
          type: 'heatmap',
          x: chart.columns,
          y: chart.columns,
          z: chart.matrix,
          zmin: -1,
          zmax: 1,
          colorscale: [
            [0, c.danger],
            [0.5, c.low],
            [1, c.brand],
          ],
          hovertemplate: '%{y} vs %{x}<br>r = %{z:.2f}<extra></extra>',
          colorbar: { thickness: 10, outlinewidth: 0, tickfont: { size: 11, color: c.muted } },
        },
      ]}
      layout={{
        margin: { l: 120, r: 20, t: 16, b: 110 },
        xaxis: { tickangle: -45 },
        yaxis: { autorange: 'reversed' },
      }}
    />
  )
}

export function MissingValuesChart({ chart, height = 320 }: { chart: MissingChart; height?: number }) {
  const c = useChartPalette()
  return (
    <Plot
      title="Missing values by column"
      height={height}
      data={[
        {
          type: 'bar',
          x: chart.columns,
          y: chart.counts,
          marker: { color: c.amber },
          customdata: chart.percentages,
          hovertemplate: '%{x}<br>%{y:,} missing (%{customdata}%)<extra></extra>',
        },
      ]}
      layout={{
        xaxis: { tickangle: -30 },
        yaxis: { title: { text: 'Missing rows' } },
      }}
    />
  )
}

export function ScatterPlot({ chart, height = 380 }: { chart: ScatterChart; height?: number }) {
  const c = useChartPalette()
  return (
    <Plot
      title={`${chart.x_column} against ${chart.y_column}`}
      height={height}
      data={[
        {
          type: 'scatter',
          mode: 'markers',
          x: chart.x,
          y: chart.y,
          marker: { color: c.brand, size: 5, opacity: 0.55 },
          hovertemplate: `${chart.x_column}: %{x}<br>${chart.y_column}: %{y}<extra></extra>`,
        },
      ]}
      layout={{
        xaxis: { title: { text: chart.x_column } },
        yaxis: { title: { text: chart.y_column } },
      }}
    />
  )
}

// --- training results --------------------------------------------------

export function ConfusionMatrix({
  chart,
  height = 380,
}: {
  chart: NonNullable<ModelCharts['confusion_matrix']>
  height?: number
}) {
  const c = useChartPalette()
  const labels = chart.labels.map(String)
  const annotations = chart.matrix.flatMap((row, i) =>
    row.map((value, j) => {
      const peak = Math.max(...chart.matrix.flat())
      return {
        x: labels[j],
        y: labels[i],
        text: value.toLocaleString(),
        showarrow: false,
        font: { color: value > peak * 0.55 ? c.onBrand : c.ink, size: 13, weight: 600 },
      }
    }),
  )

  return (
    <Plot
      title="Confusion matrix"
      height={height}
      data={[
        {
          type: 'heatmap',
          x: labels,
          y: labels,
          z: chart.matrix,
          colorscale: [
            [0, c.low],
            [1, c.brand],
          ],
          hovertemplate: 'Actual %{y}, predicted %{x}<br>%{z:,} rows<extra></extra>',
          showscale: false,
        },
      ]}
      layout={{
        annotations,
        xaxis: { title: { text: 'Predicted' }, side: 'bottom' },
        yaxis: { title: { text: 'Actual' }, autorange: 'reversed' },
        margin: { l: 90, r: 20, t: 16, b: 56 },
      }}
    />
  )
}

export function RocCurve({
  chart,
  height = 380,
}: {
  chart: NonNullable<ModelCharts['roc_curve']>
  height?: number
}) {
  const c = useChartPalette()
  return (
    <Plot
      title={`ROC curve, area under curve ${chart.auc.toFixed(3)}`}
      height={height}
      data={[
        {
          type: 'scatter',
          mode: 'lines',
          x: [0, 1],
          y: [0, 1],
          line: { color: c.muted, dash: 'dash', width: 1 },
          hoverinfo: 'skip',
        },
        {
          type: 'scatter',
          mode: 'lines',
          x: chart.fpr,
          y: chart.tpr,
          fill: 'tozeroy',
          fillcolor: c.brandFill,
          line: { color: c.brand, width: 2 },
          hovertemplate: 'False positive rate %{x:.3f}<br>True positive rate %{y:.3f}<extra></extra>',
        },
      ]}
      layout={{
        xaxis: { title: { text: 'False positive rate' }, range: [0, 1] },
        yaxis: { title: { text: 'True positive rate' }, range: [0, 1], scaleanchor: 'x', scaleratio: 1 },
      }}
    />
  )
}

export function ResidualPlot({
  chart,
  height = 380,
}: {
  chart: NonNullable<ModelCharts['residuals']>
  height?: number
}) {
  const c = useChartPalette()
  return (
    <Plot
      title="Residuals against predicted values"
      height={height}
      data={[
        {
          type: 'scatter',
          mode: 'markers',
          x: chart.predicted,
          y: chart.residual,
          marker: { color: c.brand, size: 5, opacity: 0.55 },
          hovertemplate: 'Predicted %{x:.3f}<br>Residual %{y:.3f}<extra></extra>',
        },
      ]}
      layout={{
        xaxis: { title: { text: 'Predicted' } },
        yaxis: { title: { text: 'Residual' } },
        shapes: [
          {
            type: 'line',
            xref: 'paper',
            x0: 0,
            x1: 1,
            y0: 0,
            y1: 0,
            line: { color: c.danger, dash: 'dash', width: 1 },
          },
        ],
      }}
    />
  )
}

export function FeatureImportanceChart({
  chart,
  height = 380,
}: {
  chart: NonNullable<ModelCharts['feature_importance']>
  height?: number
}) {
  const c = useChartPalette()
  // Plotly draws horizontal bars bottom-up, so reversing puts the strongest
  // feature at the top where it is read first.
  const features = [...chart.features].reverse()
  const importance = [...chart.importance].reverse()

  return (
    <Plot
      title="Most influential features"
      height={height}
      data={[
        {
          type: 'bar',
          orientation: 'h',
          x: importance,
          y: features,
          marker: { color: c.brand },
          hovertemplate: '%{y}<br>importance %{x:.4f}<extra></extra>',
        },
      ]}
      layout={{
        margin: { l: 180, r: 24, t: 16, b: 44 },
        xaxis: { title: { text: 'Importance' } },
      }}
    />
  )
}

export function PredictionDistribution({
  distribution,
  height = 320,
}: {
  distribution: Distribution
  height?: number
}) {
  const c = useChartPalette()
  if (distribution.kind === 'empty') {
    return <p className="text-[13px] text-muted py-6 text-center">No predicted values to chart.</p>
  }

  const labels = distribution.kind === 'classification' ? distribution.labels : distribution.bins
  const axisTitle = distribution.kind === 'classification' ? 'Predicted class' : 'Predicted value'

  return (
    <Plot
      title="Distribution of predictions"
      height={height}
      data={[
        {
          type: 'bar',
          x: labels,
          y: distribution.counts,
          marker: { color: c.brand },
          text: distribution.kind === 'classification' ? distribution.counts.map((c) => c.toLocaleString()) : undefined,
          textposition: 'auto',
          hovertemplate: '%{x}<br>%{y:,} rows<extra></extra>',
        },
      ]}
      layout={{
        xaxis: { title: { text: axisTitle }, tickangle: -30 },
        yaxis: { title: { text: 'Rows' } },
        bargap: distribution.kind === 'classification' ? 0.35 : 0.04,
      }}
    />
  )
}
