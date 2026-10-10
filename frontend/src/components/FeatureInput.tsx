// Single form field for one model feature, driven entirely by the feature
// schema the backend already builds from the training data (min/max/options
// come from what was actually observed, nothing invented here). Shared by
// the Predict page's single-student form and the What-If Simulator (36).

import { formatNumber } from './ui'
import type { FeatureField } from '../types/api'

export function FeatureInput({
  field,
  value,
  onChange,
}: {
  field: FeatureField
  value: string | number
  onChange: (value: string | number) => void
}) {
  const id = `feature-${field.name}`

  if (field.kind === 'categorical' && field.options) {
    return (
      <div>
        <label htmlFor={id} className="field-label">
          {field.name}
        </label>
        <select id={id} className="input" value={String(value)} onChange={(event) => onChange(event.target.value)}>
          {field.options.map((option) => (
            <option key={String(option)} value={String(option)}>
              {String(option)}
            </option>
          ))}
        </select>
      </div>
    )
  }

  if (field.kind === 'numeric' || field.kind === 'integer') {
    return (
      <div>
        <label htmlFor={id} className="field-label">
          {field.name}
        </label>
        <input
          id={id}
          type="number"
          className="input tabular-nums"
          value={value}
          min={field.min}
          max={field.max}
          step={field.step}
          onChange={(event) => onChange(event.target.value === '' ? '' : Number(event.target.value))}
        />
        <p className="text-[12px] text-muted mt-1 tabular-nums">
          Seen in the data: {formatNumber(field.observed_min, 2)} to {formatNumber(field.observed_max, 2)}
        </p>
      </div>
    )
  }

  return (
    <div>
      <label htmlFor={id} className="field-label">
        {field.name}
      </label>
      <input id={id} type="text" className="input" value={String(value)} onChange={(event) => onChange(event.target.value)} />
    </div>
  )
}
