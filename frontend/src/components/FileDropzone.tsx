// Drag-and-drop uploader. The drop zone is a real button so it works from the
// keyboard, and the same element accepts a dropped file or opens the picker.

import { useRef, useState } from 'react'
import type { DragEvent } from 'react'

interface Props {
  onFile: (file: File) => void
  accept?: string
  busy?: boolean
  progress?: number
  hint?: string
  label?: string
}

export function FileDropzone({
  onFile,
  accept = '.csv,.xlsx',
  busy = false,
  progress = 0,
  hint = 'CSV or Excel, up to 200 MB. Files stay on your machine.',
  label = 'Drop a dataset here',
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  function handleDrop(event: DragEvent<HTMLButtonElement>) {
    event.preventDefault()
    setDragging(false)
    const file = event.dataTransfer.files?.[0]
    if (file && !busy) onFile(file)
  }

  return (
    <div>
      <button
        type="button"
        onClick={() => !busy && inputRef.current?.click()}
        onDragOver={(event) => {
          event.preventDefault()
          if (!busy) setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        disabled={busy}
        aria-describedby="dropzone-hint"
        className={`w-full rounded-card border-2 border-dashed px-6 py-10 text-center transition-colors
          ${dragging ? 'border-brand bg-brand-soft' : 'border-line bg-surface hover:border-brand/50 hover:bg-brand-soft/40'}
          ${busy ? 'cursor-wait opacity-80' : 'cursor-pointer'}`}
      >
        <p className="font-display text-[15px] font-bold text-ink">{busy ? 'Reading your file' : label}</p>
        <p id="dropzone-hint" className="text-[13px] text-muted mt-1">
          {busy ? 'Validating columns and counting rows' : `Or click to browse. ${hint}`}
        </p>

        {busy && (
          <div className="mt-5 max-w-sm mx-auto">
            <div className="h-1.5 rounded-full bg-line overflow-hidden">
              <div
                className="h-full bg-brand transition-[width] duration-200"
                style={{ width: `${Math.max(progress, 6)}%` }}
              />
            </div>
            <p className="text-[12px] text-muted mt-2 tabular-nums">
              {progress < 100 ? `Uploading ${progress}%` : 'Processing on the server'}
            </p>
          </div>
        )}
      </button>

      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="sr-only"
        onChange={(event) => {
          const file = event.target.files?.[0]
          if (file) onFile(file)
          event.target.value = ''
        }}
      />
    </div>
  )
}
