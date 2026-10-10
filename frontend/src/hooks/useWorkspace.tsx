// The app's shared state: which dataset is active, which model is trained,
// where the user is in the workflow, and what happened last.
//
// This is the client-side mirror of the backend session. It exists so moving
// between pages re-uses what is already loaded instead of re-fetching — the
// behaviour the migration was asked for: no re-upload, no retraining, no
// full-page rerun when switching pages.

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'

import * as api from '../services/api'
import type { DatasetSummary, ModelInfo, Stage } from '../types/api'

interface WorkspaceValue {
  dataset: DatasetSummary | null
  model: ModelInfo | null
  stage: Stage
  lastOperation: string
  ready: boolean
  setDataset: (dataset: DatasetSummary) => void
  setModel: (model: ModelInfo) => void
  setStage: (stage: Stage) => void
  note: (operation: string) => void
  refresh: () => Promise<void>
}

const WorkspaceContext = createContext<WorkspaceValue | null>(null)

export function WorkspaceProvider({ children }: { children: ReactNode }) {
  const [dataset, setDatasetState] = useState<DatasetSummary | null>(null)
  const [model, setModelState] = useState<ModelInfo | null>(null)
  const [stage, setStage] = useState<Stage>('upload')
  const [lastOperation, setLastOperation] = useState('')
  const [ready, setReady] = useState(false)

  // Rehydrate from the backend on load, so a page refresh keeps the active
  // dataset and model instead of dropping the user back to step one.
  const refresh = useCallback(async () => {
    try {
      const status = await api.getSessionStatus()
      setStage(status.stage)
      setLastOperation(status.last_operation)

      if (status.dataset) {
        const summary = await api.getDatasetSummary(status.dataset.dataset_id)
        setDatasetState(summary)
      } else {
        setDatasetState(null)
      }

      if (status.model) {
        const info = await api.getModel(status.model.model_id)
        setModelState(info)
      } else {
        setModelState(null)
      }
    } catch {
      // A cold or unreachable backend just means an empty workspace; the
      // page-level error states explain the rest.
      setDatasetState(null)
      setModelState(null)
    } finally {
      setReady(true)
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const setDataset = useCallback((next: DatasetSummary) => {
    // A new dataset always drops the trained model, matching the backend rule.
    setDatasetState(next)
    setModelState(null)
    setStage('analyze')
    setLastOperation(`Uploaded ${next.filename}`)
  }, [])

  const setModel = useCallback((next: ModelInfo) => {
    setModelState(next)
    setStage('predict')
    setLastOperation(`Trained ${next.model_choice} on ${next.target_column}`)
  }, [])

  const note = useCallback((operation: string) => setLastOperation(operation), [])

  const value = useMemo<WorkspaceValue>(
    () => ({ dataset, model, stage, lastOperation, ready, setDataset, setModel, setStage, note, refresh }),
    [dataset, model, stage, lastOperation, ready, setDataset, setModel, note, refresh],
  )

  return <WorkspaceContext.Provider value={value}>{children}</WorkspaceContext.Provider>
}

export function useWorkspace(): WorkspaceValue {
  const value = useContext(WorkspaceContext)
  if (!value) throw new Error('useWorkspace must be used inside WorkspaceProvider')
  return value
}
