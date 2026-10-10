import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'

import { ScrollToTop } from './components/ScrollToTop'
import { ThemeProvider } from './hooks/useTheme'
import { WorkspaceProvider, useWorkspace } from './hooks/useWorkspace'
import { AppLayout } from './layouts/AppLayout'
import { AboutPage } from './pages/AboutPage'
import { AnalysisPage } from './pages/AnalysisPage'
import { DashboardPage } from './pages/DashboardPage'
import { HomePage } from './pages/HomePage'
import { AtRiskPage } from './pages/insights/AtRiskPage'
import { ExplainabilityPage } from './pages/insights/ExplainabilityPage'
import { PredictionExplorerPage } from './pages/insights/PredictionExplorerPage'
import { StudentComparePage } from './pages/insights/StudentComparePage'
import { StudentProfilePage } from './pages/insights/StudentProfilePage'
import { WhatIfPage } from './pages/insights/WhatIfPage'
import { ModelsPage } from './pages/ModelsPage'
import { PerformancePage } from './pages/PerformancePage'
import { PredictPage } from './pages/PredictPage'
import { PredictionHistoryPage } from './pages/PredictionHistoryPage'
import { TrainPage } from './pages/TrainPage'
import { UploadPage } from './pages/UploadPage'

/** Pages render only once the workspace has asked the backend what is active,
 *  so a refresh never flashes an empty state before the real one. */
function Routed() {
  const { ready } = useWorkspace()

  if (!ready) {
    return (
      <div className="min-h-screen grid place-items-center">
        <p className="text-[13px] text-muted">Loading your workspace</p>
      </div>
    )
  }

  return (
    <AppLayout>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/analysis" element={<AnalysisPage />} />
        <Route path="/train" element={<TrainPage />} />
        <Route path="/predict" element={<PredictPage />} />
        <Route path="/models" element={<ModelsPage />} />
        <Route path="/predictions/history" element={<PredictionHistoryPage />} />
        <Route path="/insights/at-risk" element={<AtRiskPage />} />
        <Route path="/insights/at-risk/:predictionId" element={<AtRiskPage />} />
        <Route path="/insights/explorer" element={<PredictionExplorerPage />} />
        <Route path="/insights/explorer/:predictionId" element={<PredictionExplorerPage />} />
        <Route path="/insights/profile/:predictionId/:rowIndex" element={<StudentProfilePage />} />
        <Route path="/insights/compare/:predictionId" element={<StudentComparePage />} />
        <Route path="/insights/whatif/:predictionId/:rowIndex" element={<WhatIfPage />} />
        <Route path="/insights/explain" element={<ExplainabilityPage />} />
        <Route path="/performance" element={<PerformancePage />} />
        <Route path="/about" element={<AboutPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppLayout>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <ScrollToTop />
      <ThemeProvider>
        <WorkspaceProvider>
          <Routed />
        </WorkspaceProvider>
      </ThemeProvider>
    </BrowserRouter>
  )
}
