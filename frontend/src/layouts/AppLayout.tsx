// The frame every page renders inside.
//
// Green glass shell:
// - Collapsible sidebar (icons + labels / icons only, tooltip when collapsed),
//   compact on tablet, slide-in drawer on mobile
// - Dark glass top bar: dataset, model, stage, workflow rail, theme toggle
// - localStorage-persisted sidebar state
// - Keyboard-accessible navigation with aria labels

import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { NavLink, useLocation } from 'react-router-dom'
import type { ReactNode } from 'react'

import {
  LayoutDashboard,
  BarChart3,
  Brain,
  Sparkles,
  Trophy,
  Check,
  TriangleAlert,
  Lock,
  Layers,
  Search,
  Lightbulb,
  GitCompare,
  Activity,
  Info,
  Upload,
  Sun,
  Moon,
  PanelLeft,
  PanelLeftClose,
  Menu,
  X,
  Home,
  Database,
  ChevronRight,
} from 'lucide-react'

import { useTheme } from '../hooks/useTheme'
import { useWorkspace } from '../hooks/useWorkspace'
import type { Stage } from '../types/api'

const NAV_ITEMS = [
  { section: 'Workspace', items: [
    { to: '/', label: 'Home', requires: null as 'dataset' | 'model' | null, icon: Home },
    { to: '/dashboard', label: 'Dashboard', requires: 'dataset' as const, icon: LayoutDashboard },
    { to: '/upload', label: 'Upload Dataset', requires: null as 'dataset' | 'model' | null, icon: Upload },
    { to: '/analysis', label: 'Data Analysis', requires: 'dataset' as const, icon: BarChart3 },
    { to: '/train', label: 'Train Model', requires: 'dataset' as const, icon: Brain },
    { to: '/predict', label: 'Prediction', requires: 'model' as const, icon: Sparkles },
    { to: '/predictions/history', label: 'Results', requires: 'dataset' as const, icon: Trophy },
  ]},
  { section: 'Insights', items: [
    { to: '/insights/at-risk', label: 'At-Risk Students', requires: 'dataset' as const, icon: TriangleAlert },
    { to: '/insights/explorer', label: 'Student Explorer', requires: 'dataset' as const, icon: Search },
    { to: '/insights/explain', label: 'Explainability', requires: 'model' as const, icon: Lightbulb },
  ]},
  { section: 'Models', items: [
    { to: '/models', label: 'Model Comparison', requires: 'dataset' as const, icon: GitCompare },
  ]},
  { section: 'System', items: [
    { to: '/performance', label: 'Performance', requires: null as 'dataset' | 'model' | null, icon: Activity },
    { to: '/about', label: 'About', requires: null as 'dataset' | 'model' | null, icon: Info },
  ]},
]

const STAGES: { key: Stage; label: string; to: string }[] = [
  { key: 'upload', label: 'Upload', to: '/upload' },
  { key: 'analyze', label: 'Analyse', to: '/analysis' },
  { key: 'train', label: 'Train', to: '/train' },
  { key: 'predict', label: 'Predict', to: '/predict' },
  { key: 'export', label: 'Export', to: '/predict' },
]

function ThemeToggle() {
  const { theme, toggle } = useTheme()
  return (
    <button
      type="button"
      onClick={toggle}
      className="relative w-8 h-8 grid place-items-center rounded-xl border border-line text-ink overflow-hidden
                 transition-[transform,box-shadow,border-color,color] duration-200 hover:border-brand/60 hover:text-brand-mid hover:shadow-glow-brand active:scale-95"
      aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} theme`}
      title={`Switch to ${theme === 'light' ? 'dark' : 'light'} theme`}
    >
      <span className="grid place-items-center transition-transform duration-300" style={{ transform: theme === 'light' ? 'rotate(0deg)' : 'rotate(90deg)' }}>
        {theme === 'light' ? <Moon size={15} /> : <Sun size={15} />}
      </span>
    </button>
  )
}

interface Tip { label: string; top: number; left: number }

/** Tooltip for the collapsed sidebar. Portalled to <body> because the sidebar
 *  clips overflow and its backdrop-filter would trap a fixed child. */
function CollapsedTip({ tip }: { tip: Tip | null }) {
  if (!tip) return null
  return createPortal(
    <div
      role="tooltip"
      className="nav-tip animate-fade-in"
      style={{ top: tip.top, left: tip.left }}
    >
      {tip.label}
    </div>,
    document.body,
  )
}

function NavContent({ onNavigate, collapsed }: { onNavigate?: () => void; collapsed?: boolean }) {
  const { dataset, model } = useWorkspace()
  const [tip, setTip] = useState<Tip | null>(null)

  const showTip = (label: string, el: HTMLElement) => {
    if (!collapsed) return
    const r = el.getBoundingClientRect()
    setTip({ label, top: r.top + r.height / 2, left: r.right + 12 })
  }
  const hideTip = () => setTip(null)

  useEffect(() => {
    if (!collapsed) setTip(null)
  }, [collapsed])

  return (
    <nav className={`flex-1 px-2.5 py-2 ${collapsed ? 'space-y-1' : 'space-y-4'} overflow-y-auto overflow-x-hidden`} aria-label="Main">
      <CollapsedTip tip={tip} />
      {NAV_ITEMS.map((group, gIdx) => (
        <div key={group.section}>
          {group.section && !collapsed && (
            <p className="px-3 mb-1.5 text-[11px] font-medium text-muted/60">{group.section}</p>
          )}
          {group.section && collapsed && gIdx > 0 && (
            <div className="h-px bg-line my-2 mx-2" aria-hidden />
          )}
          <div className="space-y-0.5">
            {group.items.map((item) => {
              const locked = (item.requires === 'dataset' && !dataset) || (item.requires === 'model' && !model)
              const Icon = item.icon
              return (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.to === '/'}
                  onClick={onNavigate}
                  aria-label={locked ? `${item.label} (locked)` : item.label}
                  onMouseEnter={(e) => showTip(item.label, e.currentTarget)}
                  onFocus={(e) => showTip(item.label, e.currentTarget)}
                  onMouseLeave={hideTip}
                  onBlur={hideTip}
                  className={({ isActive }) =>
                    `group relative flex items-center ${
                      collapsed
                        ? 'justify-center w-10 h-10 mx-auto rounded-xl'
                        : 'gap-3 h-9 pl-3 pr-2 rounded-xl text-[13px] font-medium'
                    } transition-all duration-200
                     ${isActive
                       ? 'bg-brand/10 text-brand-deep font-semibold border border-brand/25'
                       : 'text-ink/70 border border-transparent hover:text-ink hover:bg-ink/[0.05]'}`
                  }
                >
                  {({ isActive }) => (
                    <>
                      {isActive && (
                        <span
                          className={`absolute ${collapsed ? '-left-1.5' : '-left-2.5'} top-1/2 -translate-y-1/2 h-4 w-[3px] rounded-full bg-brand shadow-[0_0_8px_rgb(var(--color-brand)/0.45)]`}
                          aria-hidden
                        />
                      )}
                      <Icon
                        size={collapsed ? 18 : 16}
                        strokeWidth={isActive ? 2.1 : 1.8}
                        className={`shrink-0 transition-colors duration-200 ${
                          isActive ? 'text-brand-mid' : 'text-muted group-hover:text-ink'
                        }`}
                      />
                      {!collapsed && <span className="truncate">{item.label}</span>}
                      {!collapsed && locked && (
                        <Lock size={11} className="ml-auto text-muted/50 shrink-0" aria-hidden />
                      )}
                      {collapsed && locked && (
                        <span className="absolute bottom-1 right-1 w-1.5 h-1.5 rounded-full bg-muted/40" aria-hidden />
                      )}
                    </>
                  )}
                </NavLink>
              )
            })}
          </div>
        </div>
      ))}
    </nav>
  )
}

/** Green-rimmed mark with the wordmark; "AI" in green. */
function BrandMark({ size = 'md' }: { size?: 'sm' | 'md' }) {
  const box = size === 'sm' ? 'w-8 h-8 text-[10px]' : 'w-9 h-9 text-[11px]'
  return (
    <div
      className={`${box} rounded-xl grid place-items-center font-display font-extrabold tracking-tight text-ink shrink-0
                  border border-brand/50 shadow-[0_0_16px_-6px_rgb(var(--color-brand)/0.6)]`}
      style={{ background: 'linear-gradient(145deg, rgb(var(--color-surface)), rgb(var(--color-canvas)))' }}
      aria-hidden
    >
      <span>E<span className="text-brand-mid">P</span></span>
    </div>
  )
}

function BrandText() {
  return (
    <div className="min-w-0">
      <p className="font-display text-[14px] font-bold tracking-[-0.01em] text-ink truncate">
        EduPredict <span className="text-brand-mid font-semibold">AI</span>
      </p>
      <p className="text-[10.5px] text-muted/70 truncate">Student Performance</p>
    </div>
  )
}

function SidebarBrand({ collapsed, onToggle, lock }: { collapsed: boolean; onToggle: () => void; lock: boolean }) {
  if (collapsed) {
    return (
      <div className="py-3 border-b border-line flex flex-col items-center gap-2">
        <BrandMark />
        {!lock && (
          <button
            type="button"
            onClick={onToggle}
            className="w-7 h-7 grid place-items-center rounded-lg text-muted hover:text-brand-mid hover:bg-brand/10 transition-colors"
            title="Expand sidebar"
            aria-label="Expand sidebar"
            aria-expanded={false}
          >
            <PanelLeft size={15} />
          </button>
        )}
      </div>
    )
  }

  return (
    <div className="px-4 py-3 border-b border-line flex items-center justify-between gap-2">
      <div className="flex items-center gap-2.5 min-w-0">
        <BrandMark />
        <BrandText />
      </div>
      <button
        type="button"
        onClick={onToggle}
        className="w-7 h-7 grid place-items-center rounded-lg text-muted hover:text-brand-mid hover:bg-brand/10 transition-colors shrink-0"
        title="Collapse sidebar"
        aria-label="Collapse sidebar"
        aria-expanded
      >
        <PanelLeftClose size={15} />
      </button>
    </div>
  )
}

function Sidebar({ collapsed, onToggle, lock }: { collapsed: boolean; onToggle: () => void; lock: boolean }) {
  return (
    <aside
      aria-label="Sidebar"
      className={`hidden md:flex md:flex-col ${
        collapsed ? 'w-[68px]' : 'w-[244px]'
      } shrink-0 m-3 mr-0 panel overflow-hidden sticky top-3 self-start transition-[width] duration-300 ease-[cubic-bezier(0.4,0,0.2,1)]`}
      style={{ height: 'calc(100vh - 24px)' }}
    >
      <SidebarBrand collapsed={collapsed} onToggle={onToggle} lock={lock} />
      <NavContent collapsed={collapsed} />
    </aside>
  )
}

function MobileSidebar({ open, onClose }: { open: boolean; onClose: () => void }) {
  const location = useLocation()

  useEffect(() => {
    onClose()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.pathname])

  useEffect(() => {
    if (!open) return
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [open, onClose])

  if (!open) return null

  return (
    <div className="md:hidden fixed inset-0 z-40" role="dialog" aria-modal="true" aria-label="Navigation">
      <div className="absolute inset-0 bg-ink/40 backdrop-blur-sm animate-fade-in" onClick={onClose} aria-hidden />
      <aside className="absolute inset-y-3 left-3 w-[80vw] max-w-[290px] panel overflow-hidden flex flex-col animate-slide-in">
        <div className="px-4 py-3.5 border-b border-line flex items-center justify-between">
          <div className="flex items-center gap-2.5 min-w-0">
            <BrandMark size="sm" />
            <BrandText />
          </div>
          <button
            type="button"
            onClick={onClose}
            className="w-7 h-7 grid place-items-center rounded-lg text-muted hover:text-brand-mid hover:bg-brand/10"
            aria-label="Close navigation"
          >
            <X size={16} />
          </button>
        </div>
        <NavContent onNavigate={onClose} />
      </aside>
    </div>
  )
}

function formatDims(rows: number, cols: number): string {
  const r = rows >= 1000 ? `${Math.round(rows / 1000)}K` : String(rows)
  return `${r} × ${cols}`
}

function StatusItem({ icon: Icon, label, value, sub, active }: {
  icon: typeof Database
  label: string
  value: string
  sub?: string
  active: boolean
}) {
  return (
    <div className="flex items-center gap-2 min-w-0">
      <span
        className={`w-7 h-7 rounded-lg grid place-items-center shrink-0 border ${
          active ? 'border-brand/35 bg-brand/10 text-brand-mid' : 'border-line text-muted/50'
        }`}
      >
        <Icon size={13} aria-hidden />
      </span>
      <div className="min-w-0 leading-tight">
        <p className="text-[10.5px] text-muted/70">{label}</p>
        <p className="text-[12px] font-semibold text-ink truncate max-w-[18ch]" title={value}>
          {value}
          {sub && <span className="ml-1.5 font-medium text-muted tabular-nums">{sub}</span>}
        </p>
      </div>
    </div>
  )
}

function StatusStrip() {
  const { dataset, model, stage } = useWorkspace()
  const stageLabel = STAGES.find((s) => s.key === stage)?.label ?? '—'

  return (
    <div className="flex items-center gap-5 min-w-0" aria-label="Workspace status">
      <StatusItem
        icon={Database}
        label="Dataset"
        value={dataset ? dataset.filename : 'None'}
        sub={dataset ? formatDims(dataset.num_rows, dataset.num_cols) : undefined}
        active={Boolean(dataset)}
      />
      <div className="hidden sm:block">
        <StatusItem icon={Brain} label="Model" value={model ? model.model_choice : 'None'} active={Boolean(model)} />
      </div>
      <div className="hidden md:block">
        <StatusItem icon={Layers} label="Stage" value={stageLabel} active />
      </div>
    </div>
  )
}

function WorkflowRail() {
  const { stage, dataset, model } = useWorkspace()
  const currentIndex = STAGES.findIndex((s) => s.key === stage)
  const location = useLocation()

  return (
    <ol className="hidden xl:flex items-center gap-1.5 overflow-x-auto" aria-label="Workflow progress">
      {STAGES.map((item, index) => {
        const done = index < currentIndex
        const active = index === currentIndex
        const reachable = index === 0 || (index <= 2 && Boolean(dataset)) || (index >= 3 && Boolean(model))
        const onPage = location.pathname === item.to
        const emphasised = active || onPage

        return (
          <li key={item.key} className="flex items-center gap-1.5 shrink-0">
            <NavLink
              to={reachable ? item.to : '/upload'}
              aria-current={active ? 'step' : undefined}
              className={`workflow-step ${
                emphasised
                  ? 'workflow-step-active'
                  : done
                    ? 'workflow-step-done'
                    : 'workflow-step-pending'
              }`}
            >
              <span
                className={`w-4 h-4 rounded-full grid place-items-center text-[9px] font-bold transition-colors duration-200
                  ${emphasised ? 'bg-brand text-onbrand' : done ? 'bg-brand/80 text-canvas' : 'bg-line text-muted'}`}
              >
                {done ? <Check size={10} strokeWidth={3} aria-hidden /> : index + 1}
              </span>
              {item.label}
            </NavLink>
            {index < STAGES.length - 1 && (
              <ChevronRight size={12} className={`transition-colors duration-300 ${done ? 'text-brand/50' : 'text-line'}`} />
            )}
          </li>
        )
      })}
    </ol>
  )
}

export function AppLayout({ children }: { children: ReactNode }) {
  const [mobileOpen, setMobileOpen] = useState(false)
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return localStorage.getItem('edupredict_sidebar_collapsed') === 'true'
    } catch {
      return false
    }
  })
  const location = useLocation()

  // Tablet widths (768-1023px) always get the compact icon rail.
  const [tablet, setTablet] = useState(
    () => typeof window !== 'undefined' && window.matchMedia('(min-width: 768px) and (max-width: 1023px)').matches,
  )
  useEffect(() => {
    const mq = window.matchMedia('(min-width: 768px) and (max-width: 1023px)')
    const onChange = () => setTablet(mq.matches)
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [])

  const toggleCollapsed = () => {
    setCollapsed((prev) => {
      const next = !prev
      try {
        localStorage.setItem('edupredict_sidebar_collapsed', String(next))
      } catch {
        // ignore localStorage access issues
      }
      return next
    })
  }

  // Keyboard shortcut: Ctrl+B to toggle sidebar
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'b') {
        e.preventDefault()
        toggleCollapsed()
      }
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <div className="min-h-screen flex">
      <Sidebar collapsed={collapsed || tablet} onToggle={toggleCollapsed} lock={tablet} />
      <MobileSidebar open={mobileOpen} onClose={() => setMobileOpen(false)} />

      <div className="flex-1 min-w-0 flex flex-col">
        <header className="sticky top-3 z-20 mx-3 mt-3 panel">
          <div className="px-4 lg:px-5 py-2.5 flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2 md:hidden">
              <button
                type="button"
                onClick={() => setMobileOpen(true)}
                className="w-8 h-8 grid place-items-center rounded-xl border border-line text-ink hover:border-brand/60 hover:text-brand-mid transition-colors"
                aria-label="Open navigation"
                aria-expanded={mobileOpen}
              >
                <Menu size={16} />
              </button>
              <p className="font-display text-[14px] font-bold text-ink">EduPredict <span className="text-brand-mid">AI</span></p>
            </div>
            <StatusStrip />
            <div className="flex items-center gap-1.5 ml-auto">
              <WorkflowRail />
              <div className="hidden xl:block w-px h-5 bg-line mx-1" />
              <ThemeToggle />
            </div>
          </div>
        </header>

        <main className="flex-1 px-3 lg:px-5 py-5">
          <div key={location.pathname} className="max-w-[1440px] mx-auto route-transition">
            {children}
          </div>
        </main>
      </div>
    </div>
  )
}
