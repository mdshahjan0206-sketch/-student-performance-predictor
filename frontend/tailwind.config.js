/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: 'rgb(var(--color-ink) / <alpha-value>)',
        canvas: 'rgb(var(--color-canvas) / <alpha-value>)',
        surface: 'rgb(var(--color-surface) / <alpha-value>)',
        panel: 'rgb(var(--color-panel) / <alpha-value>)',
        line: 'rgb(var(--color-line) / <alpha-value>)',
        muted: 'rgb(var(--color-muted) / <alpha-value>)',
        ok: {
          DEFAULT: 'rgb(var(--color-ok) / <alpha-value>)',
          soft: 'rgb(var(--color-ok-soft) / <alpha-value>)',
        },
        elevated: 'rgb(var(--color-elevated) / <alpha-value>)',
        faint: 'rgb(var(--color-faint) / <alpha-value>)',
        onbrand: 'rgb(var(--color-on-brand) / <alpha-value>)',
        brand: {
          DEFAULT: 'rgb(var(--color-brand) / <alpha-value>)',
          soft: 'rgb(var(--color-brand-soft) / <alpha-value>)',
          mid: 'rgb(var(--color-brand-mid) / <alpha-value>)',
          deep: 'rgb(var(--color-brand-deep) / <alpha-value>)',
        },
        amber: {
          DEFAULT: 'rgb(var(--color-amber) / <alpha-value>)',
          soft: 'rgb(var(--color-amber-soft) / <alpha-value>)',
        },
        danger: {
          DEFAULT: 'rgb(var(--color-danger) / <alpha-value>)',
          soft: 'rgb(var(--color-danger-soft) / <alpha-value>)',
        },
      },
      fontFamily: {
        display: ['"Plus Jakarta Sans"', 'system-ui', 'sans-serif'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      fontSize: {
        '2xs': ['0.6875rem', { lineHeight: '1rem' }],
      },
      boxShadow: {
        card: '0 1px 2px rgb(var(--shadow-rgb) / var(--shadow-a1)), 0 6px 20px -10px rgb(var(--shadow-rgb) / var(--shadow-a2))',
        lift: '0 2px 4px rgb(var(--shadow-rgb) / var(--shadow-a1)), 0 14px 34px -14px rgb(var(--shadow-rgb) / var(--shadow-a3))',
        glass: '0 1px 0 rgb(255 255 255 / var(--inset-hi)) inset, 0 10px 28px -14px rgb(var(--shadow-rgb) / var(--shadow-a2))',
        'glass-lift': '0 1px 0 rgb(255 255 255 / var(--inset-hi)) inset, 0 0 0 1px rgb(var(--color-brand) / 0.14), 0 16px 40px -16px rgb(var(--shadow-rgb) / var(--shadow-a3))',
        'glow-brand': '0 0 0 1px rgb(var(--color-brand) / 0.3), 0 6px 20px -8px rgb(var(--color-brand) / 0.35)',
      },
      borderRadius: {
        card: '16px',
        panel: '20px',
      },
      backdropBlur: {
        glass: '24px',
      },
      keyframes: {
        'fade-in': { from: { opacity: '0' }, to: { opacity: '1' } },
        'rise': {
          from: { opacity: '0', transform: 'translateY(6px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        'sweep': { '0%': { transform: 'translateX(-100%)' }, '100%': { transform: 'translateX(320%)' } },
        'slide-in': {
          from: { opacity: '0', transform: 'translateX(-24px)' },
          to: { opacity: '1', transform: 'translateX(0)' },
        },
        'scale-in': {
          from: { opacity: '0', transform: 'scale(0.98)' },
          to: { opacity: '1', transform: 'scale(1)' },
        },
      },
      animation: {
        'fade-in': 'fade-in 180ms ease-out',
        rise: 'rise 220ms cubic-bezier(.16,1,.3,1)',
        sweep: 'sweep 1.4s ease-in-out infinite',
        'slide-in': 'slide-in 240ms cubic-bezier(.16,1,.3,1)',
        'scale-in': 'scale-in 200ms cubic-bezier(.16,1,.3,1)',
      },
    },
  },
  plugins: [],
}
