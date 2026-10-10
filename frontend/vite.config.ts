import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The dev server proxies /api to the FastAPI backend, so the browser talks to
// a single origin and no CORS configuration is needed while developing.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.BACKEND_URL || 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    // plotly is large and only needed by the chart components, so it is split
    // into its own chunk that loads when a chart first renders.
    rollupOptions: {
      output: {
        manualChunks: {
          plotly: ['plotly.js-cartesian-dist-min'],
        },
      },
    },
  },
})
