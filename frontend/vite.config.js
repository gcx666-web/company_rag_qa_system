import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// Vite 配置：
// - 开发服务器把 /api 代理到 FastAPI（127.0.0.1:8000），避免跨域；
// - 构建产物输出到 dist/，后端 main.py 会直接托管该目录。
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    host: '127.0.0.1',
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  preview: {
    port: 4173,
    host: '127.0.0.1',
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: false,
  },
})
