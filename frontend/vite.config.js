import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

console.log("Vite config loaded")
// https://vite.dev/config/
export default defineConfig({
  base: '/rag/',
  plugins: [react()],
})
