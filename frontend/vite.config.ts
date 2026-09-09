import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';
import react, { reactCompilerPreset } from '@vitejs/plugin-react';
import babel from '@rolldown/plugin-babel';
import tailwindcss from '@tailwindcss/vite';

const r = (p: string) => fileURLToPath(new URL(p, import.meta.url));

export default defineConfig({
  plugins: [
    react(),
    babel({ include: /frontend\/src\/.*\.tsx(?:$|\?)/, presets: [reactCompilerPreset()] }),
    tailwindcss(),
  ],
  server: {
    port: 5173,
    proxy: { '/api': { target: `http://127.0.0.1:${process.env['OAK_PORT'] ?? '5174'}`, ws: true } },
  },
  build: {
    target: 'es2022',
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          if (id.includes('node_modules/three')) return 'three';
        },
      },
    },
  },
  resolve: {
    alias: {
      '@sim': r('./src/sim'),
      '@kit': r('./src/kit'),
    },
  },
});
