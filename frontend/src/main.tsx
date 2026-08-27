import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'sonner';
import './styles/theme.css';
import { App } from './app';
import { TOKENS } from './kit/tokens';

const host = document.getElementById('app');
if (host === null) throw new Error('#app missing');

for (const [slug, hex] of Object.entries(TOKENS.accents)) {
  host.ownerDocument.documentElement.style.setProperty(
    `--color-env-${slug}`,
    `#${hex.toString(16).padStart(6, '0')}`,
  );
}

createRoot(host).render(
  <StrictMode>
    <QueryClientProvider client={new QueryClient()}>
      <App />
      <Toaster
        theme="dark"
        position="top-center"
        toastOptions={{ className: 'glass', style: { color: 'var(--color-ink)' } }}
      />
    </QueryClientProvider>
  </StrictMode>,
);
