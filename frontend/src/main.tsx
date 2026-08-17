import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'sonner';
import './styles/theme.css';
import { App } from './app';

const host = document.getElementById('app');
if (host === null) throw new Error('#app missing');

createRoot(host).render(
  <StrictMode>
    <QueryClientProvider client={new QueryClient()}>
      <App />
      <Toaster theme="dark" position="top-right" />
    </QueryClientProvider>
  </StrictMode>,
);
