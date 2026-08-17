import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'sonner';
import './styles/theme.css';
import { App } from './app';
import { adoptFromUrl } from './state/selectionStore';
import { urlState } from './url/useUrlState';

adoptFromUrl(urlState());

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
