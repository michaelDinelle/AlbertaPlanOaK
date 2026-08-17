import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import './styles/theme.css';
import { App } from './app';
import { TooltipProvider } from './components/ui/tooltip';
import { adoptFromUrl } from './state/selectionStore';
import { urlState } from './url/useUrlState';

adoptFromUrl(urlState());

const host = document.getElementById('app');
if (host === null) throw new Error('#app missing');

const queryClient = new QueryClient();

createRoot(host).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <App />
      </TooltipProvider>
    </QueryClientProvider>
  </StrictMode>,
);
