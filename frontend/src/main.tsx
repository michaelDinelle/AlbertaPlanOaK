import { createRoot, type Root } from 'react-dom/client';
import { App } from './app';
import { renderPosters } from './posters';
import { TOKENS } from './kit/tokens';

for (const [slug, hex] of Object.entries(TOKENS.accents)) {
  document.documentElement.style.setProperty(`--color-env-${slug}`, `#${hex.toString(16).padStart(6, '0')}`);
}

const mounted = globalThis as { oakRoot?: Root };
if (new URLSearchParams(location.search).has('posters')) void renderPosters();
else (mounted.oakRoot ??= createRoot(document.getElementById('app')!)).render(<App />);
