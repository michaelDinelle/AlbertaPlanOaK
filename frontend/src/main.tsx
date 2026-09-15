import { createRoot, type Root } from 'react-dom/client';
import { App } from './app';
import { TOKENS } from './kit/tokens';

for (const [slug, hex] of Object.entries(TOKENS.accents)) {
  document.documentElement.style.setProperty(`--color-env-${slug}`, `#${hex.toString(16).padStart(6, '0')}`);
}

const mounted = globalThis as { oakRoot?: Root };
mounted.oakRoot ??= createRoot(document.getElementById('app')!);
mounted.oakRoot.render(<App />);
