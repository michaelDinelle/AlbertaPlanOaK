const SEEN_KEY = 'oak.about.seen';

export function aboutSeen(): boolean {
  try {
    return window.localStorage.getItem(SEEN_KEY) !== null;
  } catch {
    return true;
  }
}

export function markAboutSeen(): void {
  try {
    window.localStorage.setItem(SEEN_KEY, '1');
  } catch {
    return;
  }
}
