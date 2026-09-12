import type { EnvId } from '../data/envs';

export type KeyAction =
  | { kind: 'toggle' }
  | { kind: 'step'; frames: number }
  | { kind: 'env'; direction: 1 | -1 }
  | { kind: 'sort' }
  | { kind: 'debug' }
  | { kind: 'fullscreen' }
  | { kind: 'policy' }
  | { kind: 'panel' }
  | { kind: 'view'; index: number }
  | { kind: 'escape' };

export function keyAction(key: string, shift = false): KeyAction | null {
  switch (key) {
    case ' ':
    case 'Spacebar':
    case 'k':
    case 'K':
      return { kind: 'toggle' };
    case 'ArrowRight':
      return { kind: 'step', frames: shift ? 200 : 20 };
    case 'ArrowLeft':
      return { kind: 'step', frames: shift ? -200 : -20 };
    case '[':
    case '{':
      return { kind: 'env', direction: -1 };
    case ']':
    case '}':
      return { kind: 'env', direction: 1 };
    case 's':
    case 'S':
      return { kind: 'sort' };
    case 'g':
    case 'G':
      return { kind: 'debug' };
    case 'f':
    case 'F':
      return { kind: 'fullscreen' };
    case 'p':
    case 'P':
      return { kind: 'policy' };
    case 'c':
    case 'C':
      return { kind: 'panel' };
    case '1':
    case '2':
    case '3':
    case '4':
      return { kind: 'view', index: Number(key) - 1 };
    case 'Escape':
      return { kind: 'escape' };
    default:
      return null;
  }
}

export function cycleEnv(list: readonly EnvId[], current: EnvId | null, direction: 1 | -1): EnvId | null {
  if (list.length === 0) return null;
  const index = current === null ? -1 : list.indexOf(current);
  const next = index < 0 ? (direction > 0 ? 0 : list.length - 1) : (index + direction + list.length) % list.length;
  return list[next] ?? null;
}
