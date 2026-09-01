import { clampRange } from '@kit/mix';
import { TOKENS } from '@kit/tokens';
import { AGENT_SPAN } from './actors';
import { SCATTER_SPAN, SCATTER_TILE, SHADOW_MAP_X, SHADOW_SPAN } from './world';

const SCATTER_HALF = (SCATTER_SPAN * SCATTER_TILE) / 2;
const SHADOW_TEXEL_FLOOR = AGENT_SPAN / 8;
const SHADOW_SPAN_MAX = (SHADOW_MAP_X * SHADOW_TEXEL_FLOOR) / 2;

export interface OrchardView {
  maxDistance: number;
  far: number;
  shadowSpan(distance: number): number;
}

export function makeOrchardView(widthPx: number, heightPx: number): OrchardView {
  const tanV = Math.tan((TOKENS.fovHero * Math.PI) / 360);
  const aspect = widthPx > 0 && heightPx > 0 ? widthPx / heightPx : 16 / 9;
  const tanH = tanV * aspect;
  const tanD = Math.hypot(tanH, tanV);
  const maxDistance = SCATTER_HALF / tanD;
  const far = maxDistance * Math.hypot(1, tanD) * 4;
  return {
    maxDistance,
    far,
    shadowSpan(distance) {
      return clampRange(distance * tanH * 1.1, SHADOW_SPAN, SHADOW_SPAN_MAX);
    },
  };
}
