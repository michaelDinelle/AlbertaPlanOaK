import * as THREE from 'three/webgpu';

export const CARD = new THREE.Vector2(1, 1.4);
export const SCALE = 1.4;
export const RECORD = 0.72;
export const CHIP = 0.96;

const GAP = 0.3;
const SPAN = 14.2;
const SHOE_GAP = 0.55;
const RECORD_PITCH = 0.45;
const MARGIN = 1.15;
const UI = 1.7;

export interface Layout {
  lag: number;
  places: number;
  pitch: number;
  y: number;
  due: number;
  shoe: THREE.Vector2;
  shoeSize: THREE.Vector2;
  lamp: THREE.Vector2;
  tray: THREE.Vector2;
  traySize: THREE.Vector2;
  chip: THREE.Vector2;
  record: { x0: number; pitch: number; count: number; y: number };
  table: THREE.Box2;
  extent: THREE.Box2;
}

export function layoutFor(lag: number): Layout {
  const places = lag + 1, width = CARD.x * SCALE, height = CARD.y * SCALE;
  const pitch = Math.min(width + GAP, (SPAN - width) / Math.max(1, places - 1));
  const shoeSize = new THREE.Vector2(1.7, 2.3);
  const right = ((places - 1) * pitch + width + SHOE_GAP + shoeSize.x) / 2, left = -right;
  const due = left + width / 2, y = -1.9;
  const shoe = new THREE.Vector2(right - shoeSize.x / 2, y);
  const lamp = new THREE.Vector2(due + (places - 1) * pitch, y - height / 2 - 1.05);
  const traySize = new THREE.Vector2(width + CHIP + 0.8, height + 0.5), tray = new THREE.Vector2(due - width / 2 - 0.3 + traySize.x / 2, 1.75);
  const chip = new THREE.Vector2(due + width / 2 + 0.22 + CHIP / 2, tray.y + 0.28);
  const recordWidth = width * RECORD, x0 = tray.x + traySize.x / 2 + 0.4 + recordWidth / 2;
  const record = { x0, pitch: RECORD_PITCH, count: Math.floor((right - recordWidth / 2 - x0) / RECORD_PITCH) + 1, y: tray.y };
  const table = new THREE.Box2(new THREE.Vector2(left - MARGIN, lamp.y - 1.25), new THREE.Vector2(right + MARGIN, tray.y + traySize.y / 2 + 0.75));
  const extent = new THREE.Box2(table.min.clone().subScalar(0.35), table.max.clone().add(new THREE.Vector2(0.35, UI)));
  return { lag, places, pitch, y, due, shoe, shoeSize, lamp, tray, traySize, chip, record, table, extent };
}

export const placeX = (layout: Layout, slot: number): number => layout.due + slot * layout.pitch;

export const recordX = (layout: Layout, index: number): number => layout.record.x0 + index * layout.record.pitch;
