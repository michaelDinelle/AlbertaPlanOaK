import { CELL, type GridSize } from '../grid';
import type { FrameUpdate } from '../types';

export interface CellWalk { x: number; z: number; yaw: number; moving: number; fromX: number; fromY: number; toX: number; toY: number }

export const cellWalk = (): CellWalk => ({ x: 0, z: 0, yaw: 0, moving: 0, fromX: 0, fromY: 0, toX: 0, toY: 0 });

export function walkCells(frame: FrameUpdate, slotX: number, slotY: number, grid: GridSize, out: CellWalk): CellWalk {
  const step = frame.state.step, pose = frame.state.pose;
  const toX = Math.round(pose[slotX]!), toY = Math.round(pose[slotY]!);
  let fromX = toX, fromY = toY;
  if (step > 0 && !frame.state.reset) {
    const previous = frame.poseAt(step - 1);
    fromX = Math.round(previous[slotX]!); fromY = Math.round(previous[slotY]!);
  }
  if (Math.abs(toX - fromX) > 1 || Math.abs(toY - fromY) > 1) { fromX = toX; fromY = toY; }
  const raw = Math.min(1, Math.max(0, frame.position - step)), t = raw * raw * (3 - 2 * raw);
  out.fromX = fromX; out.fromY = fromY; out.toX = toX; out.toY = toY;
  out.x = (fromX + (toX - fromX) * t - (grid.w - 1) / 2) * CELL;
  out.z = (fromY + (toY - fromY) * t - (grid.h - 1) / 2) * CELL;
  out.moving = toX !== fromX || toY !== fromY ? 1 : 0;
  if (out.moving) out.yaw = Math.atan2(toX - fromX, toY - fromY);
  return out;
}

export function gridHeading(dx: number, dy: number, previous: number): number {
  if (dx > 0) return 1;
  if (dx < 0) return 3;
  if (dy < 0) return 0;
  if (dy > 0) return 2;
  return previous;
}
