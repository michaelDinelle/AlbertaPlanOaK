import * as THREE from 'three';
import { makeFigure, type FigureBag } from '@kit/figure';
import type { Palette } from '@kit/tokens';
import { SW_LINK_LEN } from '@sim/envs/swimmer';
import { SWIM_Y } from './world';

export interface SwimmerHandle {
  group: THREE.Group;
  root: THREE.Object3D;
  setState(x: number, y: number, yaw: number, j1: number, j2: number): void;
  setGlow(k: number): void;
  setTime(t: number): void;
  dispose(): void;
}

export interface SwimmerOptions {
  palette: Palette;
  accent: number;
}

export function buildSwimmer(opts: SwimmerOptions): SwimmerHandle {
  const bag: FigureBag = makeFigure({ palette: opts.palette, accent: opts.accent, name: 'lagoonSwimmer' });

  const root = bag.joint(bag.group, 'swimRoot', 0, SWIM_Y, 0);
  const heading = bag.joint(root, 'heading');

  bag.orb(heading, 0.16, 'swimHead');
  bag.rod(heading, SW_LINK_LEN, 0.09, 'swimBody');

  const joint1 = bag.joint(heading, 'swimJoint1', SW_LINK_LEN, 0, 0);
  bag.orb(joint1, 0.1, 'swimHip', true);
  bag.rod(joint1, SW_LINK_LEN, 0.07, 'swimMid', true);

  const joint2 = bag.joint(joint1, 'swimJoint2', SW_LINK_LEN, 0, 0);
  bag.orb(joint2, 0.08, 'swimKnee', true);
  bag.rod(joint2, SW_LINK_LEN, 0.05, 'swimTail', true);
  const fin = bag.box(joint2, 0.06, 0.02, 0.34, 'swimFin', true);
  fin.position.set(SW_LINK_LEN, 0, 0);

  return {
    group: bag.group,
    root,

    setState(x, y, yaw, j1, j2) {
      root.position.set(x, SWIM_Y, -y);
      heading.rotation.y = yaw + Math.PI;
      joint1.rotation.y = j1;
      joint2.rotation.y = j2;
    },

    setGlow(k) {
      bag.group.scale.setScalar(1 + 0.02 * k);
    },

    setTime(t) {
      root.position.y = SWIM_Y + 0.05 * Math.sin(t * 1.3);
    },

    dispose() {
      bag.dispose();
    },
  };
}
