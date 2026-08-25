import * as THREE from 'three';
import { makeFigure, type FigureBag } from '@kit/figure';
import type { Palette } from '@kit/tokens';
import {
  HP_FOOT_LEN,
  HP_LEG_LEN,
  HP_THIGH_LEN,
  HP_TORSO_HALF,
} from '@sim/envs/hopper';

export interface HopperHandle {
  group: THREE.Group;
  root: THREE.Object3D;
  setState(x: number, z: number, torso: number, thigh: number, leg: number, foot: number): void;
  setGlow(k: number): void;
  setTime(t: number): void;
  dispose(): void;
}

export interface HopperOptions {
  palette: Palette;
  accent: number;
}

export function buildHopper(opts: HopperOptions): HopperHandle {
  const bag: FigureBag = makeFigure({ palette: opts.palette, accent: opts.accent, name: 'meadowHopper' });

  const root = bag.joint(bag.group, 'hopRoot');

  const torso = bag.limb(root, HP_TORSO_HALF * 2, 0.09, 'hopTorso');
  torso.position.y = HP_TORSO_HALF;
  const head = bag.orb(root, 0.11, 'hopHead');
  head.position.y = HP_TORSO_HALF + 0.06;

  const thighJoint = bag.joint(root, 'hopThighJoint', 0, -HP_TORSO_HALF, 0);
  bag.limb(thighJoint, HP_THIGH_LEN, 0.065, 'hopThigh', true);

  const legJoint = bag.joint(thighJoint, 'hopLegJoint', 0, -HP_THIGH_LEN, 0);
  bag.limb(legJoint, HP_LEG_LEN, 0.05, 'hopLeg', true);

  const footJoint = bag.joint(legJoint, 'hopFootJoint', 0, -HP_LEG_LEN, 0);
  const foot = bag.rod(footJoint, HP_FOOT_LEN, 0.055, 'hopFoot');
  foot.position.x = -HP_FOOT_LEN / 3;

  return {
    group: bag.group,
    root,

    setState(x, z, torsoTh, thigh, leg, foot) {
      root.position.set(x, z, 0);
      root.rotation.z = -torsoTh;
      thighJoint.rotation.z = -thigh;
      legJoint.rotation.z = -leg;
      footJoint.rotation.z = -foot;
    },

    setGlow(k) {
      bag.group.scale.setScalar(1 + 0.02 * k);
    },

    setTime(_t) {},

    dispose() {
      bag.dispose();
    },
  };
}
