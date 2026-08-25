import * as THREE from 'three';
import { makeFigure, type FigureBag } from '@kit/figure';
import type { Palette } from '@kit/tokens';
import {
  WK_FOOT_LEN,
  WK_LEG_LEN,
  WK_THIGH_LEN,
  WK_TORSO_HALF,
} from '@sim/envs/walker2d';

export interface WalkerLeg {
  thighJoint: THREE.Object3D;
  legJoint: THREE.Object3D;
  footJoint: THREE.Object3D;
}

export interface WalkerHandle {
  group: THREE.Group;
  root: THREE.Object3D;
  setState(x: number, z: number, torso: number, right: readonly [number, number, number], left: readonly [number, number, number]): void;
  setGlow(k: number): void;
  setTime(t: number): void;
  dispose(): void;
}

export interface WalkerOptions {
  palette: Palette;
  accent: number;
}

function buildLeg(bag: FigureBag, root: THREE.Object3D, side: string, z: number, shaded: boolean): WalkerLeg {
  const thighJoint = bag.joint(root, `walk${side}ThighJoint`, 0, -WK_TORSO_HALF, z);
  bag.limb(thighJoint, WK_THIGH_LEN, 0.06, `walk${side}Thigh`, shaded);
  const legJoint = bag.joint(thighJoint, `walk${side}LegJoint`, 0, -WK_THIGH_LEN, 0);
  bag.limb(legJoint, WK_LEG_LEN, 0.048, `walk${side}Leg`, shaded);
  const footJoint = bag.joint(legJoint, `walk${side}FootJoint`, 0, -WK_LEG_LEN, 0);
  const foot = bag.rod(footJoint, WK_FOOT_LEN, 0.05, `walk${side}Foot`, shaded);
  foot.position.x = -WK_FOOT_LEN / 4;
  return { thighJoint, legJoint, footJoint };
}

export function buildWalker(opts: WalkerOptions): WalkerHandle {
  const bag: FigureBag = makeFigure({ palette: opts.palette, accent: opts.accent, name: 'flatsWalker' });

  const root = bag.joint(bag.group, 'walkRoot');

  const torso = bag.limb(root, WK_TORSO_HALF * 2, 0.09, 'walkTorso');
  torso.position.y = WK_TORSO_HALF;
  const head = bag.orb(root, 0.1, 'walkHead');
  head.position.y = WK_TORSO_HALF + 0.08;

  const rightLeg = buildLeg(bag, root, 'Right', 0.07, false);
  const leftLeg = buildLeg(bag, root, 'Left', -0.07, true);

  const pose = (leg: WalkerLeg, joints: readonly [number, number, number]): void => {
    leg.thighJoint.rotation.z = -joints[0];
    leg.legJoint.rotation.z = -joints[1];
    leg.footJoint.rotation.z = -joints[2];
  };

  return {
    group: bag.group,
    root,

    setState(x, z, torsoTh, right, left) {
      root.position.set(x, z, 0);
      root.rotation.z = -torsoTh;
      pose(rightLeg, right);
      pose(leftLeg, left);
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
