import * as THREE from 'three';
import { makeFigure, type FigureBag } from '@kit/figure';
import type { Palette } from '@kit/tokens';
import { ANT_LOWER_LEN, ANT_TORSO_R, ANT_UPPER_LEN } from '@sim/envs/ant';

const LEG_YAWS: readonly number[] = [Math.PI / 4, (3 * Math.PI) / 4, (-3 * Math.PI) / 4, -Math.PI / 4];
const UPPER_DROOP = -0.3;
const LOWER_DROOP = -0.9;

export interface AntLeg {
  hipJoint: THREE.Object3D;
  ankleJoint: THREE.Object3D;
}

export interface AntHandle {
  group: THREE.Group;
  root: THREE.Object3D;
  setState(px: number, py: number, pz: number, qw: number, qx: number, qy: number, qz: number, joints: ArrayLike<number>): void;
  setGlow(k: number): void;
  setTime(t: number): void;
  dispose(): void;
}

export interface AntOptions {
  palette: Palette;
  accent: number;
}

export function buildAnt(opts: AntOptions): AntHandle {
  const bag: FigureBag = makeFigure({ palette: opts.palette, accent: opts.accent, name: 'basinAnt' });

  const root = bag.joint(bag.group, 'antRoot');
  bag.orb(root, ANT_TORSO_R, 'antTorso');
  const eye = bag.orb(root, 0.07, 'antEye', true);
  eye.position.set(ANT_TORSO_R * 0.8, ANT_TORSO_R * 0.35, 0);

  const legs: AntLeg[] = LEG_YAWS.map((yaw, k) => {
    const legRoot = bag.joint(root, `antLegRoot${k}`);
    legRoot.rotation.y = yaw;
    const hipJoint = bag.joint(legRoot, `antHip${k}`, ANT_TORSO_R * 0.8, 0, 0);
    const upperPitch = bag.joint(hipJoint, `antUpperPitch${k}`);
    upperPitch.rotation.z = UPPER_DROOP;
    bag.rod(upperPitch, ANT_UPPER_LEN, 0.05, `antUpper${k}`, true);
    const ankleJoint = bag.joint(upperPitch, `antAnkle${k}`, ANT_UPPER_LEN, 0, 0);
    bag.rod(ankleJoint, ANT_LOWER_LEN, 0.04, `antLower${k}`);
    const toe = bag.orb(ankleJoint, 0.05, `antToe${k}`, true);
    toe.position.x = ANT_LOWER_LEN;
    return { hipJoint, ankleJoint };
  });

  return {
    group: bag.group,
    root,

    setState(px, py, pz, qw, qx, qy, qz, joints) {
      root.position.set(px, pz, -py);
      root.quaternion.set(qx, qz, -qy, qw);
      for (let k = 0; k < legs.length; k++) {
        const leg = legs[k];
        if (!leg) continue;
        leg.hipJoint.rotation.y = -(joints[k * 2] ?? 0);
        leg.ankleJoint.rotation.z = LOWER_DROOP + (joints[k * 2 + 1] ?? 0);
      }
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
