import * as THREE from 'three';
import { makeFigure, type FigureBag } from '@kit/figure';
import { emissiveMaterial, setEmissiveIntensity, type EmissiveMaterial } from '@kit/materials';
import { blendTokens } from '@kit/mix';
import type { Palette } from '@kit/tokens';
import { RE_LINK0_LEN, RE_LINK1_LEN } from '@sim/envs/reacher';
import { REACH_SCALE, TABLE_Y } from './world';

export const LINK0 = RE_LINK0_LEN * REACH_SCALE;
export const LINK1 = RE_LINK1_LEN * REACH_SCALE;

export interface ArmHandle {
  group: THREE.Group;
  joint0: THREE.Object3D;
  joint1: THREE.Object3D;
  target: THREE.Object3D;
  fingertip: THREE.Vector3;
  setState(q0: number, q1: number, tx: number, ty: number): void;
  setGlow(k: number): void;
  setTime(t: number): void;
  dispose(): void;
}

export interface ArmOptions {
  palette: Palette;
  accent: number;
}

export function buildArm(opts: ArmOptions): ArmHandle {
  const bag: FigureBag = makeFigure({ palette: opts.palette, accent: opts.accent, name: 'reachArm' });

  const base = bag.joint(bag.group, 'armBase', 0, TABLE_Y + 0.08, 0);
  const hub = bag.orb(base, 0.11, 'armHub', true);
  hub.scale.y = 0.7;

  const joint0 = bag.joint(base, 'joint0');
  bag.rod(joint0, LINK0, 0.05, 'link0');
  const elbowCap = bag.orb(joint0, 0.075, 'elbowCap', true);
  elbowCap.position.set(LINK0, 0, 0);

  const joint1 = bag.joint(joint0, 'joint1', LINK0, 0, 0);
  bag.rod(joint1, LINK1, 0.042, 'link1');
  const finger = bag.orb(joint1, 0.06, 'fingertip');
  finger.position.set(LINK1, 0, 0);

  const targetGeometry = new THREE.IcosahedronGeometry(0.09, 1);
  const targetMaterial: EmissiveMaterial = emissiveMaterial(
    blendTokens(opts.accent, opts.palette.sun, 0.3),
    2.2,
    { name: 'reachTarget' },
  );
  const target = new THREE.Mesh(targetGeometry, targetMaterial);
  target.name = 'reachTarget';
  bag.group.add(target);

  const fingertip = new THREE.Vector3();

  return {
    group: bag.group,
    joint0,
    joint1,
    target,
    fingertip,

    setState(q0, q1, tx, ty) {
      joint0.rotation.y = q0;
      joint1.rotation.y = q1;
      target.position.set(tx * REACH_SCALE, TABLE_Y + 0.1, -ty * REACH_SCALE);
      finger.getWorldPosition(fingertip);
    },

    setGlow(k) {
      setEmissiveIntensity(targetMaterial, 2.2 * (0.7 + 0.6 * k));
    },

    setTime(t) {
      target.scale.setScalar(1 + 0.1 * Math.sin(t * 3.1));
    },

    dispose() {
      targetGeometry.dispose();
      targetMaterial.dispose();
      bag.dispose();
    },
  };
}
