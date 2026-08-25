import * as THREE from 'three';
import { makeFigure, type FigureBag } from '@kit/figure';
import { emissiveMaterial, setEmissiveIntensity, type EmissiveMaterial } from '@kit/materials';
import { blendTokens } from '@kit/mix';
import type { Palette } from '@kit/tokens';
import { PUSH_SCALE, TABLE_Y } from './world';

export const UPPER_LEN = 0.55;
export const FORE_LEN = 0.5;
export const HAND_LEN = 0.3;

export interface PusherHandle {
  group: THREE.Group;
  shoulder: THREE.Object3D;
  elbow: THREE.Object3D;
  wrist: THREE.Object3D;
  puck: THREE.Object3D;
  goal: THREE.Object3D;
  setState(pose: ArrayLike<number>): void;
  setGlow(k: number): void;
  setTime(t: number): void;
  dispose(): void;
}

export interface PusherOptions {
  palette: Palette;
  accent: number;
}

export function buildPusher(opts: PusherOptions): PusherHandle {
  const bag: FigureBag = makeFigure({ palette: opts.palette, accent: opts.accent, name: 'benchPusher' });

  const base = bag.joint(bag.group, 'pushBase', 0, TABLE_Y + 0.12, 0);
  const pedestal = bag.orb(base, 0.16, 'pushPedestal', true);
  pedestal.scale.y = 0.6;

  const shoulder = bag.joint(base, 'pushShoulder');
  bag.rod(shoulder, UPPER_LEN, 0.06, 'pushUpperArm');
  const elbowCap = bag.orb(shoulder, 0.085, 'pushElbowCap', true);
  elbowCap.position.x = UPPER_LEN;

  const elbow = bag.joint(shoulder, 'pushElbow', UPPER_LEN, 0, 0);
  bag.rod(elbow, FORE_LEN, 0.05, 'pushForearm');
  const wristCap = bag.orb(elbow, 0.07, 'pushWristCap', true);
  wristCap.position.x = FORE_LEN;

  const wrist = bag.joint(elbow, 'pushWrist', FORE_LEN, 0, 0);
  bag.rod(wrist, HAND_LEN, 0.04, 'pushHand');
  const palm = bag.box(wrist, 0.05, 0.16, 0.24, 'pushPalm');
  palm.position.x = HAND_LEN;

  const puckGeometry = new THREE.CylinderGeometry(0.14, 0.14, 0.12, 14);
  const puck = new THREE.Mesh(puckGeometry, bag.material);
  puck.name = 'pushPuck';
  puck.castShadow = true;
  bag.group.add(puck);

  const goalGeometry = new THREE.TorusGeometry(0.2, 0.026, 6, 40);
  goalGeometry.rotateX(-Math.PI / 2);
  const goalMaterial: EmissiveMaterial = emissiveMaterial(
    blendTokens(opts.accent, opts.palette.sun, 0.3),
    1.8,
    { transparent: true, opacity: 0.9, name: 'pushGoal' },
  );
  const goal = new THREE.Mesh(goalGeometry, goalMaterial);
  goal.name = 'pushGoal';
  bag.group.add(goal);

  return {
    group: bag.group,
    shoulder,
    elbow,
    wrist,
    puck,
    goal,

    setState(pose) {
      shoulder.rotation.y = -(pose[0] ?? 0);
      elbow.rotation.y = -(pose[3] ?? 0);
      wrist.rotation.y = -(pose[5] ?? 0);
      base.rotation.z = -(pose[1] ?? 0) * 0.25;
      puck.position.set((pose[7] ?? 0) * PUSH_SCALE, TABLE_Y + 0.06, -(pose[8] ?? 0) * PUSH_SCALE);
      goal.position.set((pose[9] ?? 0) * PUSH_SCALE, TABLE_Y + 0.02, -(pose[10] ?? 0) * PUSH_SCALE);
    },

    setGlow(k) {
      setEmissiveIntensity(goalMaterial, 1.8 * (0.7 + 0.7 * k));
    },

    setTime(t) {
      goal.scale.setScalar(1 + 0.06 * Math.sin(t * 2.4));
    },

    dispose() {
      puckGeometry.dispose();
      goalGeometry.dispose();
      goalMaterial.dispose();
      bag.dispose();
    },
  };
}
