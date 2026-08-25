import * as THREE from 'three';
import { makeFigure, type FigureBag } from '@kit/figure';
import type { Palette } from '@kit/tokens';
import {
  HU_ABD_X,
  HU_ABD_Y,
  HU_ABD_Z,
  HU_L_ELBOW,
  HU_L_HIP_X,
  HU_L_HIP_Y,
  HU_L_HIP_Z,
  HU_L_KNEE,
  HU_L_SHOULDER1,
  HU_L_SHOULDER2,
  HU_QW,
  HU_QX,
  HU_QY,
  HU_QZ,
  HU_R_ELBOW,
  HU_R_HIP_X,
  HU_R_HIP_Y,
  HU_R_HIP_Z,
  HU_R_KNEE,
  HU_R_SHOULDER1,
  HU_R_SHOULDER2,
  HU_X,
  HU_Y,
  HU_Z,
} from '@sim/envs/humanoid';

export const THIGH_LEN = 0.4;
export const SHIN_LEN = 0.4;
export const FOOT_LEN = 0.2;
export const TORSO_LEN = 0.42;
export const UPPER_ARM_LEN = 0.28;
export const FOREARM_LEN = 0.28;

interface HumanoidLegJoints {
  hip: THREE.Object3D;
  knee: THREE.Object3D;
}

interface HumanoidArmJoints {
  shoulder: THREE.Object3D;
  elbow: THREE.Object3D;
}

export interface HumanoidHandle {
  group: THREE.Group;
  root: THREE.Object3D;
  setPose(pose: Float32Array): void;
  setGlow(k: number): void;
  setTime(t: number): void;
  dispose(): void;
}

export interface HumanoidOptions {
  palette: Palette;
  accent: number;
  name: string;
}

function buildHumanoidLeg(bag: FigureBag, pelvis: THREE.Object3D, side: string, z: number, shaded: boolean): HumanoidLegJoints {
  const hip = bag.joint(pelvis, `hum${side}Hip`, 0, -0.08, z);
  bag.limb(hip, THIGH_LEN, 0.07, `hum${side}Thigh`, shaded);
  const knee = bag.joint(hip, `hum${side}Knee`, 0, -THIGH_LEN, 0);
  bag.limb(knee, SHIN_LEN, 0.052, `hum${side}Shin`, shaded);
  const foot = bag.rod(knee, FOOT_LEN, 0.05, `hum${side}Foot`, shaded);
  foot.position.set(-FOOT_LEN / 4, -SHIN_LEN, 0);
  return { hip, knee };
}

function buildHumanoidArm(bag: FigureBag, chest: THREE.Object3D, side: string, z: number, shaded: boolean): HumanoidArmJoints {
  const shoulder = bag.joint(chest, `hum${side}Shoulder`, 0, TORSO_LEN - 0.04, z);
  bag.limb(shoulder, UPPER_ARM_LEN, 0.05, `hum${side}UpperArm`, shaded);
  const elbow = bag.joint(shoulder, `hum${side}Elbow`, 0, -UPPER_ARM_LEN, 0);
  bag.limb(elbow, FOREARM_LEN, 0.042, `hum${side}Forearm`, shaded);
  const hand = bag.orb(elbow, 0.05, `hum${side}Hand`, shaded);
  hand.position.y = -FOREARM_LEN;
  return { shoulder, elbow };
}

export function buildHumanoid(opts: HumanoidOptions): HumanoidHandle {
  const bag: FigureBag = makeFigure({ palette: opts.palette, accent: opts.accent, name: opts.name });

  const root = bag.joint(bag.group, 'humRoot');
  const pelvis = bag.box(root, 0.16, 0.14, 0.26, 'humPelvis', true);
  pelvis.position.y = -0.02;

  const chest = bag.joint(root, 'humChest', 0, 0.08, 0);
  const torso = bag.limb(chest, TORSO_LEN, 0.11, 'humTorso');
  torso.position.y = TORSO_LEN;
  const head = bag.orb(chest, 0.1, 'humHead');
  head.position.y = TORSO_LEN + 0.14;

  const rightLeg = buildHumanoidLeg(bag, root, 'Right', 0.1, false);
  const leftLeg = buildHumanoidLeg(bag, root, 'Left', -0.1, true);
  const rightArm = buildHumanoidArm(bag, chest, 'Right', 0.19, false);
  const leftArm = buildHumanoidArm(bag, chest, 'Left', -0.19, true);

  return {
    group: bag.group,
    root,

    setPose(pose) {
      root.position.set(pose[HU_X] ?? 0, pose[HU_Z] ?? 0, -(pose[HU_Y] ?? 0));
      root.quaternion.set(pose[HU_QX] ?? 0, pose[HU_QZ] ?? 0, -(pose[HU_QY] ?? 0), pose[HU_QW] ?? 1);

      chest.rotation.set(pose[HU_ABD_X] ?? 0, -(pose[HU_ABD_Z] ?? 0), -(pose[HU_ABD_Y] ?? 0));

      rightLeg.hip.rotation.set(-(pose[HU_R_HIP_X] ?? 0), -(pose[HU_R_HIP_Z] ?? 0), -(pose[HU_R_HIP_Y] ?? 0));
      rightLeg.knee.rotation.z = -(pose[HU_R_KNEE] ?? 0);
      leftLeg.hip.rotation.set(pose[HU_L_HIP_X] ?? 0, pose[HU_L_HIP_Z] ?? 0, -(pose[HU_L_HIP_Y] ?? 0));
      leftLeg.knee.rotation.z = -(pose[HU_L_KNEE] ?? 0);

      rightArm.shoulder.rotation.set(pose[HU_R_SHOULDER1] ?? 0, 0, -(pose[HU_R_SHOULDER2] ?? 0));
      rightArm.elbow.rotation.z = -(pose[HU_R_ELBOW] ?? 0);
      leftArm.shoulder.rotation.set(-(pose[HU_L_SHOULDER1] ?? 0), 0, pose[HU_L_SHOULDER2] ?? 0);
      leftArm.elbow.rotation.z = pose[HU_L_ELBOW] ?? 0;
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
