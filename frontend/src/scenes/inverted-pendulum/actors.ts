import * as THREE from 'three';
import { makeFigure, type FigureBag } from '@kit/figure';
import type { Palette } from '@kit/tokens';
import { IP_POLE_LEN } from '@sim/envs/invertedpendulum';
import { RAIL_Y } from './world';

export interface CartHandle {
  group: THREE.Group;
  cart: THREE.Object3D;
  poleJoint: THREE.Object3D;
  setState(x: number, theta: number): void;
  setGlow(k: number): void;
  setTime(t: number): void;
  dispose(): void;
}

export interface CartOptions {
  palette: Palette;
  accent: number;
}

export function buildCart(opts: CartOptions): CartHandle {
  const bag: FigureBag = makeFigure({ palette: opts.palette, accent: opts.accent, name: 'railCart' });

  const cart = bag.joint(bag.group, 'cart', 0, RAIL_Y, 0);
  bag.box(cart, 0.34, 0.18, 0.22, 'cartBody', true);
  for (const sx of [-1, 1]) {
    const wheel = bag.orb(cart, 0.05, 'cartWheel');
    wheel.position.set(sx * 0.11, -0.1, 0);
  }

  const poleJoint = bag.joint(cart, 'poleJoint', 0, 0.1, 0);
  bag.limb(poleJoint, IP_POLE_LEN, 0.038, 'pole');
  const tip = bag.orb(poleJoint, 0.07, 'poleTip');
  tip.position.set(0, -IP_POLE_LEN, 0);

  return {
    group: bag.group,
    cart,
    poleJoint,

    setState(x, theta) {
      cart.position.x = x;
      poleJoint.rotation.z = Math.PI + theta;
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
