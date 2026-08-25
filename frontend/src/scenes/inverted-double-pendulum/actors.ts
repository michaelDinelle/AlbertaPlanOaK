import * as THREE from 'three';
import { makeFigure, type FigureBag } from '@kit/figure';
import type { Palette } from '@kit/tokens';
import { IDP_LINK_LEN } from '@sim/envs/inverteddoublependulum';
import { RAIL_Y } from './world';

export interface DoubleCartHandle {
  group: THREE.Group;
  cart: THREE.Object3D;
  hinge1: THREE.Object3D;
  hinge2: THREE.Object3D;
  setState(x: number, h1: number, h2: number): void;
  setGlow(k: number): void;
  setTime(t: number): void;
  dispose(): void;
}

export interface DoubleCartOptions {
  palette: Palette;
  accent: number;
}

export function buildDoubleCart(opts: DoubleCartOptions): DoubleCartHandle {
  const bag: FigureBag = makeFigure({ palette: opts.palette, accent: opts.accent, name: 'doubleRailCart' });

  const cart = bag.joint(bag.group, 'cart', 0, RAIL_Y, 0);
  bag.box(cart, 0.34, 0.18, 0.22, 'cartBody', true);
  for (const sx of [-1, 1]) {
    const wheel = bag.orb(cart, 0.05, 'cartWheel');
    wheel.position.set(sx * 0.11, -0.1, 0);
  }

  const hinge1 = bag.joint(cart, 'hinge1', 0, 0.1, 0);
  bag.limb(hinge1, IDP_LINK_LEN, 0.04, 'link1');
  const elbow = bag.orb(hinge1, 0.06, 'elbow', true);
  elbow.position.set(0, -IDP_LINK_LEN, 0);

  const hinge2 = bag.joint(hinge1, 'hinge2', 0, -IDP_LINK_LEN, 0);
  bag.limb(hinge2, IDP_LINK_LEN, 0.034, 'link2');
  const tip = bag.orb(hinge2, 0.07, 'tip');
  tip.position.set(0, -IDP_LINK_LEN, 0);

  return {
    group: bag.group,
    cart,
    hinge1,
    hinge2,

    setState(x, h1, h2) {
      cart.position.x = x;
      hinge1.rotation.z = Math.PI + h1;
      hinge2.rotation.z = h2;
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
