import * as THREE from 'three';

export interface Disposable {
  dispose(): void;
}

export interface Timed {
  setTime(t: number): void;
}

export interface PixelSized {
  setPixelRatio(dpr: number): void;
}

export interface WorldCore {
  group: THREE.Group;
  setTime(t: number): void;
  setPixelRatio(dpr: number): void;
  dispose(): void;
}

export interface WorldHooks {
  time?(t: number): void;
  drop?(): void;
}

export interface WorldBuilder {
  group: THREE.Group;
  add<T extends THREE.Object3D>(object: T): T;
  own<T extends Disposable>(handle: T): T;
  time<T extends Timed>(handle: T): T;
  size<T extends PixelSized>(handle: T): T;
  seal<H extends WorldCore>(extra: Omit<H, keyof WorldCore>, hooks?: WorldHooks): H;
}

export function makeWorld(name: string): WorldBuilder {
  const group = new THREE.Group();
  group.name = name;
  const owned: Disposable[] = [];
  const timed: Timed[] = [];
  const sized: PixelSized[] = [];
  return {
    group,
    add: (object) => (group.add(object), object),
    own: (handle) => (owned.push(handle), handle),
    time: (handle) => (timed.push(handle), handle),
    size: (handle) => (sized.push(handle), handle),
    seal<H extends WorldCore>(extra: Omit<H, keyof WorldCore>, hooks: WorldHooks = {}): H {
      const core: WorldCore = {
        group,
        setTime(t) {
          for (let i = 0; i < timed.length; i++) timed[i]?.setTime(t);
          hooks.time?.(t);
        },
        setPixelRatio(dpr) {
          for (let i = 0; i < sized.length; i++) sized[i]?.setPixelRatio(dpr);
        },
        dispose() {
          for (let i = owned.length - 1; i >= 0; i--) owned[i]?.dispose();
          owned.length = 0;
          timed.length = 0;
          sized.length = 0;
          hooks.drop?.();
          group.removeFromParent();
          group.clear();
        },
      };
      return { ...core, ...extra } as H;
    },
  };
}
