import * as THREE from 'three/webgpu';

export interface Disposable {
  dispose(): void;
}

export interface Timed {
  setTime(t: number): void;
}

export interface WorldCore {
  group: THREE.Group;
  setTime(t: number): void;
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
  seal<H extends WorldCore>(extra: Omit<H, keyof WorldCore>, hooks?: WorldHooks): H;
}

export function makeWorld(name: string): WorldBuilder {
  const group = new THREE.Group();
  group.name = name;
  const owned: Disposable[] = [];
  const timed: Timed[] = [];
  const builder: WorldBuilder = {
    group,
    add(object) {
      let parent = object.parent;
      while (parent && parent !== group) parent = parent.parent;
      if (!parent) group.add(object);
      return object;
    },
    own(handle) {
      owned.push(handle);
      return handle;
    },
    time: (handle) => (timed.push(handle), handle),
    seal<H extends WorldCore>(extra: Omit<H, keyof WorldCore>, hooks: WorldHooks = {}): H {
      const core: WorldCore = {
        group,
        setTime(t) {
          for (let i = 0; i < timed.length; i++) timed[i]?.setTime(t);
          hooks.time?.(t);
        },
        dispose() {
          for (let i = owned.length - 1; i >= 0; i--) owned[i]?.dispose();
          owned.length = 0;
          timed.length = 0;
          hooks.drop?.();
          group.removeFromParent();
          group.clear();
        },
      };
      return { ...core, ...extra } as H;
    },
  };
  return builder;
}
