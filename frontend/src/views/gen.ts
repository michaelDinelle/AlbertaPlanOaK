import * as THREE from 'three';
import { makeOrbitRig } from '@kit/rigs';
import { makeLightRig } from '@kit/lighting';
import { skyFromPalette } from '@kit/sky';
import { oakMaterial } from '@kit/materials';
import { TOKENS } from '@kit/tokens';
import { assemble } from '@kit/gen/assembly';
import { patch, planePatch } from '@kit/gen/patch';
import { curvePath, revolve, sweep } from '@kit/gen/sweep';
import { circleOutline, polygonOutline, starOutline } from '@kit/gen/outline';
import { veil } from '@kit/gen/veil';
import type { ViewHandle } from '../router';

const PALETTE = TOKENS.palettes.dusk;

function label(text: string, x: number): THREE.Sprite {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 64;
  const g = canvas.getContext('2d');
  if (g) {
    g.fillStyle = '#0b0f1d';
    g.fillRect(0, 0, 256, 64);
    g.fillStyle = '#cfe6f0';
    g.font = '28px ui-monospace, monospace';
    g.textAlign = 'center';
    g.fillText(text, 128, 42);
  }
  const map = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map, transparent: true }));
  sprite.scale.set(2.2, 0.55, 1);
  sprite.position.set(x, -1.6, 0);
  return sprite;
}

export function mount(el: HTMLElement): ViewHandle {
  const canvas = document.createElement('canvas');
  canvas.className = 'theater-canvas';
  el.appendChild(canvas);

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setClearColor(0x05070c, 1);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = TOKENS.exposure;

  const scene = new THREE.Scene();
  scene.add(skyFromPalette(PALETTE, 'dusk', 'hero').mesh);
  const lights = makeLightRig(PALETTE, 'hero', { pointCount: 0 });
  scene.add(lights.group);

  const material = oakMaterial({
    color: 0xffffff,
    vertexColors: true,
    roughness: 0.7,
    toonSteps: 3,
    rimColor: PALETTE.sun,
    rimStrength: 0.8,
    tier: 'hero',
    name: 'genGallery',
  });

  const owned: THREE.BufferGeometry[] = [];
  const add = (geometry: THREE.BufferGeometry, x: number, text: string): void => {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.x = x;
    scene.add(mesh);
    scene.add(label(text, x));
    owned.push(geometry);
  };

  const vine = sweep({
    path: curvePath([
      new THREE.Vector3(0, -1, 0),
      new THREE.Vector3(0.35, -0.2, 0.25),
      new THREE.Vector3(-0.25, 0.5, -0.2),
      new THREE.Vector3(0.1, 1.2, 0.1),
    ]),
    rings: 48,
    shape: circleOutline(10),
    radius: (t) => 0.16 * (1 - 0.7 * t),
    twist: (t) => t * 3,
    color: (t, a, out) => out.setHSL(0.28 + 0.1 * t, 0.5, 0.28 + 0.22 * Math.abs(Math.cos(a * 6.28))),
    name: 'gallerySweep',
  });
  add(vine, -6, 'sweep');

  const lantern = revolve(
    [[0, -1], [0.42, -0.86], [0.6, -0.35], [0.55, 0.35], [0.32, 0.8], [0.1, 1], [0, 1.05]],
    18,
    {
      color: (t, _a, out) => out.setHSL(0.09, 0.62, 0.32 + 0.3 * t),
      name: 'galleryRevolve',
    },
  );
  add(lantern, -3, 'revolve');

  const petal = patch({
    segmentsU: 24,
    segmentsV: 16,
    doubleSided: true,
    name: 'galleryPatch',
    point(u, v, out) {
      const a = (u - 0.5) * Math.PI * 1.4;
      const r = 1.1 * Math.sin(Math.PI * v);
      out.set(Math.sin(a) * r, v * 1.9 - 1, Math.cos(a) * r * 0.35 + Math.sin(v * 6) * 0.08);
    },
    color: (_u, v, out) => out.setHSL(0.92, 0.42, 0.34 + 0.3 * v),
  });
  add(petal, 0, 'patch');

  const shrine = assemble(
    [
      {
        geometry: planePatch(1.6, 1.6, 4, 4, (x, z) => Math.sin(x * 2) * 0.05 + Math.cos(z * 2) * 0.05),
        color: 0x6b7f8c,
        position: [0, -1, 0],
      },
      {
        geometry: revolve([[0.34, 0], [0.3, 0.5], [0.22, 1.1], [0, 1.25]], 12),
        color: 0xb8a68a,
        position: [0, -1, 0],
      },
      {
        geometry: sweep({
          path: curvePath([new THREE.Vector3(-0.6, 0, 0), new THREE.Vector3(0, 0.55, 0), new THREE.Vector3(0.6, 0, 0)]),
          rings: 24,
          shape: polygonOutline(5, 1),
          radius: 0.075,
        }),
        color: 0x8fd47a,
        position: [0, 0.35, 0],
      },
      {
        geometry: sweep({
          path: curvePath([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0.3, 0)]),
          rings: 4,
          shape: starOutline(6, 0.35, 1),
          radius: 0.2,
        }),
        color: 0xffd166,
        position: [0, 0.95, 0],
      },
    ],
    'galleryAssembly',
  );
  add(shrine, 3, 'assembly');

  const curtain = veil({
    layers: 5,
    width: 2.2,
    height: 2.6,
    spacing: 0.16,
    color: 0x2a8fc4,
    edge: 0xff5ec8,
    drift: [0.02, -0.05],
    scale: 3.2,
    sharpness: 2.1,
    gain: 0.85,
    taper: 1.4,
    name: 'galleryVeil',
  });
  curtain.group.position.set(6, 0.2, 0);
  scene.add(curtain.group);
  scene.add(label('veil', 6));

  const rig = makeOrbitRig({
    canvas,
    target: [0, 0, 0],
    distance: 15,
    azimuth: 0.2,
    polar: 1.15,
    fov: TOKENS.fovHero,
    near: 0.1,
    far: 600,
    name: 'galleryRig',
  });
  rig.snap();

  const resize = (): void => {
    const w = Math.max(1, el.clientWidth);
    const h = Math.max(1, el.clientHeight);
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    renderer.setSize(w, h, false);
    rig.resize(w / h);
  };
  resize();
  window.addEventListener('resize', resize);

  let raf = 0;
  let last = 0;
  const tick = (now: number): void => {
    raf = requestAnimationFrame(tick);
    const dt = last === 0 ? 0 : Math.min(0.1, (now - last) / 1000);
    last = now;
    rig.update(dt);
    curtain.setTime(now / 1000);
    renderer.render(scene, rig.camera);
  };
  raf = requestAnimationFrame(tick);

  return {
    destroy() {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', resize);
      for (const g of owned) g.dispose();
      curtain.dispose();
      material.dispose();
      lights.dispose();
      renderer.dispose();
      canvas.remove();
    },
  };
}
