import * as THREE from 'three/webgpu';
import { Fn, clamp, color, cos, dot, float, floor, fract, fwidth, length, max, mix, normalize, positionGeometry, pow, sin, smoothstep, sqrt, step, uniform, varying, vec2, vec3 } from 'three/tsl';
import { glFragCoord } from './materials';
import type { Palette } from './tokens';

type N<T extends string> = THREE.Node<T>;

export interface SkyStarOptions {
  density?: number;
  size?: number;
  threshold?: number;
  twinkle?: number;
  color?: THREE.ColorRepresentation;
  intensity?: number;
}

export interface GradientSkyOptions {
  top: THREE.ColorRepresentation;
  horizon: THREE.ColorRepresentation;
  bottom: THREE.ColorRepresentation;
  topSpread?: number;
  bottomSpread?: number;

  sunDir?: THREE.Vector3 | readonly [number, number, number] | null;
  sunColor?: THREE.ColorRepresentation;
  sunSize?: number;
  sunHdr?: number;
  sunGlow?: number;
  sunGlowPower?: number;

  stars?: boolean | SkyStarOptions;
  moon?: boolean;
  radius?: number;
  dither?: number;
  name?: string;
}

export type SkyHandle = ReturnType<typeof makeGradientSky>;
export type SkyTerm = (dir: N<'vec3'>, time: N<'float'>) => N<'vec3'>;

const hash13 = (q: N<'vec3'>): N<'float'> => {
  const p = fract(q.mul(0.1031)).toVar();
  p.addAssign(dot(p, p.zyx.add(31.32)));
  return fract(p.x.add(p.y).mul(p.z));
};

function toVec3(v: THREE.Vector3 | readonly [number, number, number] | null | undefined, out: THREE.Vector3) {
  if (!v) out.set(0, 1, 0);
  else if ('x' in v) out.copy(v);
  else out.fromArray(v);
  if (out.lengthSq() < 1e-12) out.set(0, 1, 0);
  return out.normalize();
}

export function makeGradientSky(opts: GradientSkyOptions) {
  const hasSun = opts.sunDir !== null && opts.sunDir !== undefined;
  const starOpts: SkyStarOptions | null = opts.stars === true ? {} : opts.stars ? opts.stars : null;

  const sunDirection = new THREE.Vector3();
  toVec3(opts.sunDir ?? null, sunDirection);

  const uniforms = {
    uTop: uniform(new THREE.Color(opts.top)),
    uHorizon: uniform(new THREE.Color(opts.horizon)),
    uBottom: uniform(new THREE.Color(opts.bottom)),
    uSunDir: uniform(sunDirection),
    uTime: uniform(0),
  };
  const terms: SkyTerm[] = [];
  const direction = varying(normalize(positionGeometry));

  const material = new THREE.MeshBasicNodeMaterial({ side: THREE.BackSide, depthWrite: false, depthTest: false, fog: false });
  material.colorNode = Fn(() => {
    const dir = normalize(direction), time = uniforms.uTime;
    const h = clamp(dir.y, -1, 1);
    const col = vec3(0).toVar();
    col.assign(uniforms.uHorizon);
    col.assign(mix(col, uniforms.uTop, smoothstep(0, Math.max(opts.topSpread ?? 0.55, 0.001), h)));
    col.assign(mix(col, uniforms.uBottom, smoothstep(0, Math.max(opts.bottomSpread ?? 0.35, 0.001), h.negate())));
    if (hasSun) {
      const sd = normalize(uniforms.uSunDir), sunColor = color(new THREE.Color(opts.sunColor ?? 0xffffff));
      const cosA = dot(dir, sd), sz = Math.max(opts.sunSize ?? 0.045, 0.002), hdr = opts.sunHdr ?? 6, glowGain = opts.sunGlow ?? 0.35;
      const glow = pow(max(cosA, 0), Math.max(opts.sunGlowPower ?? 48, 1));
      if (opts.moon) {
        const right = normalize(sd.cross(vec3(0, 1, 0))), up = right.cross(sd);
        const p = vec2(dot(dir, right), dot(dir, up)).div(Math.sin(sz));
        const r2 = dot(p, p), edge = max(fwidth(r2), 0.001);
        const disc = smoothstep(edge.oneMinus(), edge.add(1), r2).oneMinus();
        const normal = vec3(p, sqrt(max(r2.oneMinus(), 0)));
        const maria = sin(p.x.mul(8).add(sin(p.y.mul(5)))).mul(sin(p.y.mul(7).sub(p.x.mul(3))));
        const craters = sin(p.x.mul(43).add(sin(p.y.mul(27)))).mul(sin(p.y.mul(39).add(p.x.mul(17))));
        const shade = max(dot(normal, normalize(vec3(-0.5, 0.4, 0.8))), 0).mul(0.7).add(0.3).mul(maria.mul(0.16).add(0.78).add(craters.mul(0.04)));
        col.assign(mix(col, sunColor.mul(shade).mul(hdr), disc));
        col.addAssign(sunColor.mul(glow).mul(glowGain));
      } else {
        const disc = smoothstep(Math.cos(sz * 1.4), Math.cos(sz * 0.85), cosA);
        col.addAssign(sunColor.mul(disc.mul(hdr).add(glow.mul(glowGain))));
      }
    }
    if (starOpts) {
      const above = smoothstep(-0.08, 0.12, dir.y);
      const g = dir.mul(starOpts.density ?? 160), c = floor(g), cell = hash13(c);
      const pt = vec3(hash13(c.add(1.7)), hash13(c.add(3.3)), hash13(c.add(5.9)));
      const d = length(g.sub(c).sub(pt));
      const star = smoothstep(0, Math.max(starOpts.size ?? 0.06, 0.001), d).oneMinus().mul(step(starOpts.threshold ?? 0.965, cell));
      const tw = sin(time.mul(2.7).add(cell.mul(63))).add(1).mul((starOpts.twinkle ?? 0.35) * 0.5).oneMinus();
      col.addAssign(color(new THREE.Color(starOpts.color ?? 0xffffff)).mul(star.mul(starOpts.intensity ?? 2.2).mul(tw).mul(above)));
    }
    col.addAssign(hash13(vec3(glFragCoord, 1)).sub(0.5).mul(opts.dither ?? 0.006));
    for (const term of terms) col.addAssign(term(dir, time));
    return col;
  })();
  material.name = opts.name ?? 'oakSky';

  const geometry = new THREE.IcosahedronGeometry(1, 2);
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = opts.name ?? 'oakSky';
  const radius = opts.radius ?? 400;
  mesh.scale.setScalar(radius);
  mesh.renderOrder = -1000;
  mesh.frustumCulled = false;
  mesh.matrixAutoUpdate = true;
  mesh.onBeforeRender = (_renderer, _scene, camera) => {
    mesh.position.copy(camera.position);
    mesh.updateMatrixWorld();
  };

  return {
    mesh,
    material,
    uniforms,
    sunDirection,
    add(term: SkyTerm) { terms.push(term); material.needsUpdate = true; },
    setSunDirection(x: number, y: number, z: number) { toVec3([x, y, z], sunDirection); },
    fit(far: number) { mesh.scale.setScalar(far * 0.85 / radius); },
    setColors(top: THREE.ColorRepresentation, horizon: THREE.ColorRepresentation, bottom: THREE.ColorRepresentation) {
      uniforms.uTop.value.set(top);
      uniforms.uHorizon.value.set(horizon);
      uniforms.uBottom.value.set(bottom);
    },
    setTime(t: number) {
      uniforms.uTime.value = t;
    },
    dispose() {
      mesh.onBeforeRender = () => {};
      geometry.dispose();
      material.dispose();
    },
  };
}

export function paletteSunDirection(palette: Palette, out?: THREE.Vector3): THREE.Vector3 {
  const v = out ?? new THREE.Vector3();
  const k = Math.max(0, Math.min(1, palette.sunIntensity / 3));
  const elevation = 0.22 + 0.68 * k;
  const azimuth = Math.PI * 0.32;
  const horiz = Math.sqrt(Math.max(0, 1 - elevation * elevation));
  return v.set(Math.cos(azimuth) * horiz, elevation, Math.sin(azimuth) * horiz).normalize();
}

export type PaletteSkyOptions = Partial<Omit<GradientSkyOptions, 'top' | 'horizon' | 'bottom'>>;

export function skyFromPalette(
  palette: Palette,
  paletteName: string,
  overrides: PaletteSkyOptions = {},
): SkyHandle {
  const depth = paletteName === 'underwater' || paletteName === 'abyss' || paletteName === 'cave';
  const night = paletteName === 'night';
  const sunDir = depth ? null : (overrides.sunDir ?? paletteSunDirection(palette));
  const stars = overrides.stars ?? (night ? { intensity: 2.4 } : false);
  return makeGradientSky({
    top: palette.skyTop,
    horizon: palette.skyHorizon,
    bottom: palette.skyBottom,
    sunColor: palette.sun,
    sunHdr: night ? 2.5 : 6,
    ...overrides,
    sunDir,
    stars,
  });
}
