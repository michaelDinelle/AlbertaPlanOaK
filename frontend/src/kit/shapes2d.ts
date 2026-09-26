import * as THREE from 'three/webgpu';
import { int, vec3 } from 'three/tsl';
import { tslExports } from 'vgpu/three';
import shapes from './shapes.wgsl';

type N<T extends string> = THREE.Node<T>;
type Scalar = N<'float'> | number;

const wgsl = tslExports(shapes)('sdCircle', 'sdBox', 'sdSegment', 'coverage', 'noise', 'fractal', 'grain');

export const sdCircle = (p: N<'vec2'>, radius: Scalar): N<'float'> => wgsl.sdCircle({ p, radius }) as unknown as N<'float'>;

export const sdBox = (p: N<'vec2'>, half: N<'vec2'>, radius: Scalar = 0): N<'float'> => wgsl.sdBox({ p, extent: half, radius }) as unknown as N<'float'>;

export const sdSegment = (p: N<'vec2'>, a: N<'vec2'>, b: N<'vec2'>, radius: Scalar): N<'float'> => wgsl.sdSegment({ p, a, b, radius }) as unknown as N<'float'>;

export const coverage = (distance: N<'float'>): N<'float'> => wgsl.coverage({ d: distance }) as unknown as N<'float'>;

export const noise = (p: N<'vec2'>): N<'float'> => wgsl.noise({ p }) as unknown as N<'float'>;

export const fractal = (p: N<'vec2'>, octaves: number): N<'float'> => wgsl.fractal({ p, octaves: int(octaves) }) as unknown as N<'float'>;

export const grain = (p: N<'vec2'>): N<'vec2'> => wgsl.grain({ p }) as unknown as N<'vec2'>;

export const rgb = (value: THREE.ColorRepresentation): N<'vec3'> => {
  const linear = new THREE.Color(value);
  return vec3(linear.r, linear.g, linear.b);
};
