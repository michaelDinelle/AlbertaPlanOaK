import * as THREE from 'three/webgpu';
import { Break, Fn, If, Loop, abs, convertToTexture, distance, dot, float, length, max, min, mix, pass, renderOutput, screenSize, smoothstep, uniform, uniformArray, uv, vec2, vec3, vec4 } from 'three/tsl';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import type { FogRecipe, ScenePost } from './types';
import { TOKENS } from './tokens';
import { prepareInstances } from './instances';
export const MSAA_SAMPLES = 4;

const BLOOM_THRESHOLD = 0.85;
const BLOOM_SCALE = 3;

function resolveFog(recipe: FogRecipe): THREE.Color {
  const color = new THREE.Color(TOKENS.palettes[recipe.from[0]][recipe.from[1]]);
  for (const [token, k] of recipe.blend ?? []) {
    color.lerp(new THREE.Color(TOKENS.palettes[token[0]][token[1]]), k);
  }
  return color;
}

export interface ComposerHandle {
  setBloomPulse(t01: number): void;
  render(): void;
  setGlass(rects: readonly GlassRect[]): void;
  compile(): Promise<void>;
  dispose(): void;
}

export const BLOOM_RESOLUTION_SCALE = 0.125;

export function pulsedBloomStrength(base: number, t01: number): number {
  const t = t01 < 0 ? 0 : t01 > 1 ? 1 : t01;
  return base * (1 + 1.2 * t);
}

export interface GlassRect {
  x: number;
  y: number;
  width: number;
  height: number;
  radius: number;
}

export const GLASS_RECTS = 16;

const SCALE = -112;
const CHROMA = 6;
const BORDER = 0.07;
const MAP_BLUR = 12;
const BLUR = 3;
const SATURATE = 1.5;

type N<T extends string> = THREE.Node<T>;

const rounded = (p: N<'vec2'>, rect: N<'vec4'>, inset: N<'float'> | number, radius: N<'float'>): N<'float'> => {
  const half = rect.zw.mul(0.5);
  const r = max(radius.sub(inset), 2);
  const q = abs(p.sub(rect.xy).sub(half)).sub(half.sub(inset).sub(r));
  return length(max(q, 0)).add(min(max(q.x, q.y), 0)).sub(r);
};

export function buildComposer(
  renderer: THREE.WebGPURenderer,
  scene: THREE.Scene,
  camera: THREE.Camera,
  post: ScenePost,
): ComposerHandle {
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = TOKENS.exposure;

  const scenePass = pass(scene, camera, { samples: MSAA_SAMPLES });
  const beforeRender = scene.onBeforeRender;
  scene.onBeforeRender = (...args) => { beforeRender.apply(scene, args); prepareInstances(scene, args[2], args[3] as unknown as THREE.RenderTarget | null); };
  const color = scenePass.getTextureNode('output');
  const glow = bloom(color, post.bloom * BLOOM_SCALE, 0.6, BLOOM_THRESHOLD);
  glow.setResolutionScale(BLOOM_RESOLUTION_SCALE);

  const tint = uniform(resolveFog(post.fog));
  const graded = Fn(() => {
    const col = max(color.rgb.add(glow.rgb), 0).toVar();
    col.addAssign(vec3(0.004).mul(smoothstep(0, 0.35, col).oneMinus()));
    col.assign(mix(col, col.mul(tint), 0.14));
    col.mulAssign(smoothstep(0.34, 0.86, distance(uv(), vec2(0.5))).mul(0.34).oneMinus());
    return vec4(max(col, 0), 1);
  })();
  const display = convertToTexture(renderOutput(graded));

  const rects = uniformArray(Array.from({ length: GLASS_RECTS }, () => new THREE.Vector4()), 'vec4' as const);
  const radii = uniformArray(new Array<number>(GLASS_RECTS).fill(0), 'float' as const);
  const count = uniform(0, 'int');
  const blurred = (px: N<'vec2'>): N<'vec3'> => {
    const t = vec2(1).div(screenSize), at = px.mul(t);
    return display.sample(at.add(vec2(BLUR, 0).mul(t))).rgb.add(display.sample(at.sub(vec2(BLUR, 0).mul(t))).rgb)
      .add(display.sample(at.add(vec2(0, BLUR).mul(t))).rgb).add(display.sample(at.sub(vec2(0, BLUR).mul(t))).rgb).mul(0.25);
  };
  const glass = Fn(() => {
    const px = uv().mul(screenSize);
    const rect = vec4(0).toVar(), radius = float(0).toVar(), best = float(1e30).toVar();
    Loop(GLASS_RECTS, ({ i }) => {
      If(i.greaterThanEqual(count), () => { Break(); });
      const r = rects.element(i);
      If(rounded(px, r, 0, radii.element(i)).lessThanEqual(0).and(r.z.mul(r.w).lessThan(best)), () => {
        best.assign(r.z.mul(r.w));
        rect.assign(r);
        radius.assign(radii.element(i));
      });
    });
    const out = display.sample(uv()).toVar();
    If(best.lessThan(1e30), () => {
      const centred = px.sub(rect.xy).div(rect.zw).sub(0.5);
      const open = smoothstep(-MAP_BLUR, MAP_BLUR, rounded(px, rect, min(rect.z, rect.w).mul(BORDER), radius)).oneMinus().mul(0.93).oneMinus();
      const col = vec3(
        blurred(px.add(centred.mul(open.mul(SCALE)))).r,
        blurred(px.add(centred.mul(open.mul(SCALE + CHROMA)))).g,
        blurred(px.add(centred.mul(open.mul(SCALE + 2 * CHROMA)))).b,
      );
      out.assign(vec4(mix(vec3(dot(col, vec3(0.2126, 0.7152, 0.0722))), col, SATURATE), 1));
    });
    return out;
  })();

  const pipeline = new THREE.RenderPipeline(renderer, glass);
  pipeline.outputColorTransform = false;

  return {
    setBloomPulse(t01: number) {
      glow.strength.value = pulsedBloomStrength(post.bloom, t01) * BLOOM_SCALE;
    },
    render() {
      pipeline.render();
    },
    setGlass(list) {
      count.value = Math.min(GLASS_RECTS, list.length);
      for (let i = 0; i < count.value; i++) {
        const r = list[i]!;
        (rects.array[i] as THREE.Vector4).set(r.x, r.y, r.width, r.height);
        radii.array[i] = r.radius;
      }
    },
    async compile() {
      const target = renderer.getRenderTarget();
      renderer.setRenderTarget(scenePass.renderTarget);
      await renderer.compileAsync(scene, camera);
      renderer.setRenderTarget(target);
    },
    dispose() {
      scene.onBeforeRender = beforeRender;
      pipeline.dispose();
      glow.dispose();
      scenePass.dispose();
    },
  };
}
