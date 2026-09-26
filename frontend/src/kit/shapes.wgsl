import { hash2 } from "@vgpu/wgsl-std/hash";
import { fbmPerlin2d, perlin2d } from "@vgpu/wgsl-std/noise/perlin";

export fn sdCircle(p: vec2f, radius: f32) -> f32 {
  return length(p) - radius;
}

export fn sdBox(p: vec2f, extent: vec2f, radius: f32) -> f32 {
  let q = abs(p) - extent + radius;
  return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

export fn sdSegment(p: vec2f, a: vec2f, b: vec2f, radius: f32) -> f32 {
  let pa = p - a;
  let ba = b - a;
  return length(pa - ba * clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0)) - radius;
}

export fn coverage(d: f32) -> f32 {
  return clamp(0.5 - d / max(fwidth(d), 1e-6), 0.0, 1.0);
}

export fn noise(p: vec2f) -> f32 {
  return perlin2d(p);
}

export fn fractal(p: vec2f, octaves: i32) -> f32 {
  return fbmPerlin2d(p, octaves, 2.0, 0.5);
}

export fn grain(p: vec2f) -> vec2f {
  return hash2(p);
}
