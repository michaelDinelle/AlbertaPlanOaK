import * as THREE from 'three';

export function leafSurface(material: THREE.MeshStandardMaterial, depth: THREE.MeshDepthMaterial): void {
  material.alphaToCoverage = true;
  for (const target of [material, depth]) {
    target.onBeforeCompile = shader => {
      shader.vertexShader = shader.vertexShader.replace('#include <common>', `#include <common>
attribute vec2 aLeaf;
varying vec2 vLeaf;
`).replace('#include <begin_vertex>', '#include <begin_vertex>\nvLeaf = aLeaf;')
        .replace('#include <color_vertex>', `#include <color_vertex>
#ifdef USE_INSTANCING_COLOR
if (aLeaf.x < -1.9) vColor.rgb /= max(instanceColor, vec3(0.001));
#endif
`);
      shader.fragmentShader = shader.fragmentShader.replace('#include <common>', '#include <common>\nvarying vec2 vLeaf;')
        .replace('#include <map_fragment>', `
if (vLeaf.x >= -1.9) {
  vec2 leaf = vec2(vLeaf.x / (0.9 - vLeaf.y * 0.15), vLeaf.y);
  float edge = 1.0 - dot(leaf, leaf);
  ${target === material ? 'float width = max(fwidth(edge), 0.0001); diffuseColor.a = smoothstep(-width, width, edge); if (diffuseColor.a == 0.0) discard;' : 'if (edge < 0.0) discard;'}
  float veinWidth = max(fwidth(vLeaf.x), 0.012);
  float vein = 1.0 - smoothstep(0.0, veinWidth, abs(vLeaf.x));
  diffuseColor.rgb *= 0.92 + vein * 0.12 + 0.08 * vLeaf.y;
}
`);
    };
  }
}
