import * as THREE from 'three';
import type { SkyHandle } from '@kit/sky';

export interface AuroraHandle {
  setGain(gain: number): void;
  setTime(t: number): void;
  dispose(): void;
}

export function buildAurora(sky: SkyHandle, color: THREE.ColorRepresentation, azimuth = 0): AuroraHandle {
  const gain = { value: 1 };
  sky.material.uniforms.uAuroraGain = gain;
  sky.material.uniforms.uAuroraColor = { value: new THREE.Color(color) };
  sky.material.uniforms.uAuroraAzimuth = { value: azimuth };
  sky.material.fragmentShader = `uniform float uAuroraGain;\nuniform vec3 uAuroraColor;\nuniform float uAuroraAzimuth;\n${sky.material.fragmentShader}`.replace(
    'gl_FragColor = vec4( col, 1.0 );',
    `float longitude = atan(dir.z,dir.x) + uAuroraAzimuth;
float elevation = asin(clamp(dir.y,-1.,1.));
if(elevation > .008) {
vec3 aurora = vec3(0.);
for(int band=0; band<3; band++) {
  float b = float(band);
  float fold = sin(longitude*2. + b*2.1 + uTime*.019);
  float edge = .072 + b*.048 + fold*.057 + sin(longitude*5. - uTime*.013)*.018;
  float above = elevation-edge;
  float hem = exp(-abs(above)*115.);
  float veil = smoothstep(-.015,.025,above)*exp(-max(above,0.)*28.);
  float rays = .58 + .25*sin(longitude*82.+fold*9.+uTime*.11) + .17*sin(longitude*157.-uTime*.06);
  float reach = pow(.5 + .5*sin(longitude*2.+b*.9),1.6);
  vec3 tint = mix(uAuroraColor,vec3(.64,.3,.61),smoothstep(.03,.13,above));
  aurora += tint*(hem*.28+veil*rays*.5)*reach;
}
col += aurora*uAuroraGain*smoothstep(.008,.028,elevation);
}
gl_FragColor = vec4(col,1.);`,
  );
  sky.material.needsUpdate = true;
  return {
    setGain(value) { gain.value = value; },
    setTime(t) { sky.uniforms.uTime.value = t; },
    dispose() { gain.value = 0; },
  };
}
