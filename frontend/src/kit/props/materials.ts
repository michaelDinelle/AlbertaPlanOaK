import * as THREE from 'three/webgpu';
import { oakMaterial } from '../materials';
import type { WorldBuilder } from '../world';

export function propMaterials(world: WorldBuilder) {
  return {
    sandstone: world.own(oakMaterial({ color: 0xd8c4a0, roughness: 0.92, rimColor: 0xffe1b0, rimStrength: 0.18, toonSteps: 4 })),
    shale: world.own(oakMaterial({ color: 0x7a6a5a, roughness: 0.95, toonSteps: 3 })),
    brass: world.own(new THREE.MeshStandardMaterial({ color: 0xb08a4a, roughness: 0.45, metalness: 0.55 })),
    vine: world.own(new THREE.MeshStandardMaterial({ color: 0x4f7a3c, vertexColors: true, roughness: 0.95 })),
    glass(color: THREE.ColorRepresentation, intensity = 1.2) { return world.own(new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: intensity, roughness: 0.3 })); },
    cloth(color: THREE.ColorRepresentation) { return world.own(oakMaterial({ color, roughness: 1, side: THREE.DoubleSide, wind: { amplitude: 0.05, frequency: 1.4, height: 1 }, toonSteps: 3 })); },
  };
}

export type PropMaterials = ReturnType<typeof propMaterials>;
