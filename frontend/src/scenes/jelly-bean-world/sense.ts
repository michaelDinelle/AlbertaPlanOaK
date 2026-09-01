import * as THREE from 'three';

const RING_LIFT = 0.07;
const RING_INSET = 0.86;
const CHEVRON_INSET = 1.62;

export const SENSE_DIRECTIONS: readonly (readonly [number, number])[] = [
  [0, -1],
  [1, -1],
  [1, 0],
  [1, 1],
  [0, 1],
  [-1, 1],
  [-1, 0],
  [-1, -1],
];

export class SenseRing {
  readonly group = new THREE.Group();

  private readonly cells: THREE.Mesh[] = [];
  private readonly cellMaterials: THREE.MeshBasicMaterial[] = [];
  private readonly nose: THREE.Mesh;
  private readonly noseMaterial: THREE.MeshBasicMaterial;
  private readonly chevron: THREE.Mesh;
  private readonly chevronMaterial: THREE.MeshBasicMaterial;
  private readonly disposables: { dispose(): void }[] = [];

  constructor(accent: THREE.ColorRepresentation) {
    this.group.name = 'orchardSense';
    const cellGeometry = new THREE.PlaneGeometry(RING_INSET, RING_INSET);
    cellGeometry.rotateX(-Math.PI / 2);
    this.disposables.push(cellGeometry);
    for (let i = 0; i < 8; i++) {
      const material = new THREE.MeshBasicMaterial({
        color: accent,
        transparent: true,
        opacity: 0,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        name: 'orchardSenseCell',
      });
      this.disposables.push(material);
      this.cellMaterials.push(material);
      const mesh = new THREE.Mesh(cellGeometry, material);
      mesh.renderOrder = 5;
      mesh.frustumCulled = false;
      this.cells.push(mesh);
      this.group.add(mesh);
    }
    this.noseMaterial = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      name: 'orchardSenseNose',
    });
    this.disposables.push(this.noseMaterial);
    this.nose = new THREE.Mesh(cellGeometry, this.noseMaterial);
    this.nose.renderOrder = 5;
    this.nose.frustumCulled = false;
    this.group.add(this.nose);

    const chevronGeometry = new THREE.BufferGeometry();
    chevronGeometry.setAttribute(
      'position',
      new THREE.BufferAttribute(
        new Float32Array([-0.3, 0, 0.18, 0.3, 0, 0.18, 0, 0, -0.24]),
        3,
      ),
    );
    this.disposables.push(chevronGeometry);
    this.chevronMaterial = new THREE.MeshBasicMaterial({
      color: accent,
      transparent: true,
      opacity: 0,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      name: 'orchardSenseChevron',
    });
    this.disposables.push(this.chevronMaterial);
    this.chevron = new THREE.Mesh(chevronGeometry, this.chevronMaterial);
    this.chevron.renderOrder = 6;
    this.chevron.frustumCulled = false;
    this.group.add(this.chevron);
  }

  update(
    ax: number,
    ay: number,
    heading: number,
    surface: (x: number, z: number) => number,
    energies: Float32Array,
    energyPeak: number,
    scent: readonly [number, number, number],
  ): void {
    for (let i = 0; i < 8; i++) {
      const dir = SENSE_DIRECTIONS[(2 * heading + i) & 7] ?? [0, 0];
      const cell = this.cells[i];
      const material = this.cellMaterials[i];
      if (!cell || !material) continue;
      const wx = ax + dir[0];
      const wy = ay + dir[1];
      cell.position.set(wx, surface(wx, wy) + RING_LIFT, wy);
      const level = (energies[i] ?? 0) / energyPeak;
      material.opacity = 0.05 + 0.5 * level;
    }
    this.nose.position.set(ax, surface(ax, ay) + RING_LIFT, ay);
    const peak = Math.max(scent[0], scent[1], scent[2]);
    if (peak > 0) {
      this.noseMaterial.color.setRGB(scent[0] / peak, scent[1] / peak, scent[2] / peak);
      this.noseMaterial.opacity = Math.min(0.6, 0.1 + 0.25 * Math.log1p(peak));
    } else {
      this.noseMaterial.opacity = 0.05;
    }
    const forward = SENSE_DIRECTIONS[(2 * heading) & 7] ?? [0, -1];
    const fx = ax + forward[0] * CHEVRON_INSET;
    const fy = ay + forward[1] * CHEVRON_INSET;
    this.chevron.position.set(fx, surface(fx, fy) + RING_LIFT * 2, fy);
    this.chevron.rotation.y = Math.atan2(forward[0], forward[1]) + Math.PI;
    this.chevronMaterial.opacity = 0.85;
  }

  dispose(): void {
    this.group.removeFromParent();
    this.group.clear();
    for (const d of this.disposables) d.dispose();
    this.disposables.length = 0;
  }
}
