import * as THREE from 'three/webgpu';

const SERIF = 'ui-serif, "New York", "Iowan Old Style", Georgia, "Times New Roman", serif';

const alphaTexture = (context: CanvasRenderingContext2D, texture?: THREE.DataTexture): THREE.DataTexture => {
  const { width, height } = context.canvas, pixels = context.getImageData(0, 0, width, height).data;
  const alpha = texture ? (texture.image.data as Uint8Array) : new Uint8Array(width * height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) alpha[(height - 1 - y) * width + x] = pixels[4 * (y * width + x) + 3]!;
  const out = texture ?? new THREE.DataTexture(alpha, width, height, THREE.RedFormat, THREE.UnsignedByteType);
  out.minFilter = THREE.LinearMipmapLinearFilter;
  out.magFilter = THREE.LinearFilter;
  out.generateMipmaps = true;
  out.needsUpdate = true;
  return out;
};

const pen = (width: number, height: number): CanvasRenderingContext2D => {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('Chess labels need a 2D canvas');
  context.fillStyle = '#ffffff';
  context.textBaseline = 'alphabetic';
  return context;
};

export const GLYPH_COLUMNS = 4;
const GLYPH_PX = 128;

export function glyphAtlas(glyphs: readonly string[]): THREE.DataTexture {
  const rows = Math.ceil(glyphs.length / GLYPH_COLUMNS), context = pen(GLYPH_COLUMNS * GLYPH_PX, rows * GLYPH_PX);
  context.textAlign = 'center';
  context.font = `600 ${0.6 * GLYPH_PX}px ${SERIF}`;
  const cap = context.measureText('8').actualBoundingBoxAscent;
  glyphs.forEach((glyph, i) => context.fillText(glyph, (i % GLYPH_COLUMNS + 0.5) * GLYPH_PX, (Math.floor(i / GLYPH_COLUMNS) + 0.5) * GLYPH_PX + cap / 2));
  return alphaTexture(context);
}

interface Line { text: string; font: string; size: number; baseline: number; align?: CanvasTextAlign; x?: number; spacing?: number }

export class Caption {
  private readonly context: CanvasRenderingContext2D;
  readonly texture: THREE.DataTexture;
  private shown = '';

  constructor(readonly aspect: number, private readonly px = 256) {
    this.context = pen(Math.round(aspect * px), px);
    this.texture = alphaTexture(this.context);
  }

  write(lines: readonly Line[]): void {
    const key = JSON.stringify(lines);
    if (key === this.shown) return;
    this.shown = key;
    const { context, px } = this, width = context.canvas.width;
    context.clearRect(0, 0, width, px);
    for (const line of lines) {
      const set = (scale: number) => {
        context.font = `${line.font} ${line.size * px * scale}px ${SERIF}`;
        context.letterSpacing = `${(line.spacing ?? 0) * px * scale}px`;
      };
      set(1);
      context.textAlign = line.align ?? 'center';
      const room = 0.97 * width, measured = context.measureText(line.text).width;
      if (measured > room) set(room / measured);
      context.fillText(line.text, (line.x ?? 0.5) * width, line.baseline * px);
    }
    alphaTexture(context, this.texture);
  }
}
