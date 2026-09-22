const scratch = new DataView(new ArrayBuffer(4));

export function wordFloat(word: number): number {
  scratch.setUint32(0, word >>> 0, true);
  return scratch.getFloat32(0, true);
}

export const wordInt = (word: number): number => word | 0;
