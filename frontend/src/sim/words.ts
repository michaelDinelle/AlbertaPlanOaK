const scratch = new DataView(new ArrayBuffer(4));

export function wordFloat(word: number): number {
  scratch.setUint32(0, word >>> 0, true);
  return scratch.getFloat32(0, true);
}

export const wordInt = (word: number): number => word | 0;

const wide = new DataView(new ArrayBuffer(8));

export function wordsDouble(words: Uint32Array, at: number): number {
  wide.setUint32(0, words[at]! >>> 0, true);
  wide.setUint32(4, words[at + 1]! >>> 0, true);
  return wide.getFloat64(0, true);
}
