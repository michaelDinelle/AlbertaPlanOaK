export type PaletteName = 'dawn' | 'morning' | 'noon' | 'afternoon' | 'dusk' | 'night' | 'underwater' | 'abyss' | 'cave';

export interface Palette {
  skyTop: number;
  skyHorizon: number;
  skyBottom: number;
  sun: number;
  sunIntensity: number;
  hemiSky: number;
  hemiGround: number;
  hemiIntensity: number;
  fog: number;
  fogDensityHint: number;
  groundA: number;
  groundB: number;
  accentLight: number;
}

export interface DesignTokens {
  palettes: Record<PaletteName, Palette>;
  accents: Record<string, number>;
  celebrationGold: number;
  uiCyan: number;
  failureBlue: number;
  cueCyan: number;
  cueMagenta: number;
  exposure: number;
  wispHeight: number;
  cellSize: number;
  fovHero: number;
}

const P = (
  skyTop: number, skyHorizon: number, skyBottom: number,
  sun: number, sunIntensity: number,
  hemiSky: number, hemiGround: number, hemiIntensity: number,
  fog: number, fogDensityHint: number,
  groundA: number, groundB: number, accentLight: number,
): Palette => ({
  skyTop, skyHorizon, skyBottom, sun, sunIntensity,
  hemiSky, hemiGround, hemiIntensity, fog, fogDensityHint,
  groundA, groundB, accentLight,
});

export const TOKENS: DesignTokens = {
  palettes: {
    dawn:       P(0x8fb6d9, 0xf6c9a0, 0xe8a87c, 0xffe3b3, 2.2, 0xbcd4e6, 0x6b7f5e, 0.55, 0xdBC2a6, 0.55, 0x7a8f66, 0xb8c98a, 0xffd9a0),
    morning:    P(0x7ec1e8, 0xd9ecf5, 0xf2efd8, 0xfff2cc, 2.6, 0xcfe8f5, 0x77885f, 0.6,  0xd9e8ee, 0.68, 0x86a05e, 0xc4d67e, 0xfff0c2),
    noon:       P(0x4f9fe0, 0xa8d8f0, 0xeef7dc, 0xffffff, 3.0, 0xbfe3f7, 0x7d915e, 0.65, 0xcfe6f0, 0.88, 0x7fa356, 0xd6e394, 0xffffff),
    afternoon:  P(0x5da2d6, 0xc6dfae, 0xe8dfb0, 0xfff3d6, 2.7, 0xc9e2ea, 0x83905c, 0.6,  0xd8e2c2, 0.72, 0x8aa45e, 0xd9d98a, 0xffedc2),
    dusk:       P(0x4a5d8f, 0xe89b5e, 0xd97b4a, 0xffb347, 2.0, 0xa4a8d4, 0x6e5a44, 0.5,  0xd9a577, 0.5,  0x9c7d4e, 0xd9a55e, 0xffc98a),
    night:      P(0x0b1030, 0x24356b, 0x18244d, 0xbfd4ff, 0.5, 0x2c3d70, 0x141c33, 0.4,  0x121a38, 0.6,  0x1d2b4a, 0x33456e, 0x9fc2ff),
    underwater: P(0x2a8fc4, 0x1a6ba3, 0x0d3f70, 0xbfe8ff, 1.6, 0x5eb8dd, 0x0e3a5c, 0.6,  0x1c6395, 0.7,  0x1f6f8f, 0x3fa3ab, 0x8fe0ff),
    abyss:      P(0x11294a, 0x0a1d38, 0x05101f, 0x6f9fc9, 0.62, 0x1d4266, 0x061224, 0.62, 0x08182e, 0.85, 0x123457, 0x245579, 0x9fe2f5),
    cave:       P(0x0a0d16, 0x131a2b, 0x0a0e1a, 0x6a7ba0, 0.2,  0x1a2440, 0x0a0d16, 0.35, 0x0b0f1d, 0.75, 0x1a2233, 0x2e3a52, 0xaad4ff),
  },
  accents: {
    foliage: 0xa9d954,
    acrobot: 0xf2a7c3, cartpole: 0xf0c25e, mountainCar: 0xe86e4d, pendulum: 0xffb35e, puddle: 0x7fc8e8, scentpuck: 0xe89a5e, tracking: 0xd9a8ff, catch: 0xffd166,
    hall: 0x8fd47a, blocking: 0xb9c27a, cliff: 0xd97b6c, taxi: 0xf5d35c, sokoban: 0xc9a27a, deepsea: 0x5ee0c8, riverswim: 0x6fb8e8, walk: 0xf7a8b8,
    tmaze: 0x9ff0a8, count: 0xc99cf0, keydoor: 0xe8c66e, umbrella: 0x8fb8ff, password: 0xff9ecf, recite: 0xb8a4ff, repeat: 0xa8d8ff, parity: 0xd6b3ff, aliased: 0xff9ab8, patience: 0xc6e07a,
    bandit: 0xffc857, contextual: 0xffa76e, access: 0xe0b878, calibration: 0xf2d38a, operant: 0xf7b26b, tiger: 0xffb347, rps: 0xe89ac2, coinswap: 0xf0d66c, hiddenmode: 0xd4a5ff, garnet: 0xe0666e, hashed: 0xc7b7ff,
    nexting: 0x7fd6e8, regression: 0x86c7f0, covariate: 0x9ed9d0, permuted: 0xb0c8ff, subsetparity: 0xa4e0ff, conditioning: 0xf0a8d8, ring: 0x9ad8c0, cellular: 0xb5f0a0, walkbet: 0xffc7a0,
    jbw: 0xee776b, forager: 0xa8dca0,
  },
  celebrationGold: 0xffd166,
  uiCyan: 0x5ec1ff,
  failureBlue: 0x7a95c4,
  cueCyan: 0x5ee0ff,
  cueMagenta: 0xff5ec8,
  exposure: 1.0,
  wispHeight: 0.35,
  cellSize: 1.0,
  fovHero: 40,
};
