// FROZEN Phase-0 contract: the design tokens that make 16 parallel scenes one family.
// Every color in every scene comes from here (or is a scene-local *blend* of these).
// Additive changes only after Phase 0.

export type PaletteName = 'dawn' | 'morning' | 'noon' | 'afternoon' | 'dusk' | 'night' | 'underwater' | 'abyss' | 'cave';

export interface Palette {
  skyTop: number;
  skyHorizon: number;
  skyBottom: number;
  sun: number;
  sunIntensity: number;      // DirectionalLight intensity
  hemiSky: number;
  hemiGround: number;
  hemiIntensity: number;
  fog: number;
  /** fraction of scene extent where fog reaches full density (FogExp2 hint) */
  fogDensityHint: number;
  groundA: number;           // terrain/floor color ramp low
  groundB: number;           // terrain/floor color ramp high
  accentLight: number;       // default PointLight tint
}

export interface DesignTokens {
  palettes: Record<PaletteName, Palette>;
  /** per-env accent, keyed by env slug (shared/envs.ts) */
  accents: Record<string, number>;
  celebrationGold: number;
  uiCyan: number;
  failureBlue: number;
  cueCyan: number;           // MemoryChain cue 0
  cueMagenta: number;        // MemoryChain cue 1
  /** ACESFilmicToneMapping exposure — fixed globally, scenes must not change it */
  exposure: number;
  wispHeight: number;        // shared avatar height in world units
  cellSize: number;          // 1 grid cell = 1 world unit
  fovHero: number;           // hero-tier camera FOV degrees (35–45 allowed; this is the default)
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
    dawn:       P(0x8fb6d9, 0xf6c9a0, 0xe8a87c, 0xffe3b3, 2.2, 0xbcd4e6, 0x6b7f5e, 0.55, 0xdBC2a6, 0.45, 0x7a8f66, 0xb8c98a, 0xffd9a0),
    morning:    P(0x7ec1e8, 0xd9ecf5, 0xf2efd8, 0xfff2cc, 2.6, 0xcfe8f5, 0x77885f, 0.6,  0xd9e8ee, 0.35, 0x86a05e, 0xc4d67e, 0xfff0c2),
    noon:       P(0x4f9fe0, 0xa8d8f0, 0xeef7dc, 0xffffff, 3.0, 0xbfe3f7, 0x7d915e, 0.65, 0xcfe6f0, 0.25, 0x7fa356, 0xd6e394, 0xffffff),
    afternoon:  P(0x5da2d6, 0xc6dfae, 0xe8dfb0, 0xfff3d6, 2.7, 0xc9e2ea, 0x83905c, 0.6,  0xd8e2c2, 0.3,  0x8aa45e, 0xd9d98a, 0xffedc2),
    dusk:       P(0x4a5d8f, 0xe89b5e, 0xd97b4a, 0xffb347, 2.0, 0xa4a8d4, 0x6e5a44, 0.5,  0xd9a577, 0.5,  0x9c7d4e, 0xd9a55e, 0xffc98a),
    night:      P(0x0b1030, 0x24356b, 0x18244d, 0xbfd4ff, 0.5, 0x2c3d70, 0x141c33, 0.4,  0x121a38, 0.6,  0x1d2b4a, 0x33456e, 0x9fc2ff),
    underwater: P(0x2a8fc4, 0x1a6ba3, 0x0d3f70, 0xbfe8ff, 1.6, 0x5eb8dd, 0x0e3a5c, 0.6,  0x1c6395, 0.7,  0x1f6f8f, 0x3fa3ab, 0x8fe0ff),
    abyss:      P(0x0d2340, 0x081830, 0x030a18, 0x4a7ba6, 0.35, 0x14324f, 0x040d1a, 0.45, 0x061224, 0.85, 0x0a1e33, 0x1a3a52, 0x7fd4e8),
    cave:       P(0x0a0d16, 0x131a2b, 0x0a0e1a, 0x6a7ba0, 0.2,  0x1a2440, 0x0a0d16, 0.35, 0x0b0f1d, 0.75, 0x1a2233, 0x2e3a52, 0xaad4ff),
  },
  accents: {
    'half-cheetah':       0xffb347, // savanna amber
    'four-rooms':         0x8fd47a, // garden green
    'deep-sea':           0x5ee0c8, // reef teal
    'key-corridor':       0xffc94d, // amber seed-key
    'acrobot':            0xf2a7c3, // dawn blossom
    'mountain-car':       0xe86e4d, // alpenglow ember
    'cart-pole':          0xa9d954, // young-leaf green
    'pendulum':           0xffb35e, // lantern warm
    'chain':              0xf7a8b8, // blossom pink
    'large-deep-sea':     0x7fd4e8, // biolume ice-blue
    'multi-key-corridor': 0xd9b36a, // temple gold
    'memory-chain':       0x9ff0a8, // firefly green
    'procedural-maze':    0x9fd47a, // moss
    'high-dim-maze':      0xe8975e, // terracotta
    'po-maze':            0xb8cfe0, // fog silver
    'delayed-chain':      0x7fe0c4, // aurora teal
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
