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
    Acrobot: 0xf2a7c3, CartPoleSwingUp: 0xf0c25e, MountainCar: 0xe86e4d, Pendulum: 0xffb35e, PuddleWorld: 0x7fc8e8, ScentPuck: 0xe89a5e, Tracking: 0xd9a8ff, Catch: 0xffd166,
    HalfCheetah: 0xffb347, Reacher: 0x6fe0d0, InvertedDoublePendulum: 0xc0a8ff, Hopper: 0xffc07a, Quadrotor2D: 0xa0e6ff, Arm2: 0x9ecbff, MinAtar: 0xff7ad9,
    FourRoomsHall: 0x8fd47a, FourRooms: 0x8fd47a, BlockingMaze: 0xb9c27a, ShortcutMaze: 0xd9c27a, CliffWalk: 0xd97b6c, Taxi: 0xf5d35c, Sokoban: 0xc9a27a, DeepSea: 0x5ee0c8, RiverSwim: 0x6fb8e8,
    Chain: 0xf7a8b8, DelayedChain: 0x7fe0c4, KeyCorridor: 0xffc94d, ProceduralMaze: 0x9fd47a, POMaze: 0xb8cfe0, HighDimMaze: 0xe8975e,
    TMaze: 0x9ff0a8, CountRecall: 0xc99cf0, KeyDoor: 0xe8c66e, Umbrella: 0x8fb8ff, Password: 0xff9ecf, Recite: 0xb8a4ff, Repeat: 0xa8d8ff, Parity: 0xd6b3ff, AliasedRing: 0xff9ab8, Patience: 0xc6e07a, DarkKeyToDoor: 0x8a8fd9,
    Bandit: 0xffc857, Contextual: 0xffa76e, AccessControl: 0xe0b878, Calibration: 0xf2d38a, Operant: 0xf7b26b, Tiger: 0xffb347, RockPaperScissors: 0xe89ac2, CoinSwap: 0xf0d66c, HiddenMode: 0xd4a5ff, Garnet: 0xe0666e, HashedMDP: 0xc7b7ff,
    Nexting: 0x7fd6e8, Regression: 0x86c7f0, Covariate: 0x9ed9d0, Permuted: 0xb0c8ff, SubsetParity: 0xa4e0ff, TraceConditioning: 0xf0a8d8, TracePatterning: 0xe0a0f0, Ring: 0x9ad8c0, Cellular: 0xb5f0a0, WalkBet: 0xffc7a0,
    IDBD: 0x8fc4ff, SlowRegression: 0x7fb0e0, RateTracking: 0x9fd8f0, FrostHollow: 0x9fdcff,
    JellyBeanWorld: 0xee776b, Forager: 0xa8dca0, Craftax: 0x8fcf6a, Dungeon: 0xd98c4a, Homeostasis: 0xf09a9a,
    Chess: 0xe8dcc0, Reversi: 0x5fbf8f, ConnectFour: 0xffd84d, Backgammon: 0xc98a5e, KuhnPoker: 0x6fcf97, LeducPoker: 0x4fb88a, Hanabi: 0xff8fb0, LightCycles: 0x5ef0ff, Pommerman: 0xff9448, SlimeVolley: 0x9ef07a,
    Ancients: 0xd4b06a, Generals: 0xe07a7a, MicroRTS: 0x8ab4ff, Skirmish: 0xe8866e, RelicFog: 0xb9a6d9, Eleven: 0x7ddc8a, OvercookedAI: 0xffa860, LBF: 0xb5d96a, RWARE: 0xf2c14e,
    2048: 0xedc46a, Hanoi: 0xd6a36e, PocketCube: 0xf7e26b, Minesweeper: 0xb0bccc, Tetris: 0x6ad0ff, SymbolicAlchemy: 0xc58cff,
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
