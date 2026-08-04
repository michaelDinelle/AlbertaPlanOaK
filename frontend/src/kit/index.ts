// kit — the single import surface for the nature kit.
//
// Scenes import from `@kit/index` (or a specific module when they want the smallest
// possible chunk). Everything a scene is allowed to use is re-exported here; anything
// NOT exported here is host machinery.
//
// Module map:
//   tokens      the palettes, accents and world constants — every color starts here
//   types       the frozen SceneModule / SceneContext / FrameUpdate contract
//   rng         the ONLY randomness (seeded with ctx.seed)
//   textures    procedural DataTextures (fbm, voronoi, ramps, grain, sprites)
//   materials   oakMaterial / emissiveMaterial / wispMaterial — the stylization language
//   sky         the one sky (and its underwater/cave depth-column parameterization)
//   lighting    the light budget + fog
//   terrain     displaced ground with the sampleHeight accuracy contract
//   post        the HOST post chain (scenes only parameterize it via manifest.postConfig)
//   grid        THE accuracy backbone for the 10 grid envs + HopMotion
//   wisp        THE shared avatar
//   vegetation  instanced grass / trees / rocks / flowers
//   particles   ambient particle presets + mist planes
//   water       the one water material
//   celebrate   the shared reward grammar + the host's PulseChannel
//   rigs        OrbitRig / TrackRig / FrameRig
//   debug       the GroundTruthOverlay (the QA instrument)
//   harness     the artisan dev page (host machinery — imported by views/harness.ts)

export * from './tokens';
export * from './types';
export * from './rng';
export * from './textures';
export * from './materials';
export * from './sky';
export * from './lighting';
export * from './terrain';
export * from './post';

export * from './grid';
export * from './wisp';
export * from './vegetation';
export * from './particles';
export * from './water';
export * from './celebrate';
export * from './rigs';
export * from './debug';
