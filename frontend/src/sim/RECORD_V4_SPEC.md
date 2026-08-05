# Record format v4 — FROZEN codec specification

The single source of truth for the Swift encoder (`Sources/Main.swift`) and the TS
decoder (`frontend/src/sim/record4.ts`). Both implement THIS document; the Tier-S
v3/v4 A/B run + `frontend/scripts/verify-record-v4.ts` prove agreement. Any change
bumps `codecVersion`.

All integers little-endian. All floats IEEE-754 LE.

## Goals / non-goals

- ~22× smaller than v3 (measured 2.73 GiB for 16 envs × 64 lanes × 1M steps).
- Decodes to the EXACT v3 `RawRecording` row shape (`f32[steps][16]`: ctrl[6],
  done, pose[9]) so every env decoder, the MultiKeyCorridor bitwise replay, DeepSea
  event simulation, and damaged-row semantics run unchanged.
- Action indices and done bits are bit-exact. Quantized floats are within ½ quantum.
  Anything outside contract is ESCAPED verbatim (bit-exact 64-byte v3 row).
- The kernel ring write is untouched; the host drain transcodes. Recording stays
  passive (Inv-10 bit-identity of the agent stream, proven by state-hash `cmp`).

## File layout

```
FileHeader (16 B):
  u32 magic        = 0x344B414F  ("OAK4" as bytes O,A,K,4)
  u32 formatVersion = 4
  u32 envKind       (Sources/Main.swift envKind(_:) numbering, 0..15)
  u32 codecVersion  = 1

Chunk (repeated, one per record-drain interval, append-only):
  u32 magic      = 0x344B4843   ("CHK4")
  u32 stepStart                  (0-based step index of first row in chunk)
  u32 stepCount
  u32 rawBytes                   (uncompressed payload size)
  u32 compBytes                  (compressed payload size)
  u32 crc32                      (zlib crc32 over the COMPRESSED payload bytes)
  u32 doneCount                  (count of done bits set in this chunk's rows)
  f32 rewardDelta                (lane rewardSum delta over this chunk from the host
                                  stats snapshot; NaN when unavailable)
  ...compBytes bytes             (raw DEFLATE — Swift NSData .zlib / browser
                                  DecompressionStream('deflate-raw') / Bun inflateRawSync)
```

Chunk header total: 32 B. A torn tail (killed run) is detected by: short header,
`compBytes` bytes missing, crc mismatch, or inflated size ≠ rawBytes → the reader
drops that chunk and everything after it. `steps` = Σ stepCount of valid chunks.

## meta.json (per lane)

```json
{ "envId": "...", "seed": N, "actDim": 6, "recordVersion": 4, "codecVersion": 1,
  "codec": { /* the descriptor below, verbatim from the static table */ },
  "steps": 1000000,
  "chunks": [[stepStart, stepCount, byteOffset, byteLength], ...] }
```

`chunks` (byteOffset = offset of the chunk header from file start; byteLength =
32 + compBytes) is written at finalize; it is a HINT — readers of in-flight or
torn files fall back to a sequential header scan.

## Payload (uncompressed, columnar, in this exact column order)

1. `hdr` — u8[stepCount]. bit0-3 = action index (discrete envs; 0 for Box envs);
   bit4 = done; bit5 = escape; bits6-7 = 0.
2. Box-env ctrl columns (cheetah: 6 columns; pendulum: 1 column; others: none) —
   u8[stepCount] each, `q = round((clamp(v, -1, 1) + 1) * 127.5)`,
   dequant `v = q / 127.5 - 1`.
3. Cheetah only: `chunkBaseX` f64 = the EXACT f64 promotion of the first row's
   pose[0] (f32). Then the dx column appears in pose-column position per the
   descriptor.
4. Pose columns, one per `poseFields` entry, in descriptor order:
   - `u8`: u8[stepCount], value stored as-is (small non-negative ints).
   - `i16`: i16[stepCount], `q = round(v / scale)`; if `wrap`, first wrap v into
     [−π, π) via `w = v − 2π·floor((v + π) / (2π))` (floor-based positive wrap —
     identical result in Swift and JS). Dequant `v = q * scale`.
   - `dx32`: i32[stepCount], chained deltas at scale s=0.01:
     `recon[-1] = chunkBaseX; q[i] = round((v[i] − recon[i−1]) / s);
      recon[i] = recon[i−1] + q[i]·s` (recon in f64). Dequant is the same chain.
5. Escape table: u32 count, then count × (u32 rowIdxInChunk + 64 bytes verbatim
   v3 row, f32[16]).

Escape rule (encoder): a row is escaped iff any of its 16 f32s is non-finite, OR
any quantized field would fall outside its integer range (u8: not an integer in
[0,255]; i16: |q| > 32767; dx32: |q| > 2^31−1), OR a discrete action index is not
an integer in [0,15]. Escaped rows are excluded from the quantized columns'
reconstruction chains? — NO: for simplicity the columns still carry a value for
escaped rows (encoder writes the CLAMPED/zeroed quantization, decoder overwrites
the whole row from the escape table afterwards; the dx32 chain uses the clamped
value so the chain stays deterministic for both sides).

## Decode → v3 row shape

For each non-escaped row: `ctrl[0] = float(actionIndex)` (discrete envs) or the
dequantized ctrl columns (Box envs); remaining ctrl = 0. `row[6] = done ? 1.0 : 0.0`.
Pose fields from columns per descriptor; `poseConst` entries filled; unlisted pose
indices = 0. All values stored into f32 (Math.fround semantics).
Escaped rows: the verbatim 64 bytes replace the entire row.

## Static codec table (embedded in BOTH implementations; meta carries it too)

Category `grid` (hdr + u8 pose columns; action bits used):
| envKind | envId | poseFields (idx:enc) | poseConst |
|---|---|---|---|
| 1 | FourRooms | 0:u8 1:u8 2:u8 3:u8 | — |
| 2 | DeepSea | 0:u8 1:u8 | {2:7, 3:7} |
| 3 | KeyCorridor | 0..7:u8 | — |
| 8 | Chain | 0:u8 | {1:15} |
| 9 | LargeDeepSea | 0:u8 1:u8 | {2:15, 3:15} |
| 10 | MultiKeyCorridor | 0..7:u8 | — |
| 11 | MemoryChain | 0:u8 2:u8 | {1:31} |
| 12 | ProceduralMaze | 0:u8 1:u8 2:u8 3:u8 | — |
| 13 | HighDimMaze | 0:u8 1:u8 2:u8 3:u8 4:u8 | — |
| 14 | POMaze | 0:u8 1:u8 2:u8 3:u8 | — |
| 15 | DelayedChain | 0:u8 2:u8 | {1:31} |

Category `classic` (hdr + i16 columns; Acrobot/MountainCar/CartPole use action
bits; Pendulum uses one ctrl u8 column):
| envKind | envId | poseFields (idx:enc:scale[:wrap]) | poseConst |
|---|---|---|---|
| 4 | Acrobot | 0:i16:1e-4:wrap 1:i16:1e-4:wrap 2:i16:1e-3 3:i16:1e-3 | — |
| 5 | MountainCar | 0:i16:1e-4 1:i16:1e-5 | {2:0.5} |
| 6 | CartPole | 0:i16:1e-4 1:i16:1e-3 2:i16:1e-4:wrap 3:i16:1e-3 | — |
| 7 | Pendulum | 0:i16:1e-4:wrap 1:i16:1e-3 2:i16:1e-3 | — |

Category `cheetah` (hdr with action bits 0 + 6 ctrl u8 columns + chunkBaseX):
| envKind | envId | poseFields |
|---|---|---|
| 0 | HalfCheetah | 0:dx32:0.01, 1:i16:1e-4, 2:i16:1e-4:wrap, 3..8:i16:1e-4:wrap |

Descriptor JSON shape (what meta.json `codec` carries):
```json
{ "category": "grid|classic|cheetah",
  "ctrl": "index" | {"kind": "u8box", "n": 1|6},
  "poseFields": [ {"idx":0,"enc":"u8"} |
                  {"idx":2,"enc":"i16","scale":1e-4,"wrap":true} |
                  {"idx":0,"enc":"dx32","scale":0.01} ],
  "poseConst": {"2": 7, "3": 7} }
```

## Precision contract (display-exact; justified against consumers)

- Discrete actions, done: bit-exact (small ints exact in u8→f32).
- Angles 1e-4 rad (wrapped — FK/scenes read sin/cos so wrapping is lossless);
  rootx 1 cm (invisible at scene scale, far below CHEETAH_TELEPORT_DX=10);
  velocities 1e-3 (1e-5 MountainCar, range ±0.07); ctrl ±1/255 (display glow only).
- Non-finite rows (diverged lanes) are escaped verbatim → `decode.ts` damaged-row
  semantics (nonFiniteRows count, replay freeze) behave EXACTLY as with v3.
