// The Firefly Cave — canonical poses.
//
// Six states that between them exercise every claim the scene makes: the env's own
// deterministic init (cell 0, cue 0), the same cell with the OTHER cue (the crystal must
// change colour and nothing else may), a wisp deep in the passage carrying a cue it can no
// longer see (the honesty case — the crystal must be utterly dark), the choice cell one
// step before it commits, and BOTH endings of that choice: the respawn after a correct pick
// and the respawn after a wrong one.
//
// PROVENANCE, verified row by row against
// `.oak-benchmarks/frontend-demo-16env-20260803/MemoryChain-cont-v0/seed88/actions.bin`
// (record-v3: `ctrl[6], done, pose[9]`, and pose = [pos, 31, cue]). `step` is the record
// ROW each pose is copied from, except `cue-cyan-lit`, whose step is −1 because it is the
// decoder's own synthesized init frame (`memoryChainDecoder.initFrame()`) — row 0 of the
// recording is already pos 1, so there is no row that carries the initial state.
//
// The first pass cited rows it had not read back: it claimed row 0 for pos 0 (row 0 is
// pos 1), row 12104 for pos 0 (that row is pos 1) and row 13040 for pos 14 (that row is
// pos 17), and claimed to ship "both endings" while shipping only the loss. Every step
// number below is now a row whose decoded state IS the pose drawn, and the winning ending
// is a shipped pose rather than a synthetic event in the test file.

import type { SceneFixture } from '@kit/types';
import { MEMORYCHAIN_END } from '@sim/envs/memorychain';
import { memoryChainDerived, memoryChainPose } from './manifest';

/** ctrl[0] is the discrete action bit (0 = left/back, 1 = right/forward). */
function ctrl(action: number): number[] {
  const c = new Array<number>(6).fill(0);
  c[0] = action;
  return c;
}

export const fixtures: SceneFixture[] = [
  {
    // the env's own deterministic init: the wisp on cell 0 with the cue drawn cyan — one of
    // only two frames in which the crystal is allowed to be cyan
    name: 'cue-cyan-lit',
    pose: memoryChainPose(0, 0),
    ctrl: ctrl(0),
    derived: memoryChainDerived(0, 0, 0),
    step: -1,
  },
  {
    // the same cell, the other draw: the crystal retints magenta and NOTHING else changes.
    // Row 12105 — the step after the correct choice at 12103 walked the wisp back to 0.
    name: 'cue-magenta-lit',
    pose: memoryChainPose(0, 1),
    ctrl: ctrl(0),
    derived: memoryChainDerived(0, 1, 0),
    step: 12105,
  },
  {
    // mid-passage, carrying a cue it can no longer see: the crystal must be utterly dark
    // and no cue colour may exist anywhere in the frame except the two mouths
    name: 'carrying-the-cue',
    pose: memoryChainPose(14, 1),
    ctrl: ctrl(1),
    derived: memoryChainDerived(14, 1, 1),
    step: 12312,
  },
  {
    // standing on the choice cell one step before it commits, the cue still magenta
    name: 'at-the-choice',
    pose: memoryChainPose(MEMORYCHAIN_END, 1),
    ctrl: ctrl(1),
    derived: memoryChainDerived(MEMORYCHAIN_END, 1, 1),
    step: 13122,
  },
  {
    // ...and the very next row: the magenta choice was RIGHT. pos is already back at 0, the
    // cue has been redrawn cyan, and ctrl[0] is the action that won — which is what the
    // celebration reads its colour from
    name: 'correct-choice-respawn',
    pose: memoryChainPose(0, 0),
    ctrl: ctrl(1),
    derived: memoryChainDerived(0, 0, 1),
    step: 13123,
  },
  {
    // the respawn frame of a WRONG choice: pos is already back at 0, the cue has already
    // been redrawn, and ctrl[0] is the action that lost
    name: 'wrong-choice-respawn',
    pose: memoryChainPose(0, 1),
    ctrl: ctrl(1),
    derived: memoryChainDerived(0, 1, 1),
    step: 16155,
  },
];

export default fixtures;
