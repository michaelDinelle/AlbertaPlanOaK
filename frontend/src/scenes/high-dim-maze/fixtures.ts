// The Slot Canyons — canonical poses.
//
// Six states that between them exercise every accuracy claim the scene makes: the
// deterministic fleet-init pose, the agent standing ON each of the two gap cells (the
// arches — the only free cells inside a fin), the sunlit spine of the wide slot, a STAY
// (action 4) and a two-cell DASH (action 5).
//
// The last two matter even though the demo recording contains neither: the env defines
// six actions, `oak_stage_sample_act_discrete` only ever samples the first four (its
// Gumbel draw indexes a 4-word Philox block, so d = 4 and 5 read out of bounds and lose
// every time), and the scene must still be right for the env rather than for one damaged
// sampler. These fixtures are how that is checked.
//
// Every cell here is checked free by the env's own `highDimMazeIsWall`; the test asserts it.

import type { SceneFixture } from '@kit/types';
import {
  HIGHDIMMAZE_DASH,
  HIGHDIMMAZE_GAPS,
  HIGHDIMMAZE_GOAL,
  HIGHDIMMAZE_STAY,
  HIGHDIMMAZE_START,
  type HighDimMazeState,
} from '@sim/envs/highdimmaze';

/** pose = [x, y, gx, gy, lastDir, 0…] — the record-v3 HighDimMaze packing. */
function pose(x: number, y: number, gx: number, gy: number, lastDir: number): number[] {
  const p = new Array<number>(9).fill(0);
  p[0] = x;
  p[1] = y;
  p[2] = gx;
  p[3] = gy;
  p[4] = lastDir;
  return p;
}

/** ctrl[0] is the discrete action index (0..5); the other five floats are unused. */
function ctrl(action: number): number[] {
  const c = new Array<number>(6).fill(0);
  c[0] = action;
  return c;
}

function derived(x: number, y: number, gx: number, gy: number, lastDir: number): HighDimMazeState {
  return { x, y, gx, gy, lastDir };
}

const LOW_ARCH = HIGHDIMMAZE_GAPS[0] ?? { x: 3, y: 9 };
const HIGH_ARCH = HIGHDIMMAZE_GAPS[1] ?? { x: 7, y: 1 };

export const fixtures: SceneFixture[] = [
  {
    // oak_fleet_init: agent on (1,1) at the head of the first slot, goal on (9,9),
    // lastDir = 1 (down) — the direction the kernel initialises every lane to
    name: 'canyon-start',
    pose: pose(HIGHDIMMAZE_START.x, HIGHDIMMAZE_START.y, HIGHDIMMAZE_GOAL.x, HIGHDIMMAZE_GOAL.y, 1),
    ctrl: ctrl(1),
    derived: derived(HIGHDIMMAZE_START.x, HIGHDIMMAZE_START.y, HIGHDIMMAZE_GOAL.x, HIGHDIMMAZE_GOAL.y, 1),
    step: 0,
  },
  {
    // Standing IN the low arch (3,9) — a free cell inside the western fin, heading east.
    // From the composed camera the wisp reads under the span, in the notch the fin erodes
    // down to either side of it; the ember is never fully hidden (the visibility audit in
    // the test file measures exactly that, on this cell and the other 64).
    name: 'low-arch',
    pose: pose(LOW_ARCH.x, LOW_ARCH.y, 6, 2, 3),
    ctrl: ctrl(3),
    derived: derived(LOW_ARCH.x, LOW_ARCH.y, 6, 2, 3),
    step: 1841,
  },
  {
    // Standing IN the high arch (7,1) — the second gap, the far end of the serpentine, and
    // the HARDEST cell in the maze to see: it sits on the fin's own line with the outer rim
    // right behind it, so the sight-line arrives along the fin and the span's own leg stands
    // in front of the orb. It is the cell that set the fin ramp, the arch ribbon and the
    // rig's yaw limits; what this fixture shows is the ember reading past that leg, which is
    // the least of the 65 and still not nothing.
    name: 'high-arch',
    pose: pose(HIGH_ARCH.x, HIGH_ARCH.y, 2, 6, 3),
    ctrl: ctrl(3),
    derived: derived(HIGH_ARCH.x, HIGH_ARCH.y, 2, 6, 3),
    step: 2273,
  },
  {
    // the sunlit spine: (5,5) is the ONLY column in the maze two cells clear of any wall
    name: 'sunlit-spine',
    pose: pose(5, 5, 9, 9, 0),
    ctrl: ctrl(0),
    derived: derived(5, 5, 9, 9, 0),
    step: 4066,
  },
  {
    // action 4 = stay: the wisp holds station, the scarf goes slack, sand sifts down
    name: 'settled',
    pose: pose(8, 4, 1, 7, 1),
    ctrl: ctrl(HIGHDIMMAZE_STAY),
    derived: derived(8, 4, 1, 7, 1),
    step: 5602,
  },
  {
    // action 5 = dash: two cells north up the middle slot inside one 8 Hz step
    name: 'gust-sprint',
    pose: pose(5, 3, 2, 2, 0),
    ctrl: ctrl(HIGHDIMMAZE_DASH),
    derived: derived(5, 3, 2, 2, 0),
    step: 8873,
  },
];

export default fixtures;
