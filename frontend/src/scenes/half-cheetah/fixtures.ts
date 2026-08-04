// Gold Hour Savanna — canonical poses.
//
// Every one of these is a REAL qpos9 lifted out of the demo run's own HalfCheetah lane
// (`frontend-demo-16env-20260803`, seed 0), except `reset-drop`, which is the decoder's
// synthesized frame −1 for that same seed. Between them they exercise every claim the
// scene makes: feet in the air, belly-down flailing, a full extended stride at 0.96 m/s,
// a nose-down stumble, and — the one that matters for a treadmill — a pose 431 metres
// down the plain, where f32 world coordinates have to keep the toes on the ground.

import type { SceneFixture } from '@kit/types';
import { CHEETAH_DT, type CheetahState } from '@sim/envs/halfcheetah';
import { ROOT_Z0 } from '@sim/fk/cheetah';

function state(pose: readonly number[], velX: number): CheetahState {
  const qpos = new Float32Array(9);
  for (let i = 0; i < 9; i++) qpos[i] = pose[i] ?? 0;
  return {
    qpos,
    rootX: qpos[0] ?? 0,
    rootZ: ROOT_Z0 + (qpos[1] ?? 0),
    pitch: qpos[2] ?? 0,
    velX,
  };
}

function fixture(name: string, step: number, pose: number[], ctrl: number[], velX: number): SceneFixture {
  return { name, pose, ctrl, derived: state(pose, velX), step };
}

/** frame −1: `HC_QPOS0 + U(-0.1, 0.1)^9` for lane seed 0 — dropped, feet clear of the ground */
const RESET_POSE = [
  -0.020190715789794922, 0.07610403746366501, 0.04714255779981613, 0.0210963636636734,
  -0.01655038446187973, 0.08292048424482346, -0.009654700756072998, -0.09430766105651855,
  -0.03047356754541397,
];

export const fixtures: SceneFixture[] = [
  // the deterministic init draw: airborne, about to land. Toes at z ≈ 0.22.
  // (its velX is the reset draw's own qvel[0] — the one frame that knows its true speed)
  fixture('reset-drop', -1, RESET_POSE, [0, 0, 0, 0, 0, 0], 0.031030923128128052),

  // step 617: the early stream. Belly low, both toes planted, barely creeping forward.
  fixture(
    'flail-sprawl',
    617,
    [
      -0.08328090608119965, -0.14171047508716583, 0.05032094195485115, 0.09538014978170395,
      0.06836573034524918, -0.09387730062007904, -0.00908216368407011, -0.19277887046337128,
      -0.13347908854484558,
    ],
    [
      0.26014187932014465, 0.08511065691709518, -0.06522200256586075, 0.05277609825134277,
      -0.12780967354774475, -0.1088983565568924,
    ],
    0.2486179769039154,
  ),

  // step 52025: THE hero pose, and the manifest's thumbFrame. 0.97 m/s with the back toe
  // PLANTED (0.044, one geom radius = standing on the ground) and the front leg swinging
  // clear at 0.185 — a real contact frame, which is what makes the dust and the shadow
  // line up in the gallery thumbnail.
  fixture(
    'stride-plant',
    52025,
    [
      201.27566528320312, -0.08060205727815628, -0.08686039596796036, 0.20897798240184784,
      0.10743971914052963, -0.14246860146522522, 0.08087541162967682, -0.22010691463947296,
      -0.12991316616535187,
    ],
    [0, 0, 0, 0, 0, 0],
    0.966,
  ),

  // step 51472: mid-bound — BOTH feet clear of the ground at 0.96 m/s. The frame that
  // proves the scene never fakes contact: no dust, no scuff, shadow only.
  fixture(
    'stride-extend',
    51472,
    [
      189.5414581298828, -0.10307499021291733, -0.054662074893713, 0.25581443309783936,
      0.12558843195438385, 0.07066120952367783, -0.01605881191790104, -0.10866694152355194,
      0.04460061714053154,
    ],
    [
      0.3684020936489105, -0.011527961120009422, 0.5995810627937317, 0.16003301739692688,
      0.15079060196876526, 0.1936870962381363,
    ],
    0.95977783203125,
  ),

  // step 49693: a stumble — nose pitched 12° down, sliding backwards, back foot high.
  fixture(
    'nose-dip',
    49693,
    [
      158.67002868652344, -0.16460973024368286, 0.2164866328239441, -0.1697898954153061,
      -0.053383827209472656, -0.3772735893726349, 0.6151124238967896, 0.19018620252609253,
      0.193485826253891,
    ],
    [
      -0.33811426162719727, -0.09804369509220123, -0.9219779372215271, 0.23864629864692688,
      -0.3236398994922638, -0.2848738729953766,
    ],
    -0.2606201171875,
  ),

  // step 99999: 431 m down the treadmill, reared up at 56° with the back leg in the air.
  // If the world snap or the f32 pose ever drifted, this is the fixture that shows it.
  fixture(
    'far-field-rear',
    99999,
    [
      431.10174560546875, -0.17006294429302216, 0.9814415574073792, -0.021347766742110252,
      -0.021117085590958595, -0.013453810475766659, 0.38369736075401306, 0.18500185012817383,
      0.4280204474925995,
    ],
    [
      -0.1762012392282486, -0.058001890778541565, -0.08507302403450012, 0.16246499121189117,
      -0.08525727689266205, 0.28236696124076843,
    ],
    -0.0982666015625,
  ),
];

/** the control-step the velocities above are measured over (documented, not re-derived) */
export const FIXTURE_DT = CHEETAH_DT;

export default fixtures;
