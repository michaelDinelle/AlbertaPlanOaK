import { accessDecoder } from '@sim/envs/access';
import { acrobotDecoder } from '@sim/envs/acrobot';
import { aliasedDecoder } from '@sim/envs/aliased';
import { banditDecoder } from '@sim/envs/bandit';
import { blockingDecoder } from '@sim/envs/blocking';
import { calibrationDecoder } from '@sim/envs/calibration';
import { cartpoleDecoder } from '@sim/envs/cartpole';
import { catchDecoder } from '@sim/envs/catch';
import { cellularDecoder } from '@sim/envs/cellular';
import { walkDecoder } from '@sim/envs/walk';
import { cliffDecoder } from '@sim/envs/cliff';
import { coinswapDecoder } from '@sim/envs/coinswap';
import { conditioningDecoder } from '@sim/envs/conditioning';
import { contextualDecoder } from '@sim/envs/contextual';
import { countDecoder } from '@sim/envs/count';
import { covariateDecoder } from '@sim/envs/covariate';
import { deepseaDecoder } from '@sim/envs/deepsea';
import { foragerDecoder } from '@sim/envs/forager';
import { hallDecoder } from '@sim/envs/hall';
import { garnetDecoder } from '@sim/envs/garnet';
import { hashedDecoder } from '@sim/envs/hashed';
import { hiddenmodeDecoder } from '@sim/envs/hiddenmode';
import { jbwDecoder } from '@sim/envs/jbw';
import { keydoorDecoder } from '@sim/envs/keydoor';
import { tmazeDecoder } from '@sim/envs/tmaze';
import { mountainCarDecoder } from '@sim/envs/mountainCar';
import { nextingDecoder } from '@sim/envs/nexting';
import { operantDecoder } from '@sim/envs/operant';
import { parityDecoder } from '@sim/envs/parity';
import { passwordDecoder } from '@sim/envs/password';
import { patienceDecoder } from '@sim/envs/patience';
import { pendulumDecoder } from '@sim/envs/pendulum';
import { permutedDecoder } from '@sim/envs/permuted';
import { puddleDecoder } from '@sim/envs/puddle';
import { reciteDecoder } from '@sim/envs/recite';
import { regressionDecoder } from '@sim/envs/regression';
import { repeatDecoder } from '@sim/envs/repeat';
import { ringDecoder } from '@sim/envs/ring';
import { riverswimDecoder } from '@sim/envs/riverswim';
import { rpsDecoder } from '@sim/envs/rps';
import { scentpuckDecoder } from '@sim/envs/scentpuck';
import { sokobanDecoder } from '@sim/envs/sokoban';
import { subsetparityDecoder } from '@sim/envs/subsetparity';
import { taxiDecoder } from '@sim/envs/taxi';
import { tigerDecoder } from '@sim/envs/tiger';
import { trackingDecoder } from '@sim/envs/tracking';
import { umbrellaDecoder } from '@sim/envs/umbrella';
import { walkbetDecoder } from '@sim/envs/walkbet';
import type { EnvDescriptor } from '../data/envs';
import type { EnvDecoder } from '@sim/types';
import type { SceneFactory } from '@kit/types';

export interface SceneRow {
  decoder: EnvDecoder;
  load: () => Promise<{ default: SceneFactory }>;
}

export const SCENES: Record<string, SceneRow | undefined> = {
  access: { decoder: accessDecoder, load: () => import('./access/index') },
  acrobot: { decoder: acrobotDecoder, load: () => import('./acrobot/index') },
  aliased: { decoder: aliasedDecoder, load: () => import('./aliased/index') },
  bandit: { decoder: banditDecoder, load: () => import('./bandit/index') },
  blocking: { decoder: blockingDecoder, load: () => import('./blocking/index') },
  calibration: { decoder: calibrationDecoder, load: () => import('./calibration/index') },
  cartpole: { decoder: cartpoleDecoder, load: () => import('./cartpole/index') },
  catch: { decoder: catchDecoder, load: () => import('./catch/index') },
  cellular: { decoder: cellularDecoder, load: () => import('./cellular/index') },
  walk: { decoder: walkDecoder, load: () => import('./walk/index') },
  cliff: { decoder: cliffDecoder, load: () => import('./cliff/index') },
  coinswap: { decoder: coinswapDecoder, load: () => import('./coinswap/index') },
  conditioning: { decoder: conditioningDecoder, load: () => import('./conditioning/index') },
  contextual: { decoder: contextualDecoder, load: () => import('./contextual/index') },
  count: { decoder: countDecoder, load: () => import('./count/index') },
  covariate: { decoder: covariateDecoder, load: () => import('./covariate/index') },
  deepsea: { decoder: deepseaDecoder, load: () => import('./deepsea/index') },
  forager: { decoder: foragerDecoder, load: () => import('./forager/index') },
  hall: { decoder: hallDecoder, load: () => import('./hall/index') },
  garnet: { decoder: garnetDecoder, load: () => import('./garnet/index') },
  hashed: { decoder: hashedDecoder, load: () => import('./hashed/index') },
  hiddenmode: { decoder: hiddenmodeDecoder, load: () => import('./hiddenmode/index') },
  jbw: { decoder: jbwDecoder, load: () => import('./jbw/index') },
  keydoor: { decoder: keydoorDecoder, load: () => import('./keydoor/index') },
  tmaze: { decoder: tmazeDecoder, load: () => import('./tmaze/index') },
  mountainCar: { decoder: mountainCarDecoder, load: () => import('./mountainCar/index') },
  nexting: { decoder: nextingDecoder, load: () => import('./nexting/index') },
  operant: { decoder: operantDecoder, load: () => import('./operant/index') },
  parity: { decoder: parityDecoder, load: () => import('./parity/index') },
  password: { decoder: passwordDecoder, load: () => import('./password/index') },
  patience: { decoder: patienceDecoder, load: () => import('./patience/index') },
  pendulum: { decoder: pendulumDecoder, load: () => import('./pendulum/index') },
  permuted: { decoder: permutedDecoder, load: () => import('./permuted/index') },
  puddle: { decoder: puddleDecoder, load: () => import('./puddle/index') },
  recite: { decoder: reciteDecoder, load: () => import('./recite/index') },
  regression: { decoder: regressionDecoder, load: () => import('./regression/index') },
  repeat: { decoder: repeatDecoder, load: () => import('./repeat/index') },
  ring: { decoder: ringDecoder, load: () => import('./ring/index') },
  riverswim: { decoder: riverswimDecoder, load: () => import('./riverswim/index') },
  rps: { decoder: rpsDecoder, load: () => import('./rps/index') },
  scentpuck: { decoder: scentpuckDecoder, load: () => import('./scentpuck/index') },
  sokoban: { decoder: sokobanDecoder, load: () => import('./sokoban/index') },
  subsetparity: { decoder: subsetparityDecoder, load: () => import('./subsetparity/index') },
  taxi: { decoder: taxiDecoder, load: () => import('./taxi/index') },
  tiger: { decoder: tigerDecoder, load: () => import('./tiger/index') },
  tracking: { decoder: trackingDecoder, load: () => import('./tracking/index') },
  umbrella: { decoder: umbrellaDecoder, load: () => import('./umbrella/index') },
  walkbet: { decoder: walkbetDecoder, load: () => import('./walkbet/index') },
};

export const sceneOf = (env: EnvDescriptor): SceneRow | null => SCENES[env.family] ?? null;
export const posterOf = (family: string): string => `/posters/${family}.webp`;
