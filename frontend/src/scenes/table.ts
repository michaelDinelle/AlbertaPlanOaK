import { acrobotDecoder } from '@sim/envs/acrobot';
import { walkDecoder } from '@sim/envs/walk';
import { deepseaDecoder } from '@sim/envs/deepsea';
import { hallDecoder } from '@sim/envs/hall';
import { jbwDecoder } from '@sim/envs/jbw';
import { mountainCarDecoder } from '@sim/envs/mountainCar';
import { pendulumDecoder } from '@sim/envs/pendulum';
import { environmentName, type EnvDescriptor } from '../data/envs';
import type { EnvDecoder } from '@sim/types';
import type { SceneFactory } from '@kit/types';
import { cliffDecoder } from '@sim/envs/cliff';
import { mazeDecoder } from '@sim/envs/maze';
import { tMazeDecoder } from '@sim/envs/tmaze';
import { halfCheetahDecoder } from '@sim/envs/planar';
import { arm2Decoder } from '@sim/envs/arm2';
import { corridorDecoder } from '@sim/envs/corridor';
import { g2048Decoder } from '@sim/envs/g2048';
import { accessDecoder } from '@sim/envs/access';
import { ancientsDecoder } from '@sim/envs/ancients';
import { backgammonDecoder } from '@sim/envs/backgammon';
import { banditDecoder } from '@sim/envs/bandit';
import { cartPoleDecoder } from '@sim/envs/cartpole';
import { catchDecoder } from '@sim/envs/catch';
import { calibrationDecoder } from '@sim/envs/calibration';
import { blockingDecoder } from '@sim/envs/blocking';
import { coinSwapDecoder } from '@sim/envs/coinswap';
import { contextualDecoder } from '@sim/envs/contextual';
import { chessDecoder } from '@sim/envs/chess';
import { connectFourDecoder } from '@sim/envs/connect4';
import { covariateDecoder } from '@sim/envs/covariate';
import { countDecoder } from '@sim/envs/count';
import { darkKeyToDoorDecoder } from '@sim/envs/darkkey';
import { craftaxDecoder } from '@sim/envs/craftax';
import { elevenDecoder } from '@sim/envs/eleven';
import { dungeonDecoder } from '@sim/envs/dungeon';
import { frostDecoder } from '@sim/envs/frost';
import { generalsDecoder } from '@sim/envs/generals';
import { hanabiDecoder } from '@sim/envs/hanabi';
import { hanoiDecoder } from '@sim/envs/hanoi';
import { hashedDecoder } from '@sim/envs/hashed';
import { hiddenModeDecoder } from '@sim/envs/hiddenmode';
import { invertedDoublePendulumDecoder } from '@sim/envs/planar';
import { hopperDecoder } from '@sim/envs/planar';
import { homeostasisDecoder } from '../sim/envs/homeostasis';
import { idbdDecoder } from '@sim/envs/supervised';
import { keyDoorDecoder } from '@sim/envs/keydoor';
import { lbfDecoder } from '@sim/envs/lbf';
import { pokerDecoder } from '@sim/envs/poker';
import { lightCyclesDecoder } from '@sim/envs/lightcycles';
import { micrortsDecoder } from '@sim/envs/microrts';
import { minesweeperDecoder } from '@sim/envs/minesweeper';
import { nextingDecoder } from '@sim/envs/nexting';
import { quadrotorDecoder } from '@sim/envs/quadrotor';
import { puddleDecoder } from '@sim/envs/puddle';
import { overcookedDecoder } from '@sim/envs/overcooked';
import { pommermanDecoder } from '@sim/envs/pommerman';
import { parityDecoder } from '@sim/envs/parity';
import { operantDecoder } from '@sim/envs/operant';
import { pocketCubeDecoder } from '@sim/envs/pocketcube';
import { patienceDecoder } from '@sim/envs/patience';
import { rwareDecoder } from '@sim/envs/rware';
import { rateTrackingDecoder } from '@sim/envs/supervised';
import { reacherDecoder } from '@sim/envs/planar';
import { regressionDecoder } from '@sim/envs/regression';
import { relicFogDecoder } from '@sim/envs/relicfog';
import { repeatDecoder } from '@sim/envs/repeat';
import { ringDecoder } from '@sim/envs/ring';
import { scentPuckDecoder } from '@sim/envs/scentpuck';
import { reversiDecoder } from '@sim/envs/reversi';
import { riverSwimDecoder } from '@sim/envs/riverswim';
import { rpsDecoder } from '@sim/envs/rps';
import { slowRegressionDecoder } from '@sim/envs/supervised';
import { slimeDecoder } from '@sim/envs/slime';
import { skirmishDecoder } from '@sim/envs/skirmish';
import { subsetParityDecoder } from '@sim/envs/subsetparity';
import { tetrisDecoder } from '@sim/envs/tetris';
import { sokobanDecoder } from '@sim/envs/sokoban';
import { taxiDecoder } from '@sim/envs/taxi';
import { tigerDecoder } from '@sim/envs/tiger';
import { alchemyDecoder } from '@sim/envs/alchemy';
import { conditioningDecoder } from '@sim/envs/conditioning';
import { aliasedDecoder } from '@sim/envs/aliased';
import { trackingDecoder } from '@sim/envs/tracking';
import { walkBetDecoder } from '@sim/envs/walkbet';
import { umbrellaDecoder } from '@sim/envs/umbrella';
import { cellularDecoder } from '@sim/envs/cellular';
import { garnetDecoder } from '@sim/envs/garnet';
import { foragerDecoder } from '@sim/envs/forager';
import { minatarDecoder } from '@sim/envs/minatar';
import { passwordDecoder } from '@sim/envs/password';
import { permutedDecoder } from '@sim/envs/permuted';
import { reciteDecoder } from '@sim/envs/recite';

export interface SceneRow {
  decoder: EnvDecoder;
  load: () => Promise<{ default: SceneFactory }>;
}

const walk: SceneRow = { decoder: walkDecoder, load: () => import('./walk/index') };
const hall: SceneRow = { decoder: hallDecoder, load: () => import('./hall/index') };

const SCENES: Record<string, SceneRow | undefined> = {
  Acrobot: { decoder: acrobotDecoder, load: () => import('./acrobot/index') },
  Chain: walk,
  DelayedChain: { decoder: walkDecoder, load: () => import('./delayedChain/index') },
  DeepSea: { decoder: deepseaDecoder, load: () => import('./deepsea/index') },
  FourRoomsHall: hall,
  FourRooms: hall,
  JellyBeanWorld: { decoder: jbwDecoder, load: () => import('./jbw/index') },
  MountainCar: { decoder: mountainCarDecoder, load: () => import('./mountainCar/index') },
  Pendulum: { decoder: pendulumDecoder, load: () => import('./pendulum/index') },
  CliffWalk: { decoder: cliffDecoder, load: () => import('./cliffWalk/index') },
  POMaze: { decoder: mazeDecoder, load: () => import('./poMaze/index') },
  ProceduralMaze: { decoder: mazeDecoder, load: () => import('./proceduralMaze/index') },
  TMaze: { decoder: tMazeDecoder, load: () => import('./tMaze/index') },
  HighDimMaze: { decoder: mazeDecoder, load: () => import('./highDimMaze/index') },
  HalfCheetah: { decoder: halfCheetahDecoder, load: () => import('./halfCheetah/index') },
  Arm2: { decoder: arm2Decoder, load: () => import('./arm2/index') },
  KeyCorridor: { decoder: corridorDecoder, load: () => import('./keyCorridor/index') },
  '2048': { decoder: g2048Decoder, load: () => import('./g2048/index') },
  AccessControl: { decoder: accessDecoder, load: () => import('./accessControl/index') },
  Ancients: { decoder: ancientsDecoder, load: () => import('./ancients/index') },
  Backgammon: { decoder: backgammonDecoder, load: () => import('./backgammon/index') },
  Bandit: { decoder: banditDecoder, load: () => import('./bandit/index') },
  CartPoleSwingUp: { decoder: cartPoleDecoder, load: () => import('./cartPoleSwingUp/index') },
  Catch: { decoder: catchDecoder, load: () => import('./catch/index') },
  Calibration: { decoder: calibrationDecoder, load: () => import('./calibration/index') },
  BlockingMaze: { decoder: blockingDecoder, load: () => import('./blockingMaze/index') },
  CoinSwap: { decoder: coinSwapDecoder, load: () => import('./coinSwap/index') },
  Contextual: { decoder: contextualDecoder, load: () => import('./contextual/index') },
  Chess: { decoder: chessDecoder, load: () => import('./chess/index') },
  ConnectFour: { decoder: connectFourDecoder, load: () => import('./connectFour/index') },
  Covariate: { decoder: covariateDecoder, load: () => import('./covariate/index') },
  CountRecall: { decoder: countDecoder, load: () => import('./countRecall/index') },
  DarkKeyToDoor: { decoder: darkKeyToDoorDecoder, load: () => import('./darkKeyToDoor/index') },
  Craftax: { decoder: craftaxDecoder, load: () => import('./craftax/index') },
  Eleven: { decoder: elevenDecoder, load: () => import('./eleven/index') },
  Dungeon: { decoder: dungeonDecoder, load: () => import('./dungeon/index') },
  FrostHollow: { decoder: frostDecoder, load: () => import('./frostHollow/index') },
  Generals: { decoder: generalsDecoder, load: () => import('./generals/index') },
  Hanabi: { decoder: hanabiDecoder, load: () => import('./hanabi/index') },
  Hanoi: { decoder: hanoiDecoder, load: () => import('./hanoi/index') },
  HashedMDP: { decoder: hashedDecoder, load: () => import('./hashedMDP/index') },
  HiddenMode: { decoder: hiddenModeDecoder, load: () => import('./hiddenMode/index') },
  InvertedDoublePendulum: { decoder: invertedDoublePendulumDecoder, load: () => import('./invertedDoublePendulum/index') },
  Hopper: { decoder: hopperDecoder, load: () => import('./hopper/index') },
  Homeostasis: { decoder: homeostasisDecoder, load: () => import('./homeostasis/index') },
  IDBD: { decoder: idbdDecoder, load: () => import('./idbd/index') },
  KeyDoor: { decoder: keyDoorDecoder, load: () => import('./keyDoor/index') },
  LBF: { decoder: lbfDecoder, load: () => import('./lbf/index') },
  KuhnPoker: { decoder: pokerDecoder, load: () => import('./kuhnPoker/index') },
  LeducPoker: { decoder: pokerDecoder, load: () => import('./leducPoker/index') },
  LightCycles: { decoder: lightCyclesDecoder, load: () => import('./lightCycles/index') },
  MicroRTS: { decoder: micrortsDecoder, load: () => import('./microRTS/index') },
  Minesweeper: { decoder: minesweeperDecoder, load: () => import('./minesweeper/index') },
  Nexting: { decoder: nextingDecoder, load: () => import('./nexting/index') },
  Quadrotor2D: { decoder: quadrotorDecoder, load: () => import('./quadrotor2D/index') },
  PuddleWorld: { decoder: puddleDecoder, load: () => import('./puddleWorld/index') },
  OvercookedAI: { decoder: overcookedDecoder, load: () => import('./overcookedAI/index') },
  Pommerman: { decoder: pommermanDecoder, load: () => import('./pommerman/index') },
  Parity: { decoder: parityDecoder, load: () => import('./parity/index') },
  Operant: { decoder: operantDecoder, load: () => import('./operant/index') },
  PocketCube: { decoder: pocketCubeDecoder, load: () => import('./pocketCube/index') },
  Patience: { decoder: patienceDecoder, load: () => import('./patience/index') },
  RWARE: { decoder: rwareDecoder, load: () => import('./rware/index') },
  RateTracking: { decoder: rateTrackingDecoder, load: () => import('./rateTracking/index') },
  Reacher: { decoder: reacherDecoder, load: () => import('./reacher/index') },
  Regression: { decoder: regressionDecoder, load: () => import('./regression/index') },
  RelicFog: { decoder: relicFogDecoder, load: () => import('./relicFog/index') },
  Repeat: { decoder: repeatDecoder, load: () => import('./repeat/index') },
  Ring: { decoder: ringDecoder, load: () => import('./ring/index') },
  ScentPuck: { decoder: scentPuckDecoder, load: () => import('./scentPuck/index') },
  Reversi: { decoder: reversiDecoder, load: () => import('./reversi/index') },
  RiverSwim: { decoder: riverSwimDecoder, load: () => import('./riverSwim/index') },
  RockPaperScissors: { decoder: rpsDecoder, load: () => import('./rockPaperScissors/index') },
  SlowRegression: { decoder: slowRegressionDecoder, load: () => import('./slowRegression/index') },
  ShortcutMaze: { decoder: blockingDecoder, load: () => import('./shortcutMaze/index') },
  SlimeVolley: { decoder: slimeDecoder, load: () => import('./slimeVolley/index') },
  Skirmish: { decoder: skirmishDecoder, load: () => import('./skirmish/index') },
  SubsetParity: { decoder: subsetParityDecoder, load: () => import('./subsetParity/index') },
  Tetris: { decoder: tetrisDecoder, load: () => import('./tetris/index') },
  Sokoban: { decoder: sokobanDecoder, load: () => import('./sokoban/index') },
  Taxi: { decoder: taxiDecoder, load: () => import('./taxi/index') },
  Tiger: { decoder: tigerDecoder, load: () => import('./tiger/index') },
  SymbolicAlchemy: { decoder: alchemyDecoder, load: () => import('./symbolicAlchemy/index') },
  TracePatterning: { decoder: conditioningDecoder, load: () => import('./tracePatterning/index') },
  AliasedRing: { decoder: aliasedDecoder, load: () => import('./aliasedRing/index') },
  Tracking: { decoder: trackingDecoder, load: () => import('./tracking/index') },
  WalkBet: { decoder: walkBetDecoder, load: () => import('./walkBet/index') },
  TraceConditioning: { decoder: conditioningDecoder, load: () => import('./traceConditioning/index') },
  Umbrella: { decoder: umbrellaDecoder, load: () => import('./umbrella/index') },
  Cellular: { decoder: cellularDecoder, load: () => import('./cellular/index') },
  Garnet: { decoder: garnetDecoder, load: () => import('./garnet/index') },
  Forager: { decoder: foragerDecoder, load: () => import('./forager/index') },
  MinAtar: { decoder: minatarDecoder, load: () => import('./minAtar/index') },
  Password: { decoder: passwordDecoder, load: () => import('./password/index') },
  Permuted: { decoder: permutedDecoder, load: () => import('./permuted/index') },
  Recite: { decoder: reciteDecoder, load: () => import('./recite/index') },
};

export function sceneOf(env: EnvDescriptor): SceneRow {
  const row = SCENES[environmentName(env)];
  if (!row) throw new Error(`No scene is registered for ${environmentName(env)}`);
  return row;
}

export const posterOf = (name: string): string => `/posters/${name}.webp`;
