export const STAGE_KEYS = ['fc', 'st', 'o', 'm', 'p'] as const;

export type StageKey = (typeof STAGE_KEYS)[number];

export interface Stage {
  key: StageKey;
  badge: string;
  name: readonly string[];
  plan: string;
  handoff: string;
  symbols: readonly string[];
}

export const STAGES = {
  fc: {
    key: 'fc',
    badge: 'FC',
    name: ['Feature', 'Construction'],
    plan: 'Construct a new feature of state out of the stream, and keep the ones that turn out to matter.',
    handoff: 'a feature worth attaining',
    symbols: ['constructFeatures()', 'perceptsForStream()', 'CONSTRUCTED_PER_PERCEPT', 'CANDIDATES_PER_SLOT'],
  },
  st: {
    key: 'st',
    badge: 'ST',
    name: ['SubTask'],
    plan: 'Pose a subtask from that feature: reach it, and know when reaching it is worth more than carrying on.',
    handoff: 'a subtask, and a slot to solve it in',
    symbols: ['replaceWorstFeature()', 'optionFeature[k] = worst', 'stoppingValue', 'nurseryScore'],
  },
  o: {
    key: 'o',
    badge: 'O',
    name: ['Option'],
    plan: 'Learn an option: a policy that solves the subtask, running until its stopping condition fires.',
    handoff: 'behaviour, and what it did next',
    symbols: ['emitOptionActions()', 'blendKeyboard()', 'importanceStaging', 'OPTION_COUNT'],
  },
  m: {
    key: 'm',
    badge: 'M',
    name: ['Model'],
    plan: 'Learn a model of the option: where it leads, how long it takes, and what it earns on the way.',
    handoff: 'model error, and where it is thickest',
    symbols: ['optionReward', 'optionDuration', 'optionNextValue', 'optionNextValueTwin'],
  },
  p: {
    key: 'p',
    badge: 'P',
    name: ['Planning'],
    plan: 'Plan with the option models — and let planning decide what is worth perceiving next.',
    handoff: 'utility — which feature dies next',
    symbols: ['planAllOptions()', 'optionPriority', 'updateUtilities()', 'PLAN_PERIOD'],
  },
} satisfies Record<StageKey, Stage>;

export const STAGE_ORDER: readonly Stage[] = STAGE_KEYS.map((key) => STAGES[key]);

export interface Paper {
  title: string;
  cite: string;
  href: string;
}

export const ALBERTA_PLAN: Paper = {
  title: 'The Alberta Plan for AI Research',
  cite: 'Sutton, Bowling, Pilarski · arXiv:2208.11173',
  href: 'https://arxiv.org/abs/2208.11173',
};

export interface Talk {
  id: string;
  title: string;
  venue: string;
  note: string;
}

export const TALKS: readonly Talk[] = [
  {
    id: 'gEbbGyNkR2U',
    title: 'The OaK Architecture: A Vision of SuperIntelligence from Experience',
    venue: 'Amii · RLC 2025',
    note: 'The talk that names the FC-STOMP progression.',
  },
  {
    id: 'FLOL2f4iHKA',
    title: 'The Era of Experience & The Age of Design',
    venue: 'Amii · Upper Bound 2025',
    note: 'Why experience, not human data, is the ground to build on.',
  },
  {
    id: 'NvfK1TkXmOQ',
    title: 'Richard Sutton’s new path for AI',
    venue: 'Amii · Approximately Correct',
    note: 'The same argument, in conversation.',
  },
];
