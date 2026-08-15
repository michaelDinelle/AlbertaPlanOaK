export const SLOT_NAMES = [
  'none', 'x', 'y', 'gx', 'gy', 'kx', 'ky', 'carry', 'gates', 'tokens', 'cue', 'heading', 'effort',
] as const;

export type NamedSlot = (typeof SLOT_NAMES)[number] | `q${number}`;

export type RecordEnc = 'u8' | 'i16' | 'dx32';

export interface RecordFieldSpec {
  slot: number;
  enc: RecordEnc;
  scale: number;
  wrap: boolean;
}

export interface EnvEntry {
  envId: string;
  kind: number;
  actionKind: 'box' | 'discrete';
  actionCount: number;
  controlHz: number;
  gridW: number;
  gridH: number;
  slots: NamedSlot[];
  ctrlBox: number;
  fields: RecordFieldSpec[];
  slotAt: Partial<Record<NamedSlot, number>>;
  isGrid: boolean;
  discretePose: boolean;
}

export type EnvCatalog = ReadonlyMap<string, EnvEntry>;

export class CatalogError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'CatalogError';
  }
}

function requireArray(v: unknown, what: string): unknown[] {
  if (!Array.isArray(v)) throw new CatalogError(`${what} is missing from the run catalog`);
  return v;
}

function requireNumber(v: unknown, what: string): number {
  if (typeof v !== 'number' || !Number.isFinite(v)) {
    throw new CatalogError(`${what} is not a finite number in the run catalog`);
  }
  return v;
}

function requireString(v: unknown, what: string): string {
  if (typeof v !== 'string' || v.length === 0) {
    throw new CatalogError(`${what} is not a string in the run catalog`);
  }
  return v;
}

function readSlot(v: unknown, what: string): NamedSlot {
  const s = requireString(v, what);
  if ((SLOT_NAMES as readonly string[]).includes(s)) return s as NamedSlot;
  if (/^q\d+$/.test(s)) return s as NamedSlot;
  throw new CatalogError(`${what} names an unknown slot ${JSON.stringify(s)}`);
}

function readField(v: unknown, what: string): RecordFieldSpec {
  const o = v as Record<string, unknown>;
  const enc = requireString(o?.['enc'], `${what}.enc`);
  if (enc !== 'u8' && enc !== 'i16' && enc !== 'dx32') {
    throw new CatalogError(`${what}.enc is ${JSON.stringify(enc)}`);
  }
  return {
    slot: requireNumber(o['slot'], `${what}.slot`),
    enc,
    scale: requireNumber(o['scale'], `${what}.scale`),
    wrap: o['wrap'] === true,
  };
}

function readEntry(v: unknown, at: number): EnvEntry {
  const o = v as Record<string, unknown>;
  const what = `envCatalog[${at}]`;
  const envId = requireString(o?.['envId'], `${what}.envId`);
  const actionKind = requireString(o['actionKind'], `${what}.actionKind`);
  if (actionKind !== 'box' && actionKind !== 'discrete') {
    throw new CatalogError(`${what}.actionKind is ${JSON.stringify(actionKind)}`);
  }
  const slots = requireArray(o['slots'], `${what}.slots`).map((s, i) =>
    readSlot(s, `${what}.slots[${i}]`),
  );
  const gridW = requireNumber(o['gridW'], `${what}.gridW`);
  const gridH = requireNumber(o['gridH'], `${what}.gridH`);
  const slotAt: Partial<Record<NamedSlot, number>> = {};
  for (let i = 0; i < slots.length; i++) {
    const name = slots[i];
    if (name !== undefined && name !== 'none') slotAt[name] = i;
  }
  return {
    envId,
    kind: requireNumber(o['kind'], `${what}.kind`),
    actionKind,
    actionCount: requireNumber(o['actionCount'], `${what}.actionCount`),
    controlHz: requireNumber(o['controlHz'], `${what}.controlHz`),
    gridW,
    gridH,
    slots,
    ctrlBox: requireNumber(o['ctrlBox'], `${what}.ctrlBox`),
    fields: requireArray(o['fields'], `${what}.fields`).map((f, i) =>
      readField(f, `${what}.fields[${i}]`),
    ),
    slotAt,
    isGrid: gridW > 0 && gridH > 0,
    discretePose: gridW > 0 && gridH > 0,
  };
}

export function parseCatalog(raw: unknown): EnvCatalog {
  const rows = requireArray(raw, 'envCatalog');
  const out = new Map<string, EnvEntry>();
  rows.forEach((row, i) => {
    const entry = readEntry(row, i);
    out.set(entry.envId, entry);
  });
  if (out.size === 0) throw new CatalogError('run catalog declares no environments');
  return out;
}

export function entryFor(catalog: EnvCatalog, envId: string): EnvEntry {
  const entry = catalog.get(envId);
  if (entry === undefined) {
    throw new CatalogError(`run catalog has no entry for ${envId}`);
  }
  return entry;
}

export function slotValue(entry: EnvEntry, pose: ArrayLike<number>, slot: NamedSlot): number {
  const at = entry.slotAt[slot];
  return at === undefined ? Number.NaN : (pose[at] ?? Number.NaN);
}

export function hasSlot(entry: EnvEntry, slot: NamedSlot): boolean {
  return entry.slotAt[slot] !== undefined;
}
