export interface RecordFieldSpec {
  slot: number;
  scale: number;
  wrap: boolean;
  anchor: boolean;
}

export interface EnvEntry {
  envId: string;
  controlHz: number;
  discretePose: boolean;
  fields: RecordFieldSpec[];
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

function readField(v: unknown, what: string): RecordFieldSpec {
  const o = v as Record<string, unknown>;
  return {
    slot: requireNumber(o?.['slot'], `${what}.slot`),
    scale: requireNumber(o['scale'], `${what}.scale`),
    wrap: o['wrap'] === true,
    anchor: o['anchor'] === true,
  };
}

function readEntry(v: unknown, at: number): EnvEntry {
  const o = v as Record<string, unknown>;
  const what = `envCatalog[${at}]`;
  const envId = requireString(o?.['envId'], `${what}.envId`);
  const category = requireString(o['category'], `${what}.category`);
  return {
    envId,
    controlHz: requireNumber(o['controlHz'], `${what}.controlHz`),
    discretePose: category === 'grid' || category === 'jellyBean',
    fields: requireArray(o['fields'], `${what}.fields`).map((f, i) =>
      readField(f, `${what}.fields[${i}]`),
    ),
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
