import { join } from 'node:path';

export async function readKeeper(benchRoot: string): Promise<string | null> {
  try {
    const keeper = (await Bun.file(join(benchRoot, '..', 'keeper')).text()).trim();
    return keeper.length > 0 ? keeper : null;
  } catch {
    return null;
  }
}
