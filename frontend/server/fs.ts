import { lstat, stat } from 'node:fs/promises';

export interface FileStat {
  size: number;
  mtimeMs: number;
}

export async function statOrNull(path: string): Promise<FileStat | null> {
  try {
    const st = await stat(path);
    return { size: st.size, mtimeMs: st.mtimeMs };
  } catch {
    return null;
  }
}

export interface LinkStat extends FileStat {
  isSymbolicLink: boolean;
  isDirectory: boolean;
  isFile: boolean;
}

export async function lstatOrNull(path: string): Promise<LinkStat | null> {
  try {
    const st = await lstat(path);
    return {
      size: st.size,
      mtimeMs: st.mtimeMs,
      isSymbolicLink: st.isSymbolicLink(),
      isDirectory: st.isDirectory(),
      isFile: st.isFile(),
    };
  } catch {
    return null;
  }
}
