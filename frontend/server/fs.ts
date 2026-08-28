import { lstat, stat } from 'node:fs/promises';

export async function statOrNull(path: string): Promise<{ size: number; mtimeMs: number } | null> {
  try {
    const st = await stat(path);
    return { size: st.size, mtimeMs: st.mtimeMs };
  } catch {
    return null;
  }
}

export async function lstatOrNull(
  path: string,
): Promise<{ size: number; mtimeMs: number; isSymbolicLink: boolean; isDirectory: boolean; isFile: boolean } | null> {
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
