import type { BunPlugin } from 'bun';
import { transformWgsl } from '@vgpu/wgsl/loader-vite';

const wgsl: BunPlugin = {
  name: 'vgpu-wgsl',
  setup(build) {
    build.onLoad({ filter: /\.wgsl$/ }, async ({ path }) => ({ contents: (await transformWgsl({ source: await Bun.file(path).text(), id: path })).code, loader: 'js' }));
  },
};

export default wgsl;
