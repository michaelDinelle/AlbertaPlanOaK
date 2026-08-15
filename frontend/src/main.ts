import { mount } from './views/wall';

const app = document.getElementById('app');
if (!app) throw new Error('#app missing');

let view = await mount(app);
let generation = 0;

window.addEventListener('hashchange', () => {
  const mine = ++generation;
  const previous = view;
  void Promise.resolve(mount(app)).then((next) => {
    if (mine !== generation) {
      next.destroy();
      return;
    }
    previous.destroy();
    view = next;
  });
});
