/* The light/dark switch in the header; the choice is kept in localStorage (vl-theme). */
import { fill, icon } from './dom.js';

export function initTheme() {
  const root = document.documentElement;
  const button = document.getElementById('theme-toggle');
  const current = () => {
    if (root.dataset.theme) return root.dataset.theme;
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  };
  const paint = () => {
    if (!button) return;
    fill(button, icon(current() === 'dark' ? 'sun' : 'moon', 20));
    button.setAttribute('aria-pressed', current() === 'dark' ? 'true' : 'false');
  };
  if (button) {
    button.addEventListener('click', () => {
      const next = current() === 'dark' ? 'light' : 'dark';
      root.dataset.theme = next;
      try {
        localStorage.setItem('vl-theme', next);
      } catch (e) {
        /* storage blocked: the choice lasts for this page only */
      }
      paint();
    });
  }
  paint();
}
