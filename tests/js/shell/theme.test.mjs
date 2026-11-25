import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body></body></html>', {
  url: 'http://localhost/',
});
for (const [name, value] of Object.entries({
  window, document: window.document, Node: window.Node, localStorage: window.localStorage,
})) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const { initTheme } = await import('../../../src/activelabel/web/static/js/shell/theme.js');

test('the theme button switches light and dark, says which is on and remembers the choice', () => {
  const button = document.createElement('button');
  button.id = 'theme-toggle';
  document.body.appendChild(button);
  const root = document.documentElement;
  root.dataset.theme = 'light';
  initTheme();
  assert.equal(button.getAttribute('aria-pressed'), 'false');
  assert.ok(button.querySelector('.icon-moon'));
  button.click();
  assert.equal(root.dataset.theme, 'dark');
  assert.equal(button.getAttribute('aria-pressed'), 'true');
  assert.ok(button.querySelector('.icon-sun'));
  assert.equal(localStorage.getItem('vl-theme'), 'dark');
  button.click();
  assert.equal(root.dataset.theme, 'light');
  assert.equal(localStorage.getItem('vl-theme'), 'light');
  button.remove();
});
