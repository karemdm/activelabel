import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body><div id="toasts"></div></body></html>');
for (const [name, value] of Object.entries({ window, document: window.document, Node: window.Node })) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
// jsdom has no top layer and may not parse `dialog:modal`: such a selector then matches nothing, as with no dialog open
const query = window.document.querySelectorAll.bind(window.document);
window.document.querySelectorAll = (selector) => {
  try { return query(selector); } catch (err) { return []; }
};
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const { toast, toastError } = await import('../../../src/activelabel/web/static/js/shell/toast.js');

setMessages({ 'common.close': 'Close' }, 'en');
const box = () => document.getElementById('toasts');
const clear = () => { for (const n of [...box().children]) { clearTimeout(n.timer); n.remove(); } };

test('a toast is a status message with a close button', () => {
  clear();
  toast('Saved', 'success', 0);
  const node = box().lastElementChild;
  assert.equal(node.getAttribute('role'), 'status');
  assert.equal(node.querySelector('.toast-text').textContent, 'Saved');
  const close = node.querySelector('button');
  assert.equal(close.getAttribute('aria-label'), 'Close');
  close.click();
  assert.equal(box().children.length, 0);
});

test('the same message of the same kind is shown once', () => {
  clear();
  toast('Twice', 'info', 0);
  toast('Twice', 'info', 0);
  toast('Twice', 'error', 0);
  assert.equal(box().children.length, 2);
  clear();
});

test('toastError shows the error text as an alert, without an "Error:" prefix', () => {
  clear();
  toastError(new Error('Disk full'));
  const node = box().lastElementChild;
  assert.equal(node.getAttribute('role'), 'alert');
  assert.equal(node.dataset.text, 'Disk full');
  toastError('plain text');
  assert.equal(box().lastElementChild.dataset.text, 'plain text');
  clear();
});
