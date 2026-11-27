import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body><div id="toasts"></div></body></html>');
for (const [name, value] of Object.entries({ window, document: window.document, Node: window.Node })) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const query = window.document.querySelectorAll.bind(window.document);
window.document.querySelectorAll = (selector) => {
  try { return query(selector); } catch (err) { return []; }
};
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const { dropTab, reserveTab, showInTab } = await import('../../../src/activelabel/web/static/js/shell/newtab.js');

setMessages({ 'cur.fo_popup': 'Blocked. Open it here:', 'common.close': 'Close' }, 'en');

/** A stand-in for a tab the browser opened. */
const fakeTab = () => ({ closed: false, opener: 'page', location: { href: 'about:blank' }, document: { title: '', body: { textContent: '' } }, close() { this.closed = true; } });
const toasts = () => [...document.getElementById('toasts').children];

test('a tab is opened at once, cut from the page and says it is waiting', () => {
  const tab = fakeTab();
  window.open = (url, target) => { tab.asked = [url, target]; return tab; };
  assert.equal(reserveTab('Waiting'), tab);
  assert.deepEqual(tab.asked, ['', '_blank']);
  assert.equal(tab.opener, null);
  assert.equal(tab.document.body.textContent, 'Waiting');
});

test('a browser that blocks the tab gives none, and dropping nothing or a closed tab is harmless', () => {
  window.open = () => null;
  assert.equal(reserveTab('Waiting'), null);
  window.open = () => { throw new Error('blocked'); };
  assert.equal(reserveTab('Waiting'), null);
  dropTab(null);
  const tab = fakeTab();
  dropTab(tab);
  assert.equal(tab.closed, true);
  dropTab(tab);  // already closed: nothing happens
});

test('the address goes to the reserved tab', () => {
  const tab = fakeTab();
  assert.equal(showInTab(tab, 'http://localhost:5161/datasets/x'), true);
  assert.equal(tab.location.href, 'http://localhost:5161/datasets/x');
});

test('without the reserved tab a new one is tried, and when that is blocked too a link stays on screen', () => {
  const opened = fakeTab();
  window.open = (url) => { opened.url = url; return opened; };
  assert.equal(showInTab(null, 'http://localhost:5161/datasets/x'), true);
  assert.equal(opened.url, 'http://localhost:5161/datasets/x');
  assert.equal(opened.opener, null);

  window.open = () => null;
  const gone = fakeTab();
  gone.closed = true;  // the person closed the waiting tab
  assert.equal(showInTab(gone, 'http://localhost:5161/datasets/y'), false);
  const link = toasts().at(-1).querySelector('a[data-open-link]');
  assert.equal(link.getAttribute('href'), 'http://localhost:5161/datasets/y');
  assert.equal(link.getAttribute('target'), '_blank');
  assert.match(toasts().at(-1).textContent, /Blocked\. Open it here:/);
});
