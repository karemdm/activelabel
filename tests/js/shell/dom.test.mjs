import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body></body></html>');
for (const [name, value] of Object.entries({ window, document: window.document, Node: window.Node })) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const { esc, el, fill, add, redraw, icon } = await import('../../../src/activelabel/web/static/js/shell/dom.js');

test('esc escapes the characters that matter in HTML', () => {
  assert.equal(esc(`<a href="x">'&'</a>`), '&lt;a href=&quot;x&quot;&gt;&#39;&amp;&#39;&lt;/a&gt;');
});

test('el sets class and attributes, skips null and false, and adds listeners for on* props', () => {
  let clicked = 0;
  const node = el('button', { class: 'btn', title: 'Go', hidden: true, disabled: false, 'aria-label': null, onclick: () => { clicked += 1; } });
  assert.equal(node.className, 'btn');
  assert.equal(node.getAttribute('title'), 'Go');
  assert.equal(node.getAttribute('hidden'), '');
  assert.equal(node.hasAttribute('disabled'), false);
  assert.equal(node.hasAttribute('aria-label'), false);
  node.click();
  assert.equal(clicked, 1);
});

test('el writes strings as text, flattens arrays and skips null and false children', () => {
  const node = el('p', null, '<b>bold</b>', null, false, ['a', ['b', 0]], el('i', null, 'x'));
  assert.equal(node.innerHTML, '&lt;b&gt;bold&lt;/b&gt;ab0<i>x</i>');
});

test('fill replaces the children and add appends them, both skipping null', () => {
  const node = el('div', null, 'old');
  fill(node, null, 'new', el('span'));
  assert.equal(node.innerHTML, 'new<span></span>');
  add(node, false, 'more');
  assert.equal(node.innerHTML, 'new<span></span>more');
});

test('redraw keeps the focus on the control with the same data-act', () => {
  const box = el('div', null, el('button', { 'data-act': 'a' }, 'A'), el('button', { 'data-act': 'b' }, 'B'));
  document.body.appendChild(box);
  box.querySelector('[data-act="b"]').focus();
  redraw(box, el('button', { 'data-act': 'a' }, 'A2'), el('button', { 'data-act': 'b' }, 'B2'));
  assert.equal(document.activeElement.textContent, 'B2');
  redraw(box, el('button', { 'data-act': 'c' }, 'C'));
  assert.equal(document.activeElement.textContent, 'C');  // the same one is gone: the first one
  box.remove();
});

test('icon is a hidden span with an svg; an unknown name gives an empty svg', () => {
  const node = icon('check', 16);
  assert.equal(node.className, 'icon icon-check');
  assert.equal(node.getAttribute('aria-hidden'), 'true');
  assert.match(node.innerHTML, /^<svg [^>]*width="16"[^>]*>.+<\/svg>$/);
  assert.match(icon('nope').innerHTML, /<svg [^>]*width="18"[^>]*><\/svg>/);
});
