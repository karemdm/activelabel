import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body></body></html>');
for (const [name, value] of Object.entries({ window, document: window.document, Node: window.Node })) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
if (!globalThis.ResizeObserver) {
  // jsdom has no layout: an observer that never fires
  globalThis.ResizeObserver = class { observe() {} disconnect() {} };
}
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const { chipsInput, classList, classLine, problemRow, progressBar, emptyState } = await import(
  '../../../src/activelabel/web/static/js/shell/widgets.js'
);

setMessages({ 'field.classes': 'Classes', 'classes.repeated': 'Not added, already in the list: {names}', 'classes.less': 'Less', 'classes.see_all': 'See all', 'common.progress': 'Progress' }, 'en');

const key = (input, name) => input.dispatchEvent(new window.KeyboardEvent('keydown', { key: name, bubbles: true }));

test('chipsInput adds names on Enter or comma, once each, and Backspace removes the last one', () => {
  const chips = chipsInput('c-test', ['cat']);
  document.body.appendChild(chips.node);
  const input = chips.node.querySelector('input');
  input.value = 'dog, Cat, bird';
  key(input, 'Enter');
  assert.deepEqual(chips.get(), ['cat', 'dog', 'bird']);
  key(input, 'Backspace');
  assert.deepEqual(chips.get(), ['cat', 'dog']);
  input.value = 'fish';  // typed and not confirmed: get() still takes it
  assert.deepEqual(chips.get(), ['cat', 'dog', 'fish']);
  chips.node.querySelector('.chip-x').click();
  assert.deepEqual(chips.get(), ['dog', 'fish']);
  chips.node.remove();
});

test('chipsInput says which names it left out as repeated, and hides the example once there is a chip (#637)', () => {
  const chips = chipsInput('c-note', []);
  document.body.appendChild(chips.node);
  const input = chips.node.querySelector('input');
  const note = () => chips.node.querySelector('[role="status"]').textContent;
  const paste = (text) => {
    const event = new window.Event('paste', { cancelable: true, bubbles: true });
    event.clipboardData = { getData: () => text };
    input.dispatchEvent(event);
  };
  chips.setPlaceholder('e.g. crack');
  assert.equal(input.placeholder, 'e.g. crack');
  paste('gato\n\ncachorro; Gato\tgato,\nave');
  assert.deepEqual(chips.get(), ['gato', 'cachorro', 'ave']);
  assert.equal(note(), 'Not added, already in the list: Gato');
  assert.equal(input.placeholder, '');
  input.value = 'rato';
  key(input, 'Enter');
  assert.equal(note(), '');
  chips.node.querySelector('.chip-x').click();
  chips.node.querySelector('.chip-x').click();
  chips.node.querySelector('.chip-x').click();
  chips.node.querySelector('.chip-x').click();
  assert.equal(input.placeholder, 'e.g. crack');
  chips.node.remove();
});

test('chipsInput ignores Enter while an IME is composing (#637)', () => {
  const chips = chipsInput('c-ime', []);
  document.body.appendChild(chips.node);
  const input = chips.node.querySelector('input');
  input.value = 'ね';
  input.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Enter', isComposing: true, bubbles: true, cancelable: true }));
  assert.equal(input.value, 'ね');
  assert.deepEqual([...chips.node.querySelectorAll('.chip')], []);
  key(input, 'Enter');
  assert.deepEqual(chips.get(), ['ね']);
  chips.node.remove();
});

test('classList shows the first 12 classes and a button for the rest', () => {
  const names = Array.from({ length: 15 }, (_, i) => `c${i}`);
  const box = classList(names);
  const hidden = [...box.querySelectorAll('.chip')].filter((c) => c.hidden);
  assert.equal(hidden.length, 3);
  const toggle = box.querySelector('button');
  assert.equal(toggle.textContent, '+3');
  toggle.click();
  assert.equal(toggle.getAttribute('aria-expanded'), 'true');
  assert.equal(toggle.textContent, 'Less');
  assert.equal([...box.querySelectorAll('.chip')].filter((c) => c.hidden).length, 0);
  assert.equal(classList(['a']).querySelector('button'), null);
});

test('classLine names the count and keeps "see all" hidden until a class does not fit', () => {
  const line = classLine(['a', 'b']);
  assert.match(line.textContent, /Classes \(2\)/);
  assert.equal(line.querySelector('button').hidden, true);
});

test('progressBar rounds and caps the percentage; emptyState shows its actions', () => {
  const bar = progressBar(1, 3);
  assert.equal(bar.getAttribute('aria-valuenow'), '33');
  assert.equal(bar.getAttribute('aria-label'), 'Progress');
  assert.equal(progressBar(5, 3, 'Upload').getAttribute('aria-valuenow'), '100');
  assert.equal(progressBar(0, 0).getAttribute('aria-valuenow'), '0');
  const empty = emptyState('info', 'Nothing here', document.createElement('button'));
  assert.equal(empty.querySelector('p').textContent, 'Nothing here');
  assert.ok(empty.querySelector('.ds-empty-actions button'));
  assert.equal(emptyState('info', 'Nothing').querySelector('.ds-empty-actions'), null);
});

test('problemRow shows the translated problem, its count and up to five examples', () => {
  setMessages({ 'problem.bad_box': 'Box outside the image' }, 'en');
  const row = problemRow({ code: 'bad_box', level: 'error', count: 7, examples: ['a', 'b', 'c', 'd', 'e', 'f'] });
  assert.equal(row.querySelector('.problem-msg').textContent, 'Box outside the image');
  assert.ok(row.querySelector('.problem-head.bad'));
  assert.equal(row.querySelector('.count').textContent, '×7');
  assert.equal(row.querySelectorAll('.examples li').length, 5);
  const plain = problemRow({ code: 'unknown', message: 'Server words', level: 'warning' });
  assert.equal(plain.querySelector('details'), null);
  assert.equal(plain.querySelector('.problem-msg').textContent, 'Server words');
  assert.ok(plain.querySelector('.problem-head.warn'));
});

test('Enter adds the class without submitting the form, and a pasted list becomes one class per name (#586)', () => {
  const chips = chipsInput('c-paste', []);
  document.body.appendChild(chips.node);
  const input = chips.node.querySelector('input');
  input.value = 'cat';
  const enter = new window.KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true });
  input.dispatchEvent(enter);
  assert.equal(enter.defaultPrevented, true);  // a form around it is not submitted
  const paste = new window.Event('paste', { bubbles: true, cancelable: true });
  paste.clipboardData = { getData: () => 'dog\nbird; fish\tcat' };
  input.dispatchEvent(paste);
  assert.equal(paste.defaultPrevented, true);
  assert.deepEqual(chips.get(), ['cat', 'dog', 'bird', 'fish']);
  const one = new window.Event('paste', { bubbles: true, cancelable: true });
  one.clipboardData = { getData: () => 'single name' };
  input.dispatchEvent(one);
  assert.equal(one.defaultPrevented, false);  // one name is pasted as text, as usual
  chips.node.remove();
});
