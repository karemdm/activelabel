import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body></body></html>');
for (const [name, value] of Object.entries({ window, document: window.document, Node: window.Node, Event: window.Event })) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
// jsdom may not implement the modal methods of <dialog>: the smallest stand-ins that keep their contract
const proto = window.HTMLDialogElement.prototype;
if (!proto.showModal) proto.showModal = function showModal() { this.setAttribute('open', ''); };
if (!proto.close) {
  proto.close = function close(value) {
    if (value !== undefined) this.returnValue = value;
    this.removeAttribute('open');
    this.dispatchEvent(new window.Event('close'));
  };
}
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const { once, menuButton, choiceDialog, confirmDialog, closeOpenDialogs, returnFocus } = await import(
  '../../../src/activelabel/web/static/js/shell/dialogs.js'
);

setMessages({ 'common.cancel': 'Cancel', 'common.close': 'Close' }, 'en');

test('once keeps the button disabled while the action runs, then asks stillBusy', async () => {
  const button = document.createElement('button');
  let release;
  const running = new Promise((resolve) => { release = resolve; });
  let calls = 0;
  const handler = once(async () => { calls += 1; await running; }, () => true);
  const first = handler({ currentTarget: button });
  assert.equal(button.disabled, true);
  await handler({ currentTarget: button });  // a second click while it runs does nothing
  release();
  await first;
  assert.equal(calls, 1);
  assert.equal(button.disabled, true);  // stillBusy() said so
});

test('menuButton opens its items, marks disabled ones and closes on Escape', () => {
  let picked = 0;
  const wrap = menuButton('edit', 'More', () => [
    { label: 'Rename', onclick: () => { picked += 1; } },
    { label: 'Delete', disabled: true, reason: 'In use' },
  ]);
  document.body.appendChild(wrap);
  const trigger = wrap.querySelector('button');
  trigger.click();
  assert.equal(trigger.getAttribute('aria-expanded'), 'true');
  const items = wrap.querySelectorAll('[role="menuitem"]');
  assert.equal(items.length, 2);
  assert.equal(items[1].getAttribute('aria-disabled'), 'true');
  assert.match(items[1].textContent, /In use/);
  items[0].dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
  assert.equal(trigger.getAttribute('aria-expanded'), 'false');
  assert.equal(wrap.querySelector('[role="menu"]'), null);
  trigger.click();
  wrap.querySelector('[role="menuitem"]').click();
  assert.equal(picked, 1);
  wrap.remove();
});

test('choiceDialog resolves with the chosen value and null when cancelled', async () => {
  const pending = choiceDialog({ title: 'Pick', text: 'Which?', choices: [{ label: 'A', value: 'a' }, { label: 'B', value: 'b', danger: true }] });
  const dlg = document.querySelector('dialog[aria-label="Pick"]');
  assert.equal(document.activeElement.textContent, 'Cancel');  // Enter never picks an answer
  [...dlg.querySelectorAll('.choices button')][1].click();
  assert.equal(await pending, 'b');
  assert.equal(document.querySelector('dialog[aria-label="Pick"]'), null);
  const cancelled = choiceDialog({ title: 'Pick', text: 'Which?', choices: [{ label: 'A', value: 'a' }] });
  closeOpenDialogs();
  assert.equal(await cancelled, null);
});

test('confirmDialog with requireText enables its button only for the exact text', async () => {
  const pending = confirmDialog({ title: 'Delete', text: 'Sure?', confirmLabel: 'Delete', requireText: 'cats' });
  const dlg = document.querySelector('dialog[aria-label="Delete"]');
  const input = dlg.querySelector('input');
  const ok = dlg.querySelector('button[type="submit"]');
  assert.equal(ok.disabled, true);
  input.value = 'cat';
  input.dispatchEvent(new window.Event('input'));
  assert.equal(ok.disabled, true);
  assert.equal(input.getAttribute('aria-invalid'), 'true');
  input.value = 'cats';
  input.dispatchEvent(new window.Event('input'));
  assert.equal(ok.disabled, false);
  [...dlg.querySelectorAll('button')].find((b) => b.textContent === 'Cancel').click();
  assert.equal(await pending, false);
});

test('closing a dialog gives the focus back to its button, or to what opened it when that button is hidden', () => {
  const header = document.createElement('button');
  const empty = document.createElement('button');
  document.body.append(header, empty);
  header.hidden = true;   // the header button is hidden while the list is empty
  returnFocus(header, empty);
  assert.equal(document.activeElement, empty);
  header.hidden = false;
  returnFocus(header, empty);
  assert.equal(document.activeElement, header);
  header.remove();
  empty.remove();
  returnFocus(header, empty);   // both gone: nothing to do, no error
  const redrawn = document.createElement('button');   // the empty state drawn again while the dialog was open
  document.body.append(redrawn);
  returnFocus(header, empty, redrawn);
  assert.equal(document.activeElement, redrawn);
  redrawn.remove();
});
