import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body></body></html>');
for (const [name, value] of Object.entries({ window, document: window.document, Node: window.Node })) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const { initNew, readDraft, writeDraft } = await import('../../../src/activelabel/web/static/js/pages/new.js');

setMessages({
  'new.name_required': 'Give the project a name',
  'error.project_name_taken': 'That name is taken',
  'field.classes': 'Classes',
  'field.targets': 'Targets',
}, 'en');

const box = document.createElement('div');
box.innerHTML = `<form id="new-form"><input name="name"><textarea name="description"></textarea>
  <div id="task-picker"></div><label id="classes-label"></label><div id="classes-field"></div>
  <p id="classes-hint"></p><button id="create-btn">Create</button></form>`;
document.body.appendChild(box);
if (!document.getElementById('toasts')) {
  const toasts = document.createElement('div');
  toasts.id = 'toasts';
  document.body.appendChild(toasts);
}
initNew();
const form = document.getElementById('new-form');
const submit = () => form.dispatchEvent(new window.Event('submit', { cancelable: true }));
const lastToast = () => document.getElementById('toasts').lastElementChild;

test('the form gets the task cards and the class chips, and starts on the name', () => {
  assert.equal(form.querySelectorAll('.task-card').length, 5);
  assert.ok(form.querySelector('.chips-box'));
  assert.equal(document.activeElement, form.elements.name);
});

test('a regression project names its classes field Targets (#619)', () => {
  const label = document.getElementById('classes-label');
  assert.equal(label.textContent, 'Classes');
  const input = form.querySelector('input[value="regression"]');
  input.checked = true;
  input.dispatchEvent(new window.Event('change', { bubbles: true }));
  assert.equal(label.textContent, 'Targets');
  const back = form.querySelector('input[value="classification"]');
  back.checked = true;
  back.dispatchEvent(new window.Event('change', { bubbles: true }));
  assert.equal(label.textContent, 'Classes');
});

test('without a name nothing is sent and the form says why', () => {
  let sent = 0;
  const real = globalThis.fetch;
  globalThis.fetch = async () => { sent += 1; };
  submit();
  globalThis.fetch = real;
  assert.equal(sent, 0);
  assert.equal(lastToast().dataset.text, 'Give the project a name');
});

test('the project is sent with its name, task and classes; an error gives the button back', async () => {
  form.elements.name.value = '  Cats ';
  form.querySelector('.chips-box input').value = 'cat, dog';
  form.querySelector('input[value="segmentation"]').checked = true;
  form.querySelector('input[value="segmentation"]').dispatchEvent(new window.Event('change'));
  let body = null;
  const real = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    body = JSON.parse(init.body);
    return { ok: false, status: 409, text: async () => JSON.stringify({ code: 'project_name_taken', detail: 'taken', params: {} }) };
  };
  submit();
  assert.equal(document.getElementById('create-btn').disabled, true);
  await new Promise((resolve) => setTimeout(resolve, 20));
  globalThis.fetch = real;
  assert.deepEqual(body, { name: 'Cats', description: '', task: 'segmentation', classes: ['cat', 'dog'] });
  assert.equal(document.getElementById('create-btn').disabled, false);
  /* the taken name stays marked on its field, not only in a toast (#637) */
  const nameError = document.getElementById('f-name-error');
  assert.equal(form.elements.name.getAttribute('aria-invalid'), 'true');
  assert.equal(nameError.hidden, false);
  assert.equal(nameError.textContent, 'That name is taken');
  assert.equal(form.elements.name.getAttribute('aria-describedby'), 'f-name-error');
  assert.equal(document.activeElement, form.elements.name);
  form.elements.name.dispatchEvent(new window.Event('input', { bubbles: true }));
  assert.equal(form.elements.name.hasAttribute('aria-invalid'), false);
  assert.equal(nameError.hidden, true);
  box.remove();
});

/** A sessionStorage stand-in. */
function memoryStorage() {
  const data = new Map();
  return { getItem: (k) => (data.has(k) ? data.get(k) : null), setItem: (k, v) => data.set(k, String(v)), removeItem: (k) => data.delete(k), data };
}

test('the typed form is kept for the reload of a language switch and forgotten when empty (#637)', () => {
  const storage = memoryStorage();
  const draft = { name: 'Cats', description: 'pets', task: 'segmentation', classes: ['cat', 'dog'] };
  writeDraft(storage, draft);
  assert.deepEqual(readDraft(storage), draft);
  writeDraft(storage, { name: '', description: '', task: 'detection', classes: [] });
  assert.equal(readDraft(storage), null);
  writeDraft(storage, draft);
  writeDraft(storage, null);
  assert.equal(storage.data.size, 0);
});

test('a damaged or blocked storage gives an empty form, never an error', () => {
  const storage = memoryStorage();
  storage.setItem('activelabel.new-project-draft', '{not json');
  assert.equal(readDraft(storage), null);
  storage.setItem('activelabel.new-project-draft', '[1]');
  assert.equal(readDraft(storage), null);
  const blocked = { getItem: () => { throw new Error('denied'); }, setItem: () => { throw new Error('denied'); }, removeItem: () => {} };
  assert.equal(readDraft(blocked), null);
  writeDraft(blocked, { name: 'x' });
  assert.equal(readDraft(null), null);
  window.dispatchEvent(new window.Event('pagehide'));   // the page without storage saves nothing and does not throw
});
