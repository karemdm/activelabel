import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body><div id="toasts"></div></body></html>', { url: 'http://localhost/' });
for (const [name, value] of Object.entries({ window, document: window.document, Node: window.Node, Event: window.Event })) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
// jsdom may not parse `dialog:modal` (toasts look for an open modal): such a selector then matches nothing
const query = window.document.querySelectorAll.bind(window.document);
window.document.querySelectorAll = (selector) => {
  try { return query(selector); } catch (err) { return []; }
};
const proto = window.HTMLDialogElement.prototype;
if (!proto.showModal) proto.showModal = function showModal() { this.setAttribute('open', ''); };
if (!proto.close) proto.close = function close() { this.removeAttribute('open'); this.dispatchEvent(new window.Event('close')); };
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const f = await import('../../../src/activelabel/web/static/js/shell/features.js');

const MISSING = {
  fiftyone: { installed: false, version: null, install: 'pip install "activelabel[fiftyone]"', provides: ['fiftyone_app', 'cvat_import'] },
  training_environment: { exists: true, path: '/ws/training-env', bytes: 3 * 1024 ** 3, packages: { torch: '2.5.1', torchvision: '0.20.1' } },
};

setMessages({
  'features.title': 'Installed parts', 'features.not_installed': 'Not installed', 'features.fo_missing_short': 'FiftyOne is not installed. Install it with: {install}',
  'features.provides.fiftyone_app': 'Open in FiftyOne', 'features.provides.cvat_import': 'Import from CVAT', 'features.env_size': 'Takes {size}',
  'features.env_remove': 'Remove the environment', 'features.env_removed': 'Removed ({size} freed).', 'features.env_packages_n_other': '{n} packages',
}, 'en');

/** Calls fn with the API answering by method and path; returns the [method, url, body] asked. */
async function withApi(answers, fn) {
  const asked = [];
  const real = globalThis.fetch;
  globalThis.fetch = async (url, init = {}) => {
    const method = init.method || 'GET';
    asked.push([method, url, init.body || null]);
    const body = answers[`${method} ${url}`];
    return { ok: body !== undefined, status: body !== undefined ? 200 : 404, text: async () => JSON.stringify(body ?? {}) };
  };
  try { await fn(); } finally { globalThis.fetch = real; }
  return asked;
}

test('fiftyoneMissing is false while unknown and true only when the server says it is not installed (#658)', async () => {
  f.setFeatures(null);
  assert.equal(f.fiftyoneMissing(), false);
  await withApi({ 'GET /api/features': MISSING }, () => f.loadFeatures(true));
  assert.equal(f.fiftyoneMissing(), true);
  assert.equal(f.fiftyoneInstall(), 'pip install "activelabel[fiftyone]"');
  assert.equal(f.sayFiftyoneMissing(), true);
  assert.match(document.getElementById('toasts').textContent, /Install it with: pip install "activelabel\[fiftyone\]"/);
  f.setFeatures({ ...MISSING, fiftyone: { ...MISSING.fiftyone, installed: true, version: '1.4.0' } });
  assert.equal(f.fiftyoneMissing(), false);
  assert.equal(f.sayFiftyoneMissing(), false);
});

test('the missing note shows the command with Copy and a way to the installed parts (#658)', () => {
  f.setFeatures(MISSING);
  const note = f.fiftyoneMissingNote();
  assert.equal(note.querySelector('.cmd-box code').textContent, 'pip install "activelabel[fiftyone]"');
  assert.ok(note.querySelector('.cmd-box button'));
  assert.ok(note.querySelector('.link-btn'));
});

test('the dialog shows FiftyOne and the training environment, and removes the environment after confirming (#658)', async () => {
  const after = { ...MISSING, training_environment: { exists: false, path: '/ws/training-env', bytes: 0, packages: {} } };
  const asked = await withApi({ 'GET /api/features': MISSING, 'DELETE /api/training/environment': after }, async () => {
    await f.openFeatures();
    const dlg = document.querySelector('dialog[open]');
    const text = dlg.textContent;
    assert.match(text, /Not installed/);
    assert.match(text, /Open in FiftyOne/);
    assert.match(text, /Takes 3/);
    assert.match(text, /2 packages/);
    dlg.querySelector('#feat-env-remove').click();
    await new Promise((resolve) => setTimeout(resolve, 10));
    const confirm = [...document.querySelectorAll('dialog[open]')].pop();
    confirm.returnValue = 'ok';   // the OK button of a method="dialog" form (jsdom does not submit it)
    confirm.close();
    await new Promise((resolve) => setTimeout(resolve, 30));
    assert.equal(document.querySelector('#feat-env-remove'), null, 'nothing left to remove');
    dlg.close();
  });
  const del = asked.find(([m]) => m === 'DELETE');
  assert.deepEqual([del[1], del[2]], ['/api/training/environment', '{"confirm":true}']);
});

test('a FiftyOne button is dimmed and aria-disabled without FiftyOne, and back when it is installed (#675)', () => {
  f.setFeatures(null);
  const button = document.createElement('button');
  document.body.append(button);
  f.fiftyoneButton(button);
  assert.equal(button.getAttribute('aria-disabled'), null, 'unknown: as installed');
  f.setFeatures(MISSING);
  assert.equal(button.getAttribute('aria-disabled'), 'true');
  assert.ok(button.classList.contains('is-unavailable'));
  assert.equal(button.disabled, false, 'a tap still says why');
  assert.match(button.title, /pip install/);
  f.setFeatures({ ...MISSING, fiftyone: { ...MISSING.fiftyone, installed: true } });
  assert.equal(button.getAttribute('aria-disabled'), null);
  assert.ok(!button.classList.contains('is-unavailable'));
  button.remove();
});

test('a 501 fiftyone_not_installed marks FiftyOne as missing and keeps the command in view (#675)', () => {
  f.setFeatures({ ...MISSING, fiftyone: { ...MISSING.fiftyone, installed: true } });
  const button = f.fiftyoneButton(document.createElement('button'));
  document.body.append(button);
  assert.equal(f.fiftyoneError(new Error('other')), false);
  const err = Object.assign(new Error('missing'), { code: 'fiftyone_not_installed', params: { install: 'pip install x' } });
  assert.equal(f.fiftyoneError(err), true);
  assert.equal(f.fiftyoneMissing(), true);
  assert.equal(button.getAttribute('aria-disabled'), 'true');
  assert.match(document.getElementById('toasts').lastElementChild.textContent, /pip install x/);
  button.remove();
});

test('a FiftyOne button made before its panel is in the page is still marked when the answer comes (#675)', () => {
  f.setFeatures(null);
  const early = f.fiftyoneButton(document.createElement('button'));   // not in the page yet
  f.setFeatures(MISSING);
  assert.equal(early.getAttribute('aria-disabled'), 'true');
});
