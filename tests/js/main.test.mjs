import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body data-page="none"></body></html>', { url: 'http://localhost/' });
window.I18N = { 'task.detection': 'Detection', 'annotate.tool.cvat': 'CVAT', 'annotate.status.open': 'Open' };
window.VL = { locale: 'en' };
for (const [name, value] of Object.entries({
  window, document: window.document, Node: window.Node, localStorage: window.localStorage,
})) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
await import('../../src/activelabel/web/static/js/main.js');
const i18n = await import('../../src/activelabel/web/static/js/shell/i18n.js');
const { errorParams } = await import('../../src/activelabel/web/static/js/shell/api.js');

test('main.js sets the texts and the locale the server put in the page', () => {
  assert.equal(i18n.LOCALE, 'en');
  assert.equal(i18n.t('task.detection'), 'Detection');
});

test('main.js names the error params that need task, tool and status names', () => {
  assert.deepEqual(errorParams({ task: 'detection', tool: 'cvat', status: 'open' }), { task: 'detection', tool: 'CVAT', status: 'open' });
});
