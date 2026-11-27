import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body></body></html>');
for (const [name, value] of Object.entries({ window, document: window.document, Node: window.Node })) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const tools = await import('../../../src/activelabel/web/static/js/integrations/tools.js');

const { toolName, statusText, jobMessage, statusLabel, isHttp, serverName, imagesOf, originBadge, linkBadge, isUrlError } = tools;

setMessages({
  'annotate.tool.cvat': 'CVAT', 'annotate.tool.fiftyone': 'FiftyOne', 'annotate.tool_removed': 'a discontinued tool',
  'job.status.running': 'Running', 'annotate.status.open': 'Open', 'jobmsg.reading': 'Reading {tool}',
  'jobmsg.Saving': 'Saving…', 'project.imported_from': 'Imported from {tool}', 'project.origin_removed': 'Imported',
  'project.linked_to': 'Linked to {tool}', 'project.link_removed': 'Linked to a discontinued tool',
}, 'en');

test('tools are named; a tool this version no longer has is named generically', () => {
  assert.deepEqual(tools.TOOLS, ['fiftyone', 'cvat']);
  assert.deepEqual(tools.SESSION_TOOLS, ['fiftyone']);
  assert.equal(toolName('cvat'), 'CVAT');
  assert.equal(toolName('labelstudio'), 'a discontinued tool');
});

test('statuses and progress messages are translated, with the tool name matched by shape', () => {
  assert.equal(statusText('running'), 'Running');
  assert.equal(statusText('odd'), 'odd');
  assert.equal(statusLabel('open'), 'Open');
  assert.equal(statusLabel(undefined), '?');
  assert.equal(jobMessage({ message: 'Reading CVAT' }), 'Reading CVAT');
  assert.equal(jobMessage({ message: 'Reading fiftyone' }), 'Reading FiftyOne');
  assert.equal(jobMessage({ message: 'Saving' }), 'Saving…');
  assert.equal(jobMessage({ message: 'Something new' }), 'Something new');
  assert.equal(jobMessage(null), '');
});

test('addresses: only http(s) is a link, the host names the server', () => {
  assert.ok(isHttp('https://app.cvat.ai'));
  assert.ok(!isHttp('javascript:alert(1)'));
  assert.ok(!isHttp(null));
  assert.equal(serverName('https://cvat.example.org:8080/tasks'), 'cvat.example.org:8080');
  assert.equal(serverName('not a url'), 'not a url');
  assert.equal(imagesOf({ images: [1, 2] }), 2);
  assert.equal(imagesOf({ images: '3' }), 3);
  assert.equal(imagesOf({}), 0);
});

test('origin and link badges name the tool, and a removed one generically', () => {
  assert.equal(originBadge(null), null);
  const badge = originBadge({ tool: 'cvat', name: 'Remote cats' });
  assert.equal(badge.textContent, 'Imported from CVAT');
  assert.equal(badge.getAttribute('title'), 'Remote cats');
  assert.equal(badge.getAttribute('aria-label'), 'Imported from CVAT: Remote cats');
  assert.equal(originBadge({ tool: 'labelstudio' }).textContent, 'Imported');
  assert.equal(linkBadge({ tool: 'cvat' }).textContent, 'Linked to CVAT');
  assert.equal(linkBadge({ tool: 'gone', remote_name: 'x' }).textContent, 'Linked to a discontinued tool');
});

test('errors about the address are told apart, so they stay under the field (#634)', () => {
  for (const code of ['url_invalid', 'url_has_credentials', 'url_required', 'cvat_url_invalid']) assert.equal(isUrlError({ code }), true);
  assert.equal(isUrlError(Object.assign(new Error('x'), { urlField: true })), true);
  assert.equal(isUrlError({ code: 'cvat_login_failed' }), false);
  assert.equal(isUrlError(null), false);
});

/* a <dialog> that jsdom can open and close */
const dialogProto = window.HTMLDialogElement && window.HTMLDialogElement.prototype;
if (dialogProto && !dialogProto.showModal) dialogProto.showModal = function showModal() { this.setAttribute('open', ''); };
if (dialogProto && !dialogProto.close) dialogProto.close = function close() { this.removeAttribute('open'); };

test('a test that fails after the address was saved still refreshes the page with the new address (#634)', async () => {
  const calls = [];
  globalThis.fetch = async (url, init) => {
    calls.push(`${init.method} ${url}`);
    if (init.method === 'PUT') return { ok: true, status: 200, text: async () => JSON.stringify({ tool: 'cvat', url: 'http://b', configured: true }) };
    return { ok: false, status: 400, text: async () => JSON.stringify({ detail: 'Set up CVAT first.', code: 'tool_not_configured', params: {} }) };
  };
  let changed = 0;
  tools.openToolSetup('cvat', { url: 'http://a', username: 'u', configured: true }, async () => { changed += 1; });
  const dlg = document.querySelector('dialog[open]');
  dlg.querySelector('#tool-url').value = 'http://b';
  const testBtn = [...dlg.querySelectorAll('button')].find((b) => b.textContent === 'annotate.test');
  testBtn.click();
  for (let i = 0; i < 20 && !changed; i += 1) await new Promise((r) => setTimeout(r, 5));
  assert.deepEqual(calls, ['PUT /api/annotation-tools/cvat', 'POST /api/annotation-tools/cvat/test']);
  assert.equal(changed, 1);
  assert.match(dlg.querySelector('#tool-result').className, /bad/);
  dlg.remove();
  delete globalThis.fetch;
});
