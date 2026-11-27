import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' });
for (const [name, value] of Object.entries({
  window, document: window.document, Node: window.Node, localStorage: window.localStorage,
})) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const { annotateBlock } = await import('../../../src/activelabel/web/static/js/integrations/external.js');

setMessages({ 'annotate.tool.cvat': 'CVAT', 'annotate.tool.fiftyone': 'FiftyOne' }, 'en');

const ANSWERS = {
  '/api/annotation-tools': [{ tool: 'cvat', configured: false }, { tool: 'fiftyone', configured: true }],
  '/api/projects/2/annotate': { sessions: [] },
  '/api/projects/2/tasks': { tasks: [] },
  '/api/projects/2/preannotations': { models: [], runs: [] },
  '/api/projects/2/curation': { issues: {} },
};

test('the CVAT and FiftyOne tabs load the tools, sessions, tasks, pre-annotations and issues of the project', async () => {
  const ctx = { project: { id: 2, task: 'detection', classes: ['cat'], counts: { active: 3 } }, jobs: [], links: [], panels: {}, watchers: [], syncDismissed: new Set(), syncTool: {} };
  const page = {
    ctx, projectId: 2, isActive: () => false, jobStarted: async () => {}, refreshProject: async () => true,
    startSync: async () => {}, syncBlock: () => ({ node: document.createElement('div'), draw: () => {}, reveal: () => {} }),
    diffView: () => null, issueName: (x) => x,
  };
  const asked = [];
  const real = globalThis.fetch;
  globalThis.fetch = async (url) => {
    asked.push(url);
    const body = ANSWERS[url];
    return { ok: body !== undefined, status: body !== undefined ? 200 : 404, text: async () => JSON.stringify(body ?? {}) };
  };
  try {
    const block = annotateBlock(page);
    assert.ok(block.cvatSection && block.fiftyoneSection);
    await block.mount();
  } finally {
    globalThis.fetch = real;
  }
  assert.deepEqual([...new Set(asked)].sort(), Object.keys(ANSWERS).sort());
});

test('the CVAT card shows no link to a project deleted in CVAT, and the open row says what to do here (#695)', async () => {
  setMessages({ 'annotate.tool.cvat': 'CVAT', 'sync.remote_deleted': '{name} (deleted in CVAT)', 'sync.remote_project': 'Project in {tool}', 'open.cvat_remote_gone': 'Use Unlink, or compare again.' }, 'en');
  const link = { tool: 'cvat', url: 'http://cvat.test/projects/7', remote_id: '7', remote_name: 'Pets', images: 3 };
  const failed = { id: 'j1', kind: 'sync_plan', status: 'failed', created: '2026-11-27T10:00:00', error_code: 'remote_gone' };
  const ctx = { project: { id: 2, task: 'detection', classes: ['cat'], counts: { active: 3 } }, jobs: [failed], links: [link], panels: {}, watchers: [], syncDismissed: new Set(), syncTool: {} };
  const page = {
    ctx, projectId: 2, isActive: () => false, isRemoteGone: (job) => job.error_code === 'remote_gone', jobStarted: async () => {}, refreshProject: async () => true,
    startSync: async () => {}, syncBlock: () => ({ node: document.createElement('div'), draw: () => {}, reveal: () => {} }),
    diffView: () => null, issueName: (x) => x,
  };
  const answers = { ...ANSWERS, '/api/annotation-tools': [{ tool: 'cvat', configured: true, url: 'http://cvat.test', username: 'u' }, { tool: 'fiftyone', configured: true }] };
  const real = globalThis.fetch;
  globalThis.fetch = async (url) => ({ ok: answers[url] !== undefined, status: answers[url] !== undefined ? 200 : 404, text: async () => JSON.stringify(answers[url] ?? {}) });
  try {
    const block = annotateBlock(page);
    document.body.append(block.cvatSection);
    await block.mount();
    const facts = block.cvatSection.querySelector('.link-facts');
    assert.equal(facts.querySelector('a'), null);
    assert.match(facts.textContent, /Pets \(deleted in CVAT\)/);
    assert.match(block.cvatSection.querySelector('#cvat-open-reason').textContent, /Use Unlink, or compare again\./);
  } finally {
    globalThis.fetch = real;
    document.body.replaceChildren();
  }
});
