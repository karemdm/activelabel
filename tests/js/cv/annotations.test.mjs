import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' });
for (const [name, value] of Object.entries({
  window, document: window.document, Node: window.Node, localStorage: window.localStorage, sessionStorage: window.sessionStorage,
})) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const { annotationsPanel } = await import('../../../src/activelabel/web/static/js/cv/annotations.js');

setMessages({ 'step.annotations': 'Annotations' }, 'en');

test('the Annotations step is a hidden panel with its tabs, and it can open pre-annotation on a model', () => {
  const tabs = [];
  const page = {
    ctx: { project: { id: 1, task: 'detection', classes: [], counts: {} }, jobs: [], links: [], panels: {}, watchers: [], syncDismissed: new Set(), syncTool: {} },
    projectId: 1, refreshProject: async () => true,
    tabsView: (key, label, list) => {
      tabs.push(...list.map((x) => x.id));
      return { node: document.createElement('div'), select: (id) => tabs.push(`selected:${id}`), current: () => list[0].id, setCount: () => {} };
    },
    isActive: () => false, jobStarted: async () => {}, startSync: async () => {}, diffView: () => null, issueName: (x) => x,
    syncBlock: () => ({ node: document.createElement('div'), draw: () => {}, reveal: () => {} }),
  };
  const panel = annotationsPanel(page);
  assert.equal(panel.node.id, 'panel-annotations');
  assert.equal(panel.node.hidden, true);
  assert.ok(tabs.includes('preannotate') && tabs.includes('cvat') && tabs.includes('fiftyone'));
  panel.preannotateWith(3);
  assert.ok(tabs.includes('selected:preannotate'));
});

test('a regression project has no CVAT tab, even though the project is read after the panel is built (#619)', () => {
  const hidden = [];
  const page = {
    ctx: { project: null, jobs: [], links: [], panels: {}, watchers: [], syncDismissed: new Set(), syncTool: {} },
    projectId: 1, refreshProject: async () => true,
    tabsView: () => ({ node: document.createElement('div'), select: () => {}, hide: (id) => hidden.push(id), current: () => 'summary', setCount: () => {} }),
    isActive: () => false, jobStarted: async () => {}, startSync: async () => {}, diffView: () => null, issueName: (x) => x,
    syncBlock: () => ({ node: document.createElement('div'), draw: () => {}, reveal: () => {} }),
  };
  const panel = annotationsPanel(page);
  page.ctx.project = { id: 1, task: 'detection', classes: [], counts: {} };
  panel.refresh();
  assert.deepEqual(hidden, [], 'other types keep the CVAT tab');
  page.ctx.project = { id: 1, task: 'regression', classes: ['age'], counts: {} };
  panel.refresh();
  assert.deepEqual(hidden, ['cvat']);
});

test('the import report comes back after a reload, with one message for the unmatched annotations', async () => {
  setMessages({ 'ann.report_title': 'Check of the files', 'problem.no_images_matched': 'Nothing matches', 'ann.unmatched_other': '{n} unmatched' }, 'en');
  const page = {
    ctx: { project: { id: 1, task: 'detection', classes: [], counts: {} }, jobs: [], links: [], panels: {}, watchers: [], syncDismissed: new Set(), syncTool: {} },
    projectId: 1, refreshProject: async () => true,
    tabsView: (key, label, list) => {
      const node = document.createElement('div');
      list.forEach((x) => (x.nodes || []).forEach((n) => node.append(n)));
      return { node, select: () => {}, current: () => list[0].id, setCount: () => {} };
    },
    isActive: () => false, jobStarted: async () => {}, startSync: async () => {}, diffView: () => null, issueName: (x) => x,
    syncBlock: () => ({ node: document.createElement('div'), draw: () => {}, reveal: () => {} }),
  };
  const report = {
    token: 'a'.repeat(16), format: 'coco', inferred_task: 'detection', counts: {}, classes_found: [], options: ['discard'],
    problems: [{ level: 'error', code: 'no_images_matched', message: 'x', count: 3, examples: ['a.png'] }],
    unmatched: { count: 3, examples: ['a.png'] },
  };
  const real = globalThis.fetch;
  globalThis.fetch = async (url) => {
    const body = String(url).endsWith('/annotations/pending') ? { report } : {};
    return { ok: true, status: 200, text: async () => JSON.stringify(body) };
  };
  try {
    const panel = annotationsPanel(page);
    document.body.append(panel.node);
    panel.mount();
    await new Promise((resolve) => setTimeout(resolve, 30));
    const text = panel.node.querySelector('.ann-report').textContent;
    assert.ok(text.includes('Check of the files') && text.includes('Nothing matches'));
    assert.ok(!text.includes('3 unmatched'));  // the problem row already says it
  } finally {
    globalThis.fetch = real;
  }
});

const reportPage = () => ({
  ctx: { project: { id: 1, task: 'detection', classes: [], counts: {} }, jobs: [], links: [], panels: {}, watchers: [], syncDismissed: new Set(), syncTool: {} },
  projectId: 1, refreshProject: async () => true,
  tabsView: (key, label, list) => {
    const node = document.createElement('div');
    list.forEach((x) => (x.nodes || []).forEach((n) => node.append(n)));
    return { node, select: () => {}, current: () => list[0].id, setCount: () => {} };
  },
  isActive: () => false, jobStarted: async () => {}, startSync: async () => {}, diffView: () => null, issueName: (x) => x,
  syncBlock: () => ({ node: document.createElement('div'), draw: () => {}, reveal: () => {} }),
});
const mkReport = (extra) => ({
  token: 'b'.repeat(16), format: 'coco', inferred_task: 'detection', counts: {}, classes_found: [], options: ['discard'],
  problems: [], ...extra,
});
const withFetch = async (handler, body) => {
  const real = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    const out = handler(String(url), init || {});
    return { ok: true, status: 200, text: async () => JSON.stringify(out) };
  };
  try { await body(); } finally { globalThis.fetch = real; }
};
const wait = () => new Promise((resolve) => setTimeout(resolve, 30));
const mountPanel = async () => {
  const panel = annotationsPanel(reportPage());
  document.body.append(panel.node);
  panel.mount();
  await wait();
  return panel;
};
const checkFiles = async (panel) => {
  const input = panel.node.querySelector('input[type=file]');
  Object.defineProperty(input, 'files', { value: [new File(['{}'], 'a.json')], configurable: true });
  input.dispatchEvent(new window.Event('change'));
  await wait();
};
const KEY = 'al-ann-report-1';

test('a report that cannot be imported is kept and shown again by a new panel (#651)', async () => {
  setMessages({ 'ann.report_title': 'Check of the files', 'problem.no_images_matched': 'Nothing matches' }, 'en');
  window.sessionStorage.clear();
  const bad = mkReport({ can_import: false, problems: [{ level: 'error', code: 'no_images_matched', message: 'x', count: 3, examples: [] }] });
  await withFetch((url, init) => (init.method === 'POST' ? bad : { report: null }), async () => {
    const first = await mountPanel();
    await checkFiles(first);
    assert.ok(first.node.querySelector('.ann-report').textContent.includes('Nothing matches'));
    assert.equal(JSON.parse(window.sessionStorage.getItem(KEY)).token, bad.token);
    first.node.remove();
    const second = await mountPanel();  // the server has no pending report
    const text = second.node.querySelector('.ann-report').textContent;
    assert.ok(text.includes('Check of the files') && text.includes('Nothing matches'));
    second.node.remove();
  });
});

test('a report that can be imported removes the kept one (#651)', async () => {
  window.sessionStorage.setItem(KEY, JSON.stringify(mkReport({ can_import: false })));
  await withFetch(() => mkReport({ can_import: true, options: ['import', 'discard'] }), async () => {
    const panel = await mountPanel();
    await checkFiles(panel);
    assert.equal(window.sessionStorage.getItem(KEY), null);
    panel.node.remove();
  });
});

test('discarding a report removes the kept one (#651)', async () => {
  setMessages({ 'option.discard': 'Discard' }, 'en');
  const bad = mkReport({ can_import: false });
  window.sessionStorage.setItem(KEY, JSON.stringify(bad));
  await withFetch(() => ({ report: null }), async () => {
    const panel = await mountPanel();  // shown from the kept report
    const buttons = [...panel.node.querySelectorAll('.ann-report button')];
    assert.ok(buttons.length > 0);
    const discard = buttons.find((b) => b.textContent.includes('Discard')) || buttons[buttons.length - 1];
    discard.click();
    await wait();
    assert.equal(window.sessionStorage.getItem(KEY), null);
    panel.node.remove();
  });
});

test('classes the file names with no label are not listed, and a hint counts them (#651)', async () => {
  setMessages({ 'ann.classes_unused_one': '{n} class left out', 'ann.classes_unused_other': '{n} classes left out' }, 'en');
  const make = (found) => mkReport({
    can_import: true, options: ['import', 'discard'], classes_found: found,
    counts: { images_matched: 2, labels_total: 3, labels_per_class: { cat: 3, dog: 0, bird: 0 } },
  });
  for (const [found, chips, hint] of [[['cat', 'dog', 'bird'], 1, '2 classes left out'], [['cat', 'dog'], 1, '1 class left out'], [['cat'], 1, null]]) {
    window.sessionStorage.clear();
    await withFetch(() => ({ report: make(found) }), async () => {
      const panel = await mountPanel();
      const names = [...panel.node.querySelectorAll('.ann-report .chip.static')].map((li) => li.firstChild.textContent);
      assert.deepEqual(names, ['cat']);
      assert.equal(names.length, chips);
      const hints = [...panel.node.querySelectorAll('.ann-report p.hint')].map((p) => p.textContent);
      if (hint) assert.deepEqual(hints, [hint]); else assert.deepEqual(hints, []);
      panel.node.remove();
    });
  }
});
