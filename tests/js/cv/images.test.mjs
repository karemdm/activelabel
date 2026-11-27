import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' });
for (const [name, value] of Object.entries({
  window, document: window.document, Node: window.Node, localStorage: window.localStorage,
})) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const proto = window.HTMLDialogElement.prototype;
if (!proto.showModal) proto.showModal = function showModal() { this.setAttribute('open', ''); };
if (!proto.close) proto.close = function close() { this.removeAttribute('open'); };
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const { imagesPanel } = await import('../../../src/activelabel/web/static/js/cv/images.js');

setMessages({ 'step.images': 'Images' }, 'en');

/** The page object with a tabs stand-in that records what is selected. */
function makePage(hasImages) {
  const selected = [];
  const ctx = { project: { id: 2, task: 'detection', classes: [], counts: {} }, jobs: [], panels: {}, watchers: [], busy: new Set() };
  const page = {
    ctx, projectId: 2, hasImages: () => hasImages, jobStarted: async () => {}, refreshCounts: async () => {},
    refreshProject: async () => true, showTab: () => {},
    tabsView: () => ({ node: document.createElement('div'), remembered: false, select: (id) => selected.push(id), setCount: () => {}, current: () => 'gallery' }),
  };
  return { page, selected };
}

test('the Images step opens on its tasks when the project has no images, and loads tasks and images', async () => {
  const { page, selected } = makePage(false);
  const panel = imagesPanel(page);
  assert.equal(panel.node.id, 'panel-images');
  const asked = [];
  const real = globalThis.fetch;
  globalThis.fetch = async (url) => {
    asked.push(url.split('?')[0]);
    const body = url.includes('/tasks') ? { tasks: [] } : { total: 0, items: [] };
    return { ok: true, status: 200, text: async () => JSON.stringify(body) };
  };
  try {
    panel.mount();
    await new Promise((resolve) => setTimeout(resolve, 30));
  } finally {
    globalThis.fetch = real;
  }
  assert.deepEqual(selected, ['tasks']);
  assert.ok(asked.includes('/api/projects/2/tasks'));
  assert.ok(asked.includes('/api/projects/2/images'));
});

test('a deleted dataset version stops filtering the gallery', () => {
  const { page, selected } = makePage(true);
  const panel = imagesPanel(page);
  const real = globalThis.fetch;
  globalThis.fetch = async () => ({ ok: true, status: 200, text: async () => JSON.stringify({ total: 0, items: [] }) });
  try {
    panel.showDataset(4, 'train');
    assert.deepEqual(selected, ['gallery']);
    panel.forgetDataset(4);
  } finally {
    globalThis.fetch = real;
  }
});

/** The tasks panel with `task` listed, `links` linked and a CVAT tab stand-in that records what it is asked. */
async function panelWithTask(task, links, calls) {
  const { page } = makePage(true);
  page.ctx.links = links;
  page.ctx.panels = { annotations: { focusCvatTask: (id) => calls.push(['focus', id]), reveal: (what) => calls.push(['reveal', what]) } };
  page.showTab = (step) => calls.push(['tab', step]);
  page.tabsView = (id, label, tabs) => {  // the tab contents are shown, so the task list is in the panel
    const node = document.createElement('div');
    for (const tab of tabs) node.append(...tab.nodes);
    return { node, remembered: true, select: () => {}, setCount: () => {}, current: () => 'tasks' };
  };
  const panel = imagesPanel(page);
  const real = globalThis.fetch;
  globalThis.fetch = async (url) => {
    const body = url.includes('/tasks') ? { tasks: [task] } : { total: 0, items: [] };
    return { ok: true, status: 200, text: async () => JSON.stringify(body) };
  };
  try {
    panel.mount();
    await new Promise((resolve) => setTimeout(resolve, 30));
  } finally {
    globalThis.fetch = real;
  }
  return panel;
}

const TASK = { id: 7, name: 'T7', images: 2, jobs: [], open: false, incomplete: false, annotated: 0 };
const CVAT_LINK = { tool: 'cvat', url: 'https://cvat.example/projects/3' };

/** The CVAT item of the open menu of the task card. */
function cvatItem(panel) {
  document.body.appendChild(panel.node);
  panel.node.querySelector('.task-list .menu-wrap button').click();
  const items = [...panel.node.querySelectorAll('.menu [role="menuitem"]')];
  const item = items.find((x) => /menu_cvat/.test(x.textContent));
  panel.node.remove();
  return item;
}

test('what a comparison can fix leads from the open menu to the CVAT tab, with its reason', async () => {
  for (const cvat of ['not_linked', 'not_synchronised', 'removed_remote', 'remote_gone']) {
    const calls = [];
    const panel = await panelWithTask({ ...TASK, cvat }, [CVAT_LINK], calls);
    const item = cvatItem(panel);
    assert.ok(item, cvat);
    assert.equal(item.tagName, 'BUTTON', cvat);
    assert.match(item.textContent, /tasks\.menu_cvat_go/);
    assert.match(item.textContent, new RegExp(cvat === 'not_synchronised' ? 'tasks.cvat_not_synced' : `tasks.cvat_${cvat}`));
    item.click();
    assert.deepEqual(calls, [['tab', 'annotations'], ['focus', 7], ['reveal', 'sync']], cvat);
  }
});

test('a connection not set up, or a CVAT that does not answer, stay disabled with the reason', async () => {
  const reasons = { not_configured: 'tasks.cvat_not_configured', unreachable: 'tasks.cvat_unreachable' };
  for (const cvat of ['not_configured', 'unreachable']) {
    const calls = [];
    const item = cvatItem(await panelWithTask({ ...TASK, cvat }, cvat === 'unreachable' ? [CVAT_LINK] : [], calls));
    assert.equal(item.tagName, 'SPAN', cvat);
    assert.equal(item.getAttribute('aria-disabled'), 'true');
    assert.match(item.textContent, new RegExp(reasons[cvat]));
    assert.deepEqual(calls, []);
  }
});

test('a task that is in CVAT opens there with a link', async () => {
  const task = { ...TASK, cvat: 'ready', remote: { task: 40, url: 'https://cvat.example/tasks/40', jobs: [] } };
  const item = cvatItem(await panelWithTask(task, [CVAT_LINK], []));
  assert.equal(item.tagName, 'A');
  assert.equal(item.getAttribute('href'), 'https://cvat.example/tasks/40');
});

test('a project not linked yet still leads to the CVAT tab', async () => {
  const calls = [];
  const item = cvatItem(await panelWithTask({ ...TASK, cvat: 'not_linked' }, [], calls));
  assert.equal(item.tagName, 'BUTTON');
  assert.match(item.textContent, /tasks.cvat_not_linked/);  // the task's own reason, not the project's
  item.click();
  assert.deepEqual(calls, [['tab', 'annotations'], ['focus', 7], ['reveal', 'sync']]);
});

test('deleting a task that is in CVAT says that it stays there (#684)', async () => {
  const proto = window.HTMLDialogElement.prototype;
  if (!proto.showModal) proto.showModal = function showModal() { this.setAttribute('open', ''); };
  setMessages({ 'tasks.delete': 'Delete', 'tasks.delete_text_other': 'The task and its {n} images go.', 'tasks.delete_text_cvat': 'In CVAT the task stays.' }, 'en');
  const texts = [];
  for (const task of [{ ...TASK, cvat: 'not_linked' }, { ...TASK, cvat: 'ready', remote: { task: 40, url: 'https://cvat.example/tasks/40', jobs: [] } }]) {
    const panel = await panelWithTask(task, [CVAT_LINK], []);
    document.body.appendChild(panel.node);
    panel.node.querySelector('.task-list .icon-btn.danger, .task-list button.danger').click();
    const confirm = document.querySelector('dialog[aria-label="tasks.delete_title"]');
    texts.push(confirm.querySelector('p').textContent);
    confirm.remove();
    panel.node.remove();
  }
  assert.doesNotMatch(texts[0], /CVAT/);
  assert.match(texts[1], /In CVAT the task stays\./);
});

test('a task card shows the last edit session: tool, when and state (#526)', async () => {
  setMessages({ 'tasks.last_session': 'Last session: {tool}, {when} ({status})', 'annotate.tool.fiftyone': 'FiftyOne', 'annotate.status.applied': 'Applied' }, 'en');
  const task = { ...TASK, last_session: { tool: 'fiftyone', at: '2026-11-26T10:20:00', status: 'applied' } };
  const panel = await panelWithTask(task, [], []);
  assert.match(panel.node.querySelector('.task-last-session').textContent, /Last session: FiftyOne, .*2026.* \(applied\)/);
  const plain = await panelWithTask(TASK, [], []);
  assert.equal(plain.node.querySelector('.task-last-session'), null);
});

/** The page with a tabs stand-in that puts the tab contents in its node (so the gallery is in the DOM). */
function withTabs(made) {
  made.page.tabsView = (id, label, tabs) => {
    const node = document.createElement('div');
    tabs.forEach((tab) => tab.nodes.forEach((n) => n && node.append(n)));
    return { node, remembered: false, select: () => {}, setCount: () => {}, current: () => 'gallery' };
  };
  return made;
}

test('#706: "Edit in FiftyOne" in the lightbox sends that one image and opens the App when the job ends', async () => {
  const { page } = withTabs(makePage(true));
  page.ctx.openWhenDone = new Map();
  page.ctx.project.counts = { active: 1 };
  const posted = [];
  const real = globalThis.fetch;
  const realOpen = window.open;
  window.open = () => ({ document: { write: () => {}, close: () => {} }, close: () => {}, location: {} });
  globalThis.fetch = async (url, options = {}) => {
    if (options.method === 'POST') {
      posted.push([url, JSON.parse(options.body)]);
      return { ok: true, status: 200, text: async () => JSON.stringify({ id: 'job1', status: 'queued' }) };
    }
    const body = url.includes('/tasks') ? { tasks: [] } : { total: 1, items: [{ id: 'i0', name: 'a.png', status: 'active' }] };
    return { ok: true, status: 200, text: async () => JSON.stringify(body) };
  };
  try {
    const panel = imagesPanel(page);
    document.body.append(panel.node);
    panel.mount();
    await new Promise((resolve) => setTimeout(resolve, 30));
    panel.node.querySelector('.thumb-btn').click();
    const edit = document.querySelector('dialog.lightbox [data-action="edit-fo"]');
    assert.ok(edit, 'the lightbox offers Edit in FiftyOne');
    edit.click();
    await new Promise((resolve) => setTimeout(resolve, 10));
    assert.deepEqual(posted, [['/api/projects/2/annotate', { tool: 'fiftyone', scope: 'selected', ids: ['i0'] }]]);
    assert.ok(page.ctx.openWhenDone.has('job1'));  // the App opens in the reserved tab when the job ends
  } finally {
    globalThis.fetch = real;
    window.open = realOpen;
    document.querySelectorAll('section.panel').forEach((n) => n.remove());
  }
});

test('#706: a keypoints project gets no "Edit in FiftyOne" in the lightbox (the App cannot edit keypoints)', async () => {
  const { page } = withTabs(makePage(true));
  page.ctx.project.task = 'keypoints';
  const real = globalThis.fetch;
  globalThis.fetch = async (url) => {
    const body = url.includes('/tasks') ? { tasks: [] } : { total: 1, items: [{ id: 'i0', name: 'a.png', status: 'active' }] };
    return { ok: true, status: 200, text: async () => JSON.stringify(body) };
  };
  try {
    const panel = imagesPanel(page);
    document.body.append(panel.node);
    panel.mount();
    await new Promise((resolve) => setTimeout(resolve, 30));
    panel.node.querySelector('.thumb-btn').click();
    const box = document.querySelector('dialog.lightbox');
    assert.ok(box.hasAttribute('open'));
    assert.equal(box.querySelector('[data-action="edit-fo"]'), null);
    box.close();
  } finally {
    globalThis.fetch = real;
    document.querySelectorAll('section.panel').forEach((n) => n.remove());
  }
});
