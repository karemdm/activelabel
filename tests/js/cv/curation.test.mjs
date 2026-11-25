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
const { curationPanel } = await import('../../../src/activelabel/web/static/js/cv/curation.js');

setMessages({ 'step.curation': 'Curation', 'cur.not_analyzed': 'The images have not been analyzed yet.' }, 'en');

test('the Curation step loads the analysis, updates the step bar and draws the not-analyzed notice', async () => {
  let stepper = 0;
  const ctx = { project: { id: 6, counts: { active: 2 } }, jobs: [], panels: {}, watchers: [] };
  const page = {
    ctx, projectId: 6, renderStepper: () => { stepper += 1; }, refreshProject: async () => true,
    jobStarted: async () => {}, kindActive: () => false,
  };
  const panel = curationPanel(page);
  assert.equal(panel.node.id, 'panel-curation');
  document.body.appendChild(panel.node);
  const asked = [];
  const real = globalThis.fetch;
  globalThis.fetch = async (url) => {
    asked.push(url.split('?')[0]);
    const body = url.includes('/images') ? { total: 0, items: [] } : { totals: { measured: 0 }, issue_ids: {}, thresholds: {} };
    return { ok: true, status: 200, text: async () => JSON.stringify(body) };
  };
  try {
    await panel.mount();
    await new Promise((resolve) => setTimeout(resolve, 20));
  } finally {
    globalThis.fetch = real;
  }
  assert.equal(asked[0], '/api/projects/6/curation');
  assert.ok(asked.includes('/api/projects/6/images'));
  assert.equal(stepper, 1);
  assert.equal(ctx.project.counts.analyzed, 0);
  assert.match(panel.node.textContent, /not been analyzed/);
  panel.node.remove();
});
