import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/projects/1' });
for (const [name, value] of Object.entries({
  window, document: window.document, Node: window.Node, localStorage: window.localStorage,
  location: window.location, history: window.history, requestAnimationFrame: (fn) => setTimeout(fn, 0),
})) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const { createStepper } = await import('../../../src/activelabel/web/static/js/cv/stepper.js');

setMessages({ 'step.images': 'Images', 'step.curation': 'Curation', 'gate.need_images': 'Add images first' }, 'en');

const STEPS = ['images', 'annotations', 'curation', 'datasets', 'training', 'models'];

/** A stepper over a project with `counts`, and six stand-in panels that record their mounts. */
function makeStepper(counts) {
  const mounted = [];
  const panels = Object.fromEntries(STEPS.map((s) => [s, { node: document.createElement('section'), mounted: false, mount: () => mounted.push(s) }]));
  const ctx = { project: { id: 1, counts }, tab: 'images', panels, jobs: [] };
  const stepper = document.createElement('nav');
  document.body.appendChild(stepper);
  return { ctx, stepper, mounted, parts: createStepper({ ctx, projectId: 1, STEPS, stepper }) };
}

test('showTab marks the current step, mounts its panel once and hides the others', () => {
  const { ctx, stepper, mounted, parts } = makeStepper({ active: 3 });
  parts.showTab('curation');
  assert.equal(ctx.tab, 'curation');
  assert.equal(stepper.querySelector('.step[aria-current="step"]').dataset.step, 'curation');
  assert.deepEqual(mounted, ['curation']);
  assert.equal(ctx.panels.curation.node.hidden, false);
  assert.equal(ctx.panels.images.node.hidden, true);
  assert.equal(stepper.querySelectorAll('.step').length, 6);
  stepper.remove();
});

test('without images the data steps show the card that sends the person to step 1', () => {
  const { ctx, parts } = makeStepper({});
  assert.equal(parts.hasImages(), false);
  parts.showTab('curation');
  assert.equal(parts.gateNode.hidden, false);
  assert.equal(ctx.panels.curation.node.hidden, true);
  assert.equal(ctx.panels.curation.mounted, false);
});
