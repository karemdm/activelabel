/* The similarity map of the Curation step: embeddings of the images (a small network in the training environment),
 * a 2-D map where similar images sit together, and groups of similar images. The same map opens in the FiftyOne
 * App (Embeddings panel). */
import { api } from '../shell/api.js';
import { fiftyoneButton, fiftyoneError, sayFiftyoneMissing } from '../shell/features.js';
import { el, esc, fill, icon } from '../shell/dom.js';
import { num, t, tn, tr } from '../shell/i18n.js';
import { dropTab, reserveTab } from '../shell/newtab.js';
import { toast, toastError } from '../shell/toast.js';

export const GROUP_COLOURS = 8;   // --group-1..8 in app.css; more groups reuse the colours
const ACTIVE = (s) => s.status === 'queued' || s.status === 'running';

/** The colour token of a group (numbered from 0). */
export function groupColour(group) {
  return `var(--group-${(group % GROUP_COLOURS) + 1})`;
}

/** The line that says where a computation is. */
export function statusText(state) {
  if (state.status === 'queued') return t('sim.queued');
  if (state.status === 'running') {
    if (state.phase === 'embedding' && state.total) return t('sim.embedding', { done: num(state.done || 0), total: num(state.total) });
    return tr(`sim.phase.${state.phase || 'environment'}`, t('sim.phase.environment'));
  }
  if (state.status === 'failed') return t('sim.failed', { error: state.error === 'interrupted' ? t('sim.interrupted') : state.error || '' });
  if (state.status === 'cancelled') return t('sim.cancelled');
  return '';
}

/** SVG of the map: one dot per image, coloured by group (y grows upwards). */
export function mapSvg(points) {
  const xy = (points && points.xy) || [];
  const groups = (points && points.group) || [];
  const size = 400;
  const pad = 8;
  const dots = xy.map(([x, y], i) => {
    const cx = (pad + x * (size - 2 * pad)).toFixed(1);
    const cy = (size - pad - y * (size - 2 * pad)).toFixed(1);
    return `<circle cx="${cx}" cy="${cy}" r="4" style="fill:${groupColour(groups[i] || 0)}"></circle>`;
  }).join('');
  const label = t('sim.map_label', { n: num(xy.length) });
  return `<svg viewBox="0 0 ${size} ${size}" role="img" aria-label="${esc(label)}">${dots}</svg>`;
}

/** The similarity block. `page` gives {projectId, ctx, jobStarted}. Returns {node, load}. */
export function similarityBlock(page) {
  const { projectId, ctx, jobStarted } = page;
  const node = el('section', { class: 'sim', 'aria-labelledby': 'h-sim' });
  const live = el('p', { class: 'sr-only', 'aria-live': 'polite' });
  let state = {};
  let timer = null;

  async function load() {
    clearTimeout(timer);
    try {
      state = (await api('GET', `/projects/${projectId}/curation/similarity`)) || {};
    } catch (err) {
      if (err.status !== 404) toastError(err);
      return;
    }
    draw();
    if (ACTIVE(state) && node.isConnected) timer = setTimeout(load, 2000);
  }

  async function start(button) {
    button.disabled = true;
    try {
      state = await api('POST', `/projects/${projectId}/curation/similarity`, { json: {} });
      toast(t('sim.started'), 'info', 4000);
      draw();
      timer = setTimeout(load, 1000);
    } catch (err) {
      toastError(err);
      button.disabled = false;
    }
  }

  async function cancel() {
    try {
      state = await api('POST', `/projects/${projectId}/curation/similarity/cancel`);
      load();
    } catch (err) { toastError(err); }
  }

  async function openApp() {
    if (sayFiftyoneMissing()) return;   // #675
    const tab = ctx ? reserveTab(t('cur.fo_waiting')) : null;
    try {
      const job = await api('POST', `/projects/${projectId}/fiftyone`, { json: {} });
      if (job && job.id && ctx) ctx.openWhenDone.set(job.id, tab); else dropTab(tab);
      toast(t('sim.fo_starting'), 'info', 6000);
      if (jobStarted) await jobStarted(job);
    } catch (err) { dropTab(tab); if (!fiftyoneError(err)) toastError(err); }
  }

  function groupList() {
    return el('ul', { class: 'sim-groups' }, ...(state.groups || []).map((g) =>
      el('li', { class: 'sim-group' },
        el('img', { src: `/api/projects/${projectId}/images/${g.typical}/thumb`, alt: t('sim.typical', { n: g.group + 1 }), loading: 'lazy', width: '64', height: '64' }),
        el('span', { class: 'sim-swatch', style: `background:${groupColour(g.group)}`, 'aria-hidden': 'true' }),
        el('span', null, el('strong', null, t('sim.group', { n: g.group + 1 })), ' ', el('span', { class: 'muted' }, tn('sim.images_n', g.size))))));
  }

  function draw() {
    const active = ACTIVE(state);
    const done = state.points && (state.points.ids || []).length > 0;
    const startBtn = el('button', { type: 'button', class: `btn${done ? '' : ' primary'}`, disabled: active, onclick: (e) => start(e.currentTarget) },
      done ? t('sim.recompute') : t('sim.compute'));
    const actions = el('div', { class: 'cur-actions' }, startBtn,
      active ? el('button', { type: 'button', class: 'btn', onclick: cancel }, t('sim.cancel')) : null,
      done && !active ? fiftyoneButton(el('button', { type: 'button', class: 'btn', onclick: openApp }, t('sim.open_fo'))) : null);
    const status = statusText(state);
    live.textContent = status;
    const body = [];
    if (done) {
      const groups = (state.groups || []).length;
      body.push(el('p', null, t('sim.summary', { images: num(state.points.ids.length), groups: num(groups), method: tr(`sim.method.${state.method}`, state.method || '') })));
      if (state.outdated) body.push(el('p', { class: 'sim-note' }, icon('info'), t('sim.outdated')));
      if (state.fiftyone_error) body.push(el('p', { class: 'sim-note' }, icon('info'), t('sim.fo_failed', { error: state.fiftyone_error })));
      body.push(el('div', { class: 'sim-body' }, el('div', { class: 'sim-map', html: mapSvg(state.points) }), groupList()));
      body.push(el('p', { class: 'muted' }, t('sim.fo_hint')));
    }
    fill(node,
      el('h3', { id: 'h-sim' }, t('sim.title')),
      el('p', { class: 'muted' }, t('sim.intro')),
      actions,
      status ? el('p', { class: active ? 'muted' : 'sim-note' }, status) : null,
      ...body,
      live);
  }

  draw();
  return { node, load };
}
