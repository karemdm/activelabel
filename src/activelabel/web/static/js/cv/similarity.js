/* The similarity map of the Curation step: embeddings of the images (a small network in the training environment),
 * a 2-D map where similar images sit together, and groups of similar images. The same map opens in the FiftyOne
 * App (Embeddings panel). */
import { api } from '../shell/api.js';
import { fiftyoneButton, fiftyoneError, sayFiftyoneMissing } from '../shell/features.js';
import { el, esc, fill, icon } from '../shell/dom.js';
import { num, parseNumber, t, tn, tr } from '../shell/i18n.js';
import { confirmDialog, openModal, returnFocus } from '../shell/dialogs.js';
import { dropTab, reserveTab } from '../shell/newtab.js';
import { toast, toastError } from '../shell/toast.js';
import { deviceName } from './names.js';

export const GROUP_COLOURS = 8;   // --group-1..8 in app.css; more groups reuse the colours
export const NO_GROUP = -1;       // the images HDBSCAN leaves without a group (noise)
export const MAX_GROUPS = 50;     // most groups a person may ask for (the server refuses more)
export const MIN_GROUP_SIZE = 2;  // smallest "smallest group" for HDBSCAN
const ACTIVE = (s) => s.status === 'queued' || s.status === 'running';

/** The colour token of a group (numbered from 0); the images without a group have a neutral one. */
export function groupColour(group) {
  if (group === NO_GROUP) return 'var(--group-none)';
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

/** The line that says where a regrouping is. */
export function clusterStatusText(state) {
  if (state.status === 'queued') return t('sim.cluster.queued');
  if (state.status === 'running') return t('sim.cluster.running');
  if (state.status === 'failed') return t('sim.cluster.failed', { error: state.error === 'interrupted' ? t('sim.interrupted') : state.error || '' });
  if (state.status === 'cancelled') return t('sim.cluster.cancelled');
  return '';
}

/** The images of a map that have no group. */
export function noiseCount(points) {
  return ((points && points.group) || []).filter((g) => g === NO_GROUP).length;
}

/** The dialog that asks how to group the map again. Resolves with {method, groups, min_size}, or null when cancelled. */
export function regroupDialog({ images, current = {} }) {
  return new Promise((resolve) => {
    let method = ['auto', 'kmeans', 'hdbscan'].includes(current.method) ? current.method : 'auto';
    let result = null;
    const groupsInput = el('input', { id: 'rg-groups', type: 'text', inputmode: 'numeric', autocomplete: 'off', value: String(current.groups >= 2 ? current.groups : 4), 'aria-describedby': 'rg-error' });
    const sizeInput = el('input', { id: 'rg-size', type: 'text', inputmode: 'numeric', autocomplete: 'off', value: String(current.min_size >= MIN_GROUP_SIZE ? current.min_size : 5), 'aria-describedby': 'rg-error' });
    const error = el('p', { class: 'tr-form-error', id: 'rg-error', role: 'alert', hidden: true });
    const options = el('div', { class: 'rg-options' });
    const choice = (value, extra) => el('div', { class: 'rg-choice' },
      el('label', { class: 'radio-row' },
        el('input', { type: 'radio', name: 'rg-method', value, checked: value === method, onchange: () => { method = value; error.hidden = true; drawOptions(); } }),
        el('span', null, el('strong', null, t(`sim.regroup.${value}`)), el('span', { class: 'muted' }, ` — ${t(`sim.regroup.${value}.help`)}`))),
      value === method ? extra : null);
    function drawOptions() {
      fill(options,
        choice('auto', null),
        choice('kmeans', el('div', { class: 'field inline rg-field' }, el('label', { for: 'rg-groups' }, t('sim.regroup.groups')), groupsInput)),
        choice('hdbscan', el('div', { class: 'field inline rg-field' }, el('label', { for: 'rg-size' }, t('sim.regroup.min_size')), sizeInput, el('p', { class: 'hint' }, t('sim.regroup.noise_hint')))));
    }
    const fail = (key, vars, input) => {
      error.textContent = t(key, vars);
      error.hidden = false;
      input.setAttribute('aria-invalid', 'true');
      input.focus();
    };
    const submit = (e) => {
      e.preventDefault();
      error.hidden = true;
      groupsInput.removeAttribute('aria-invalid');
      sizeInput.removeAttribute('aria-invalid');
      const body = { method, groups: 0, min_size: 5 };
      if (method === 'kmeans') {
        const n = parseNumber(groupsInput.value);
        const most = Math.min(MAX_GROUPS, images || MAX_GROUPS);
        if (!Number.isInteger(n) || n < 2 || n > most) return fail('sim.regroup.groups_invalid', { max: num(most) }, groupsInput);
        body.groups = n;
      } else if (method === 'hdbscan') {
        const n = parseNumber(sizeInput.value);
        if (!Number.isInteger(n) || n < MIN_GROUP_SIZE) return fail('sim.regroup.min_size_invalid', { min: num(MIN_GROUP_SIZE) }, sizeInput);
        body.min_size = n;
      }
      result = body;
      dlg.close('ok');
    };
    const dlg = el('dialog', { class: 'dialog', 'aria-label': t('sim.regroup.title') },
      el('form', { class: 'dialog-body', novalidate: true, onsubmit: submit },
        el('h2', null, t('sim.regroup.title')),
        el('p', { class: 'muted' }, t('sim.regroup.intro')),
        options,
        error,
        el('div', { class: 'form-actions' },
          el('button', { type: 'submit', class: 'btn primary' }, t('sim.regroup.start')),
          el('button', { type: 'button', class: 'btn', onclick: () => dlg.close('cancel') }, t('common.cancel')))));
    drawOptions();
    dlg.addEventListener('close', () => { dlg.remove(); resolve(dlg.returnValue === 'ok' ? result : null); });
    document.body.appendChild(dlg);
    openModal(dlg);
  });
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
    return `<circle cx="${cx}" cy="${cy}" r="4" style="fill:${groupColour(groups[i] ?? 0)}"></circle>`;
  }).join('');
  const label = t('sim.map_label', { n: num(xy.length) });
  return `<svg viewBox="0 0 ${size} ${size}" role="img" aria-label="${esc(label)}">${dots}</svg>`;
}

/** The similarity block. `page` gives {projectId, ctx, jobStarted, onMap}; `onMap({ready, noise})` tells the
 *  step whether a map exists and how many images have no group. Returns {node, load}. */
export function similarityBlock(page) {
  const { projectId, ctx, jobStarted, onMap } = page;
  const node = el('section', { class: 'sim', 'aria-labelledby': 'h-sim' });
  const live = el('p', { class: 'sr-only', 'aria-live': 'polite' });
  let state = {};
  let cluster = {};     // the regrouping of the map (#614)
  let timer = null;
  let reqs = null;      // what the map needs (#664): packages to install, devices, images
  let device = 'auto';

  async function requirements() {
    try {
      reqs = await api('GET', `/projects/${projectId}/curation/similarity/requirements`);
    } catch (err) {
      if (err.status !== 404) toastError(err);
    }
    return reqs;
  }

  async function load() {
    clearTimeout(timer);
    try {
      cluster = (await api('GET', `/projects/${projectId}/curation/similarity/clustering`)) || {};   // first: the groups it made are in the map read next
    } catch (err) {
      if (err.status !== 404) toastError(err);
      return;
    }
    try {
      state = (await api('GET', `/projects/${projectId}/curation/similarity`)) || {};
    } catch (err) {
      if (err.status !== 404) toastError(err);
      return;
    }
    if (!reqs) await requirements();
    draw();
    if ((ACTIVE(state) || ACTIVE(cluster)) && node.isConnected) timer = setTimeout(load, 2000);
  }

  async function regroup(button) {
    const done = (state.points && state.points.ids) || [];
    const body = await regroupDialog({ images: done.length, current: cluster });
    if (!body) { returnFocus(button); return; }
    try {
      cluster = await api('POST', `/projects/${projectId}/curation/similarity/clustering`, { json: body });
      toast(t('sim.regroup.started'), 'info', 4000);
      draw();
      timer = setTimeout(load, 1000);
    } catch (err) {
      toastError(err);
      load();   // a 409 means the state shown was old
    }
  }

  async function cancelRegroup() {
    try {
      cluster = await api('POST', `/projects/${projectId}/curation/similarity/clustering/cancel`);
      load();
    } catch (err) { toastError(err); }
  }

  /* the first map installs PyTorch in the training environment: confirm it like the training dialog does (#664) */
  async function ready() {
    const need = await requirements();
    if (!need) return true;
    if (need.images < need.min_images) {
      toast(t('sim.too_few', { min: num(need.min_images), n: num(need.images) }), 'error', 6000);
      return false;
    }
    if (!need.install_needed) return true;
    return confirmDialog({
      title: t('sim.install_title'),
      text: t('tr.packages_note', { list: need.missing.join(', ') }),
      confirmLabel: t('sim.install_confirm'),
      danger: false,
    });
  }

  async function start(button) {
    button.disabled = true;
    try {
      if (!(await ready())) { button.disabled = false; return; }
      state = await api('POST', `/projects/${projectId}/curation/similarity`, { json: { device } });
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
    const noise = noiseCount(state.points);
    return el('ul', { class: 'sim-groups' }, noise ? el('li', { class: 'sim-group sim-nogroup' },
      el('span', { class: 'sim-swatch', style: `background:${groupColour(NO_GROUP)}`, 'aria-hidden': 'true' }),
      el('span', null, el('strong', null, t('sim.no_group')), ' ', el('span', { class: 'muted' }, tn('sim.images_n', noise)))) : null,
      ...(state.groups || []).map((g) =>
      el('li', { class: 'sim-group' },
        el('img', { src: `/api/projects/${projectId}/images/${g.typical}/thumb`, alt: t('sim.typical', { n: g.group + 1 }), loading: 'lazy', width: '64', height: '64' }),
        el('span', { class: 'sim-swatch', style: `background:${groupColour(g.group)}`, 'aria-hidden': 'true' }),
        el('span', null, el('strong', null, t('sim.group', { n: g.group + 1 })), ' ', el('span', { class: 'muted' }, tn('sim.images_n', g.size))))));
  }

  function draw() {
    const active = ACTIVE(state);
    const regrouping = ACTIVE(cluster);
    const done = state.points && (state.points.ids || []).length > 0;
    if (onMap) onMap({ ready: !!done, noise: done ? noiseCount(state.points) : 0 });
    const startBtn = el('button', { type: 'button', class: `btn${done ? '' : ' primary'}`, disabled: active || regrouping, onclick: (e) => start(e.currentTarget) },
      done ? t('sim.recompute') : t('sim.compute'));
    const few = reqs && reqs.images < reqs.min_images;
    if (few) startBtn.disabled = true;
    const devices = (reqs && reqs.devices) || ['auto', 'cpu'];
    const deviceSel = el('select', { id: 'sim-device', class: 'tr-select', disabled: active, onchange: (e) => { device = e.target.value; } },
      ...devices.map((d) => el('option', { value: d, selected: d === device }, deviceName(d))));
    const deviceField = el('label', { class: 'sim-device', for: 'sim-device' }, t('tr.device_label'), ' ', deviceSel);
    const actions = el('div', { class: 'cur-actions' }, deviceField, startBtn,
      active ? el('button', { type: 'button', class: 'btn', onclick: cancel }, t('sim.cancel')) : null,
      done && !regrouping ? el('button', { type: 'button', class: 'btn', id: 'sim-regroup', disabled: active, 'aria-describedby': active ? 'sim-regroup-reason' : null, onclick: (e) => regroup(e.currentTarget) }, t('sim.regroup')) : null,
      regrouping ? el('button', { type: 'button', class: 'btn', onclick: cancelRegroup }, t('sim.regroup.cancel')) : null,
      done && !active ? fiftyoneButton(el('button', { type: 'button', class: 'btn', onclick: openApp }, t('sim.open_fo'))) : null);
    const status = statusText(state) || clusterStatusText(cluster);
    live.textContent = status;
    const body = [];
    if (done) {
      const groups = (state.groups || []).length;
      body.push(el('p', null, t('sim.summary', { images: num(state.points.ids.length), groups: num(groups), method: tr(`sim.method.${state.method}`, state.method || '') })));
      if (cluster.status === 'done' && cluster.method) body.push(el('p', null, t(`sim.regroup.done.${cluster.method}`, { groups: num(cluster.groups || 0), min_size: num(cluster.min_size || 0), noise: num(cluster.noise || 0) })));
      if (state.outdated) body.push(el('p', { class: 'sim-note' }, icon('info'), t('sim.outdated')));
      if (state.fiftyone_error) body.push(el('p', { class: 'sim-note' }, icon('info'), t('sim.fo_failed', { error: state.fiftyone_error })));
      body.push(el('div', { class: 'sim-body' }, el('div', { class: 'sim-map', html: mapSvg(state.points) }), groupList()));
      body.push(el('p', { class: 'muted' }, t('sim.fo_hint')));
    }
    fill(node,
      el('h3', { id: 'h-sim' }, t('sim.title')),
      el('p', { class: 'muted' }, t('sim.intro')),
      actions,
      few ? el('p', { class: 'hint' }, t('sim.too_few', { min: num(reqs.min_images), n: num(reqs.images) })) : null,
      done && active ? el('p', { class: 'hint', id: 'sim-regroup-reason' }, t('sim.regroup.wait')) : null,
      status ? el('p', { class: active || regrouping ? 'muted' : 'sim-note' }, status) : null,
      ...body,
      live);
  }

  draw();
  return { node, load };
}
