/* The uniqueness of the images in the Curation step (#667): a score from 0 (almost the same as another image) to 1
 * (nothing like the others), computed on the embeddings of the similarity map. The block computes it (a job of the
 * global queue, like the map) and the controls sort the grid by it and keep only the images below a value. */
import { api } from '../shell/api.js';
import { el, fill, icon } from '../shell/dom.js';
import { nf2, num, parseNumber, t, tr } from '../shell/i18n.js';
import { toast, toastError } from '../shell/toast.js';

const ACTIVE = (s) => s.status === 'queued' || s.status === 'running';
export const SORTS = ['', 'uniqueness_desc', 'uniqueness_asc'];

/** The line that says where the scoring is. */
export function statusText(state) {
  if (state.status === 'queued') return t('uniq.queued');
  if (state.status === 'running') return tr(`uniq.phase.${state.phase || 'scoring'}`, t('uniq.phase.scoring'));
  if (state.status === 'failed') return t('uniq.failed', { error: state.error === 'interrupted' ? t('uniq.interrupted') : state.error || '' });
  if (state.status === 'cancelled') return t('uniq.cancelled');
  return '';
}

/** The value typed in the "below" field: a number from 0 to 1, or NaN. Empty is null (no filter). */
export function parseBelow(text) {
  if (!String(text).trim()) return null;
  const value = parseNumber(text);
  return Number.isFinite(value) && value >= 0 && value <= 1 ? value : NaN;
}

/** The uniqueness block. `page` gives {projectId, onChange}. Returns {node, controls, load, setMap, query}.
 *  `onChange()` is called when the grid must be asked again (a new sort or filter, or scores that appeared or went). */
export function uniquenessBlock(page) {
  const { projectId, onChange } = page;
  const node = el('section', { class: 'uniq', 'aria-labelledby': 'h-uniq' });
  const controls = el('div', { class: 'uniq-controls', hidden: true });
  const live = el('p', { class: 'sr-only', 'aria-live': 'polite' });
  let state = {};
  let timer = null;
  let mapReady = false;   // the similarity map exists: the scores are made from its embeddings
  let sort = '';
  let below = null;
  let belowText = '';
  let belowError = false;
  let hadScores = false;

  const hasScores = () => Object.keys(state.scores || {}).length > 0;
  const query = () => ({ sort, uniqueBelow: below });
  const changed = () => { if (onChange) onChange(); };

  async function load() {
    clearTimeout(timer);
    const was = state.status;
    try {
      state = (await api('GET', `/projects/${projectId}/curation/uniqueness`)) || {};
    } catch (err) {
      if (err.status !== 404) toastError(err);
      return;
    }
    const has = hasScores();
    const stale = !has && (sort || below != null);   // nothing left to sort or filter by
    if (stale) { sort = ''; below = null; belowText = ''; belowError = false; }
    draw();
    if (stale || has !== hadScores || (was !== state.status && state.status === 'done')) {
      hadScores = has;
      changed();
    }
    if (ACTIVE(state) && node.isConnected) timer = setTimeout(load, 2000);
  }

  async function start(button) {
    button.disabled = true;
    try {
      state = await api('POST', `/projects/${projectId}/curation/uniqueness`);
      toast(t('uniq.started'), 'info', 4000);
      draw();
      timer = setTimeout(load, 1000);
    } catch (err) {
      toastError(err);
      button.disabled = false;
    }
  }

  async function cancel() {
    try {
      state = await api('POST', `/projects/${projectId}/curation/uniqueness/cancel`);
      load();
    } catch (err) { toastError(err); }
  }

  function applyBelow(text) {
    belowText = text;
    const value = parseBelow(text);
    belowError = Number.isNaN(value);
    if (belowError) { drawControls(); return; }
    below = value;
    drawControls();
    changed();
  }

  function drawControls() {
    controls.hidden = !hasScores();
    if (!hasScores()) { fill(controls); return; }
    const sortSel = el('select', { id: 'uniq-sort', class: 'tr-select', onchange: (e) => { sort = e.target.value; drawControls(); changed(); } },
      ...SORTS.map((s) => el('option', { value: s, selected: s === sort }, t(`uniq.sort.${s || 'none'}`))));
    const input = el('input', {
      id: 'uniq-below', type: 'text', inputmode: 'decimal', autocomplete: 'off', value: belowText, placeholder: nf2.format(0.3),
      'aria-invalid': belowError ? 'true' : null, 'aria-describedby': 'uniq-below-help',
    });
    const clear = () => { below = null; belowError = false; applyBelow(''); };
    const form = el('form', { class: 'uniq-below', novalidate: true, onsubmit: (e) => { e.preventDefault(); applyBelow(input.value); } },
      el('label', { for: 'uniq-below' }, t('uniq.below_label')),
      input,
      el('button', { type: 'submit', class: 'btn' }, t('uniq.below_apply')),
      below != null ? el('button', { type: 'button', class: 'btn', onclick: clear }, t('uniq.below_clear')) : null);
    fill(controls,
      el('div', { class: 'uniq-row' },
        el('label', { class: 'uniq-sort', for: 'uniq-sort' }, t('uniq.sort_label'), ' ', sortSel),
        form),
      el('p', { class: belowError ? 'tr-form-error' : 'hint', id: 'uniq-below-help', role: belowError ? 'alert' : null },
        belowError ? t('uniq.below_invalid') : t('uniq.below_help')));
  }

  function draw() {
    const active = ACTIVE(state);
    const has = hasScores();
    const blocked = !mapReady && !active;
    const startBtn = el('button', {
      type: 'button', class: `btn${has ? '' : ' primary'}`, disabled: active || !mapReady,
      'aria-describedby': blocked ? 'uniq-reason' : null, onclick: (e) => start(e.currentTarget),
    }, has ? t('uniq.recompute') : t('uniq.compute'));
    const actions = el('div', { class: 'cur-actions' }, startBtn,
      active ? el('button', { type: 'button', class: 'btn', onclick: cancel }, t('uniq.cancel')) : null);
    const status = statusText(state);
    live.textContent = status;
    const body = [];
    if (has) {
      body.push(el('p', null, t('uniq.summary', { n: num(Object.keys(state.scores).length) }), state.missing ? ` ${t('uniq.missing', { n: num(state.missing) })}` : ''));
      if (state.outdated) body.push(el('p', { class: 'sim-note' }, icon('info'), t('uniq.outdated')));
      if (state.fiftyone_error) body.push(el('p', { class: 'sim-note' }, icon('info'), t('uniq.fo_failed', { error: state.fiftyone_error })));
    }
    fill(node,
      el('h3', { id: 'h-uniq' }, t('uniq.title')),
      el('p', { class: 'muted' }, t('uniq.intro')),
      actions,
      blocked ? el('p', { class: 'hint', id: 'uniq-reason' }, t('uniq.needs_map')) : null,
      status ? el('p', { class: active ? 'muted' : 'sim-note' }, status) : null,
      ...body,
      live);
    drawControls();
  }

  draw();
  return {
    node, controls, load, query,
    /** Tells whether the similarity map exists: the scores are made from its embeddings. */
    setMap(ready) { if (ready !== mapReady) { mapReady = ready; draw(); } },
  };
}
