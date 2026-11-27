/* Evaluating a saved model on one part of a dataset version (#611): the "Evaluate" button of a model card, the
 * progress while it runs and, once done, the metrics, the metrics of each class, the confusion matrix and the way
 * to the worst images and to the FiftyOne App (Model Evaluation panel). The job runs in the global queue like the
 * similarity map; the result is kept per model. */
import { api } from '../shell/api.js';
import { fiftyoneButton, fiftyoneError, sayFiftyoneMissing } from '../shell/features.js';
import { el, fill, icon } from '../shell/dom.js';
import { fmtDateTime, nf3, num, t, tn } from '../shell/i18n.js';
import { dropTab, reserveTab } from '../shell/newtab.js';
import { toast, toastError } from '../shell/toast.js';
import { partName } from './names.js';

export const running = (s) => !!s && (s.status === 'queued' || s.status === 'running');

/** The line that says where an evaluation is ('' when it is not running). */
export function statusText(state) {
  if (state.status === 'queued') return t('md.eval_queued');
  if (state.status !== 'running') return '';
  if (state.phase === 'predicting' && state.total) return t('md.eval_predicting', { done: num(state.done || 0), total: num(state.total) });
  if (state.phase === 'fiftyone') return t('md.eval_fiftyone');
  if (state.message && state.message.startsWith('train_env_install:')) return t('md.installing', { pkg: state.message.split(':')[1] });
  return t('md.eval_environment');
}

/** The part an evaluation was made on, in words: "Test of version 3" (a K-fold validation part: "Fold 1 of ..."). */
export function whereText(r) {
  const part = r.fold != null && r.part !== 'test' ? t('ds.fold', { n: r.fold + 1 }) : partName(r.part);
  return t('md.eval_where', { part, version: r.version });
}

/** The main numbers of a result, as the `[label, value]` pairs of the cards. */
export function mainMetrics(result) {
  const m = result.metrics || {};
  if (result.kind === 'classification') {
    return [[t('tr.metric.accuracy'), m.accuracy], [t('tr.metric.macro_f1'), m.macro_f1]];
  }
  return [[t('tr.metric.precision'), m.precision], [t('tr.metric.recall'), m.recall], [t('tr.metric.f1'), m.f1], ['mAP', m.map]];
}

function metricCards(result) {
  return el('dl', { class: 'tr-stats md-eval-stats' }, ...mainMetrics(result).map(([label, value]) =>
    el('div', { class: 'tr-stat' }, el('dt', null, label), el('dd', null, value == null ? '–' : nf3.format(value)))));
}

/** The table with the metrics of each class (the most frequent first). */
export function classTable(rows, kind) {
  if (!rows.length) return null;
  return el('div', { class: 'tr-block md-eval-classes' },
    el('h4', null, t('md.eval_classes')),
    el('div', { class: 'tr-table-wrap md-eval-scroll' }, el('table', { class: 'tr-table' },
      el('caption', { class: 'sr-only' }, t('md.eval_classes')),
      el('thead', null, el('tr', null,
        el('th', { scope: 'col' }, t('md.eval_class')), el('th', { scope: 'col' }, t('tr.metric.precision')),
        el('th', { scope: 'col' }, t('tr.metric.recall')), el('th', { scope: 'col' }, t('tr.metric.f1')), el('th', { scope: 'col' }, kind === 'detection' ? t('tr.support.objects') : t('md.eval_support_images')))),
      el('tbody', null, ...rows.map((r) => el('tr', null,
        el('th', { scope: 'row', title: r.class }, r.class),
        el('td', null, nf3.format(r.precision)), el('td', null, nf3.format(r.recall)), el('td', null, nf3.format(r.f1)), el('td', null, num(r.support))))))));
}

/** The confusion matrix (real class in rows, predicted in columns); `other` and `missing` are the extra last ones. */
export function confusionTable(confusion, kind = 'detection') {
  const names = [...confusion.classes];
  if (confusion.other) names.push(t('md.eval_other_classes'));
  if (confusion.missing) names.push(t('md.eval_none'));
  const matrix = confusion.matrix || [];
  if (!names.length || !matrix.length) return null;
  const rowSum = matrix.map((row) => row.reduce((a, b) => a + b, 0));
  const cell = (i, j) => {
    const v = (matrix[i] || [])[j] || 0;
    const share = rowSum[i] ? v / rowSum[i] : 0;
    return el('td', { class: `tr-cell${i === j ? ' diag' : ''}${v === 0 ? ' zero' : ''}`, style: v ? `--share:${Math.round(share * 55)}%` : null, title: `${names[i]} → ${names[j]}: ${v}` }, v ? num(v) : '·');
  };
  return el('div', { class: 'tr-block md-eval-matrix' },
    el('h4', null, t('tr.confusion')),
    el('p', { class: 'muted tr-note' }, t('tr.confusion_help'), confusion.missing ? ` ${t(kind === 'classification' ? 'md.eval_none_help_class' : 'md.eval_none_help')}` : ''),
    el('div', { class: 'tr-table-wrap' }, el('table', { class: 'tr-table tr-confusion' },
      el('caption', { class: 'sr-only' }, t('tr.confusion')),
      el('thead', null, el('tr', null, el('th', { scope: 'col', class: 'corner' }, t('tr.actual_vs_pred')), ...names.map((n) => el('th', { scope: 'col', title: n }, n)))),
      el('tbody', null, ...names.map((n, i) => el('tr', null, el('th', { scope: 'row', title: n }, n), ...names.map((_, j) => cell(i, j))))))));
}

/** The state of the evaluations of a project's models: loaded once, and again while one runs. `onChange` redraws. */
export function evaluationStore(projectId, onChange) {
  const states = new Map();
  let activeModel = null;
  const url = (n, tail = '') => `/projects/${projectId}/models/${n}/evaluation${tail}`;

  /* every answer says which model is being evaluated now */
  function keep(n, s) {
    states.set(n, s || {});
    activeModel = s && s.active != null ? s.active : null;
  }

  async function loadOne(n) {
    try {
      keep(n, await api('GET', url(n)));
    } catch (err) { if (err.status !== 404) toastError(err); }
  }

  return {
    get: (n) => states.get(n) || {},
    /** The model being evaluated now (any), or null. */
    active: () => activeModel,
    /** What the cards show, to know when to draw them again. */
    signature: () => JSON.stringify([...states]) + String(activeModel),
    /** Whether an evaluation is queued or running (the page keeps polling). */
    busy: () => [...states.values()].some(running),
    /** Reads the states of models not read yet and of the running ones. */
    async sync(numbers) {
      for (const n of [...states.keys()]) if (!numbers.includes(n)) states.delete(n);
      await Promise.all(numbers.filter((n) => !states.has(n) || running(states.get(n))).map(loadOne));
    },
    async start(n) {
      try {
        keep(n, await api('POST', url(n), { json: { device: 'auto' } }));
        toast(t('md.eval_started'), 'info', 4000);
      } catch (err) { toastError(err); await loadOne(n); }
      onChange();
    },
    async cancel(n) {
      try { keep(n, await api('POST', url(n, '/cancel'))); } catch (err) { toastError(err); }
      onChange();
    },
  };
}

/** The "Evaluate" button of a model card, or its progress and Cancel while it runs. */
export function evaluateControl(m, store) {
  const s = store.get(m.number);
  if (running(s)) {
    return el('span', { class: 'md-export busy', role: 'status' }, el('span', { class: 'tr-dot', 'aria-hidden': 'true' }), statusText(s),
      el('button', { type: 'button', class: 'btn small', 'data-act': 'eval-cancel', onclick: () => store.cancel(m.number) }, t('md.eval_cancel')));
  }
  const other = store.active() != null && store.active() !== m.number ? store.active() : null;
  const hasResult = !!s.result;
  return el('button', {
    type: 'button', class: 'btn small', 'data-act': 'evaluate', disabled: other != null, title: other != null ? t('md.eval_other', { n: other }) : t('md.eval_help'),
    onclick: () => store.start(m.number),
  }, icon('check', 14), hasResult ? t('md.eval_again') : t('md.evaluate'));
}

/** What an evaluation left: the failure or cancellation, or the result with its metrics, tables and ways on.
 *  `ways` gives `{worst(ids, model), openApp(result, model)}`. Returns null when there is nothing to show. */
export function evaluationResult(m, state, ways) {
  const r = state.result;
  const notes = [];
  if (state.status === 'failed') notes.push(el('p', { class: 'tr-fail', role: 'alert' }, icon('error', 14), el('span', null, t('md.eval_failed', { error: state.error === 'interrupted' ? t('md.interrupted') : state.error || '' }))));
  if (state.status === 'cancelled') notes.push(el('p', { class: 'muted' }, t('md.eval_cancelled')));
  if (!r) return notes.length ? el('div', { class: 'md-eval' }, ...notes) : null;
  const worst = r.worst || [];
  return el('div', { class: 'md-eval', 'aria-label': t('md.eval_title', { n: m.number }) },
    ...notes,
    el('h4', null, t('md.eval_title', { n: m.number })),
    el('p', { class: 'muted' }, [whereText(r), tn('md.eval_images_n', r.images), fmtDateTime(r.evaluated)].join(' · '),
      (r.unreadable || []).length ? ` · ${tn('md.eval_unreadable_n', r.unreadable.length)}` : ''),
    metricCards(r),
    r.kind === 'detection' && r.metrics ? el('p', { class: 'muted tr-note' }, t('md.eval_counts', { tp: num(r.metrics.tp), fp: num(r.metrics.fp), fn: num(r.metrics.fn) })) : null,
    el('div', { class: 'md-eval-grid' }, classTable(r.classes || [], r.kind), confusionTable(r.confusion || {}, r.kind)),
    el('div', { class: 'tr-actions md-eval-actions' },
      worst.length
        ? el('button', { type: 'button', class: 'btn small', 'data-act': 'eval-worst', onclick: () => ways.worst(worst, m.number) }, t('md.eval_worst', { n: worst.length }))
        : el('span', { class: 'muted' }, t('md.eval_no_worst')),
      fiftyoneButton(el('button', { type: 'button', class: 'btn small', 'data-act': 'eval-open', onclick: () => ways.openApp(r, m.number) }, t('md.eval_open_fo')))),
    el('p', { class: 'hint' }, t('md.eval_fo_hint', { key: `eval_m${m.number}` })));
}

/** Opens the App on the images a model was evaluated on (the saved view the job made). `page` gives {ctx, projectId, jobStarted}. */
export async function openApp(page, result) {
  const { ctx, projectId, jobStarted } = page;
  if (sayFiftyoneMissing()) return;
  const tab = reserveTab(t('cur.fo_waiting'));
  try {
    const job = await api('POST', `/projects/${projectId}/fiftyone`, { json: { view: result.view } });
    if (job && job.id) ctx.openWhenDone.set(job.id, tab); else dropTab(tab);
    toast(t('md.eval_fo_starting'), 'info', 6000);
    if (jobStarted) await jobStarted(job);
  } catch (err) { dropTab(tab); if (!fiftyoneError(err)) toastError(err); }
}
