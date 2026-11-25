/* Pre-annotation, a tab of the Annotations step: run a saved or public model on the images and review the
 * result. */
import { deviceName } from './names.js';
import { api } from '../shell/api.js';
import { confirmDialog } from '../shell/dialogs.js';
import { el, fill, icon } from '../shell/dom.js';
import { LOCALE, fmtDateTime, num, t, tn, tr } from '../shell/i18n.js';
import { toast, toastError } from '../shell/toast.js';

/* A saved model suggests labels for new images; a person reviews them in CVAT or FiftyOne.
   Three blocks, in the order of the work: pre-annotate, to review, history. */

/** The pre-annotation tab. `hooks` are the callbacks of the Annotations step: `changed()` after a run ends or
 *  is accepted, `reviewCvat(taskId)` and `reviewFiftyone()`. Returns {node, load, choose}. */
export function preannotateBlock(page, hooks) {
  const { projectId } = page;
  const node = el('div', { class: 'pre' });
  const live = el('p', { class: 'sr-only', 'aria-live': 'polite' });
  const nf3 = new Intl.NumberFormat(LOCALE, { maximumFractionDigits: 3 });
  const form = { model: null, task: null, scope: 'unlabeled', threshold: null, device: 'auto' };   // threshold: by kind until the person sets one
  const DEFAULT_THRESHOLD = { classification: 0.5, detection: 0.3, keypoints: 0.3 };
  let data = null;
  let timer = null;
  let sig = '';
  let wasActive = false;
  let seenRun = null;
  const ACTIVE = (r) => r.status === 'queued' || r.status === 'running';
  const metricName = (m) => tr(`tr.metric.${m}`, String(m || '').replace(/_/g, ' '));

  /* where the model runs (#553): the devices of the training environment, read once; the list keeps auto and CPU
     when the environment cannot be read */
  let devices = null;
  function loadDevices() {
    if (devices) return;
    devices = ['auto', 'cpu'];
    api('GET', '/training/environment').then((env) => { devices = env.devices || devices; if (data) draw(true); }).catch(() => {});
  }

  async function load() {
    clearTimeout(timer);
    loadDevices();
    try {
      data = await api('GET', `/projects/${projectId}/preannotations`);
    } catch (err) {
      toastError(err);
      if (wasActive && node.isConnected) timer = setTimeout(load, 5000);   // try again: a run may still be going
      return;
    }
    if (seenRun == null) seenRun = ((data.runs || [])[0] || {}).number || 0;   // runs before this visit are history
    const active = (data.runs || []).some(ACTIVE);
    if (wasActive && !active) hooks.changed();
    wasActive = active;
    draw();
    if (active && node.isConnected) timer = setTimeout(load, 2000);
  }

  function pickDefaults() {
    const models = (data.models || []).filter((m) => m.supported);
    if (!models.some((m) => m.number === form.model)) { form.model = ((models.find((m) => m.current) || models[0]) || {}).number || null; form.threshold = null; }
    const chosen = models.find((m) => m.number === form.model);
    if (form.threshold == null) form.threshold = (chosen && DEFAULT_THRESHOLD[chosen.task_kind]) || 0.5;
    const tasks = data.tasks || [];
    if (!tasks.some((x) => x.id === form.task)) form.task = ((tasks.slice().reverse().find((x) => x.unlabeled > 0)) || tasks[tasks.length - 1] || {}).id || null;
    const task = tasks.find((x) => x.id === form.task);
    if (task && !task[form.scope] && task[form.scope === 'unlabeled' ? 'unreviewed' : 'unlabeled']) form.scope = form.scope === 'unlabeled' ? 'unreviewed' : 'unlabeled';
  }

  async function start(button) {
    button.disabled = true;
    try {
      await api('POST', `/projects/${projectId}/preannotations`, { json: { model: form.model, task: form.task, scope: form.scope, threshold: form.threshold, device: form.device } });
      live.textContent = t('pre.started');
    } catch (err) { toastError(err); }
    await load();
  }

  async function cancel(run) {
    try { await api('POST', `/projects/${projectId}/preannotations/${run.number}/cancel`); } catch (err) { toastError(err); }
    await load();
  }

  async function acceptTask(task) {
    const ok = await confirmDialog({ title: t('pre.accept_title', { task: task.name }), text: tn('pre.accept_text', task.unreviewed), confirmLabel: t('pre.accept'), danger: false });
    if (!ok) return;
    try {
      const r = await api('POST', `/projects/${projectId}/preannotations/accept`, { json: { task: task.id } });
      toast(tn('pre.accepted', r.accepted), 'success', 3500);
    } catch (err) { toastError(err); }
    hooks.changed();
    await load();
  }

  function formBlock() {
    const models = (data.models || []).filter((m) => m.supported);
    const tasks = data.tasks || [];
    const task = tasks.find((x) => x.id === form.task);
    const model = models.find((m) => m.number === form.model);
    const n = task ? task[form.scope] || 0 : 0;
    const running = (data.runs || []).find(ACTIVE);
    /* a saved model this version no longer knows is listed, disabled, and says why */
    const gone = (data.models || []).filter((m) => m.missing);
    const modelSel = el('select', { id: 'pre-model', class: 'tr-select', 'aria-describedby': gone.length ? 'pre-model-hint' : null, onchange: (e) => { form.model = Number(e.target.value); form.threshold = null; draw(true); } },
      ...models.map((m) => el('option', { value: String(m.number), selected: m.number === form.model },
        [`m${m.number}`, m.label, m.value != null ? `${metricName(m.metric)} ${nf3.format(m.value)}` : null, m.current ? t('md.current') : null].filter(Boolean).join(' · '))),
      ...gone.map((m) => el('option', { value: String(m.number), disabled: true }, `m${m.number} · ${m.label} · ${t('pre.model_missing')}`)));
    const taskSel = el('select', { id: 'pre-task', class: 'tr-select', onchange: (e) => { form.task = Number(e.target.value); draw(true); } },
      ...tasks.map((x) => el('option', { value: String(x.id), selected: x.id === form.task }, `${x.name} · ${tn('pre.task_unlabeled', x.unlabeled)}`)));
    const scopeOpt = (scope, label, count) => {
      const id = `pre-scope-${scope}`;
      const input = el('input', { type: 'radio', name: 'pre-scope', id, class: 'sr-only', checked: form.scope === scope, disabled: !count, onchange: () => { form.scope = scope; draw(true); } });
      return el('label', { for: id, class: `seg-opt${count ? '' : ' is-disabled'}` }, input, label, el('span', { class: 'count' }, num(count)));
    };
    const threshold = el('input', { type: 'number', id: 'pre-threshold', min: '0.05', max: '0.95', step: '0.05', value: String(form.threshold), 'aria-describedby': 'pre-threshold-hint',
      onchange: (e) => { const v = e.target.value === '' ? NaN : Number(e.target.value); form.threshold = Number.isFinite(v) ? Math.min(0.95, Math.max(0.05, v)) : null; draw(true); } });
    const kindHint = model && model.task_kind === 'classification' ? t('pre.threshold_cls') : t('pre.threshold_det');
    const go = el('button', { type: 'button', class: 'btn primary', id: 'pre-start', disabled: !n || !!running, 'aria-describedby': !n && !running ? 'pre-start-hint' : null, onclick: (e) => start(e.currentTarget) }, icon('star', 16), tn('pre.start', n));
    return el('section', { class: 'pre-section', 'aria-labelledby': 'pre-h-form' },
      el('h3', { id: 'pre-h-form' }, t('pre.form_title')),
      el('p', { class: 'muted' }, t('pre.form_intro')),
      el('div', { class: 'pre-form' },
        el('div', { class: 'field' }, el('label', { for: 'pre-model' }, t('pre.model')), modelSel, gone.length ? el('p', { class: 'hint', id: 'pre-model-hint' }, t('pre.model_missing_hint')) : null),
        el('div', { class: 'field' }, el('label', { for: 'pre-task' }, t('pre.task')), taskSel),
        el('fieldset', { class: 'field pre-scope' }, el('legend', null, t('pre.images')),
          el('div', { class: 'seg' }, scopeOpt('unlabeled', t('pre.scope_unlabeled'), task ? task.unlabeled : 0), scopeOpt('unreviewed', t('pre.scope_unreviewed_short'), task ? task.unreviewed : 0))),
        el('div', { class: 'field' }, el('label', { for: 'pre-threshold' }, t('pre.threshold')), threshold, el('p', { class: 'hint', id: 'pre-threshold-hint' }, kindHint)),
        el('div', { class: 'field' }, el('label', { for: 'pre-device' }, t('tr.device_label')),
          el('select', { id: 'pre-device', class: 'tr-select', 'aria-describedby': 'pre-device-hint', onchange: (e) => { form.device = e.target.value; } },
            ...(devices || ['auto', 'cpu']).map((d) => el('option', { value: d, selected: d === form.device }, deviceName(d)))),
          el('p', { class: 'hint', id: 'pre-device-hint' }, t('pre.device_help')))),
      el('div', { class: 'pre-actions' }, go, running ? liveLine(running) : (n ? null : el('p', { class: 'hint', id: 'pre-start-hint' }, t('pre.start_none'))) || lastOutcome()));
  }

  function lastOutcome() {
    const r = (data.runs || [])[0];
    if (!r || r.status !== 'done' || r.number <= seenRun) return null;
    const c = r.counts || {};
    if (!c.images && c.empty) return el('p', { class: 'pre-note warn' }, icon('warning', 16), t('pre.result_none_long', { v: nf3.format(r.threshold || 0) }));
    return el('p', { class: 'pre-note ok' }, icon('check', 16), t('pre.result_done', { images: num(c.images || 0), objects: num(c.objects || 0) }));
  }

  function liveLine(run) {
    const pct = run.total ? Math.round(((run.done || 0) / run.total) * 100) : 0;
    const text = run.status === 'queued' ? t('pre.queued') : run.message && run.message.startsWith('train_env_install:') ? t('md.installing', { pkg: run.message.split(':')[1] }) : t('pre.running', { done: num(run.done || 0), total: num(run.total || 0) });
    return el('div', { class: 'pre-live' },
      el('span', { class: 'tr-dot', 'aria-hidden': 'true' }), el('span', null, text),
      run.status === 'running' ? el('progress', { class: 'tr-progress', max: '100', value: String(pct), 'aria-label': t('common.progress') }) : null,
      el('button', { type: 'button', class: 'btn small danger-outline', onclick: () => cancel(run) }, t('jobs.stop')));
  }

  function reviewBlock() {
    const pending = (data.tasks || []).filter((x) => x.unreviewed > 0);
    return el('section', { class: 'pre-section', 'aria-labelledby': 'pre-h-review' },
      el('h3', { id: 'pre-h-review' }, t('pre.review_title')),
      pending.length
        ? el('ul', { class: 'pre-review' }, ...pending.map((x) => el('li', { class: 'pre-row' },
          el('div', { class: 'pre-row-text' },
            el('strong', null, x.name),
            el('span', { class: 'muted' }, tn('pre.pending', x.unreviewed, { models: (x.models || []).join(', ') }))),
          el('div', { class: 'pre-row-actions' },
            el('button', { type: 'button', class: 'btn small', onclick: () => hooks.reviewCvat(x.id) }, icon('cvat', 14), t('pre.review_cvat')),
            el('button', { type: 'button', class: 'btn small', onclick: () => hooks.reviewFiftyone() }, icon('fiftyone', 14), t('pre.review_fiftyone')),
            el('button', { type: 'button', class: 'btn small', onclick: () => acceptTask(x) }, icon('check', 14), t('pre.accept'))))))
        : el('p', { class: 'muted' }, t('pre.review_none')),
      pending.length ? el('p', { class: 'hint' }, t('pre.review_help')) : null);
  }

  function historyBlock() {
    const runs = data.runs || [];
    if (!runs.length) return null;
    const status = (r) => {
      if (ACTIVE(r)) return t(r.status === 'queued' ? 'pre.queued' : 'pre.running', { done: num(r.done || 0), total: num(r.total || 0) });
      if (r.status === 'done') {
        const c = r.counts || {};
        if (!c.images && c.empty) return t('pre.result_none', { v: nf3.format(r.threshold || 0) });   // the model was never that sure
        return t('pre.result', { images: num(c.images || 0), objects: num(c.objects || 0) });
      }
      if (r.status === 'failed') return `${t('tr.st_failed')}: ${r.error === 'interrupted' ? t('md.interrupted') : r.error}`;
      return t('tr.st_cancelled');
    };
    return el('details', { class: 'pre-section pre-history' },
      el('summary', null, t('pre.history', { n: num(runs.length) })),
      el('div', { class: 'tr-table-wrap' }, el('table', { class: 'tr-table pre-table' },
        el('thead', null, el('tr', null, ...['#', t('pre.model'), t('pre.task'), t('pre.threshold'), t('pre.outcome'), t('pre.when')].map((h) => el('th', { scope: 'col' }, h)))),
        el('tbody', null, ...runs.map((r) => el('tr', null,
          el('th', { scope: 'row' }, String(r.number)),
          el('td', null, `m${r.model} · ${r.model_label || ''}`),
          el('td', null, `${r.task_name || r.task} · ${t(r.scope === 'unlabeled' ? 'pre.scope_unlabeled' : 'pre.scope_unreviewed')}`),
          el('td', null, nf3.format(r.threshold || 0)),
          el('td', { class: r.status === 'failed' ? 'pre-fail' : null }, status(r)),
          el('td', null, fmtDateTime(r.finished || r.created))))))));
  }

  function draw(force = false) {
    if (!data) { fill(node, el('p', { class: 'muted' }, t('common.loading'))); return; }
    const s2 = JSON.stringify([data, form]);
    if (!force && s2 === sig) return;
    const typing = document.activeElement && node.contains(document.activeElement) && document.activeElement.tagName === 'INPUT' && document.activeElement.type === 'number';
    if (!force && typing) {   // a value being typed is not replaced: only the progress line changes
      const running = (data.runs || []).find(ACTIVE);
      const line = node.querySelector('.pre-live');
      if (line && running) line.replaceWith(liveLine(running));
      return;
    }
    sig = s2;
    if (!(data.models || []).length) {
      fill(node, el('div', { class: 'ds-empty' }, icon('star', 28), el('p', { class: 'muted' }, t('pre.no_models')),
        el('a', { class: 'btn', href: '#models' }, t('pre.go_models'))));
      return;
    }
    if (!(data.models || []).some((m) => m.supported)) {
      fill(node, el('p', { class: 'report-line' }, icon('info'), t('pre.none_supported')));
      return;
    }
    pickDefaults();
    const focused = document.activeElement && node.contains(document.activeElement) ? document.activeElement.id : null;
    fill(node, live, formBlock(), reviewBlock(), historyBlock());
    if (focused) { const f = document.getElementById(focused); if (f) f.focus(); }
  }

  /** Opens the form on this model (the step 6 button "pre-annotate with this one"). */
  const choose = (number) => { form.model = number; form.threshold = null; };
  return { node, load, choose };
}
