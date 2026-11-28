/* Pre-annotation, a tab of the Annotations step: run a saved or public model on the images and review the
 * result. */
import { deviceName } from './names.js';
import { api } from '../shell/api.js';
import { confirmDialog } from '../shell/dialogs.js';
import { el, fill, icon } from '../shell/dom.js';
import { fmtDateTime, nf3, nf4, num, t, tn, tr } from '../shell/i18n.js';
import { toast, toastError } from '../shell/toast.js';

/* A saved model suggests labels for new images; a person reviews them in CVAT or FiftyOne.
   Three blocks, in the order of the work: pre-annotate, to review, history. */

/** The text of a model in the model list: its number and "Current" come first, so a narrow list that cuts the end
 *  of a long name still tells the models apart; the result of the model is said under the list (#338).
 *  @param {{number: number, label: string, current?: boolean}} m A saved model.
 *  @returns {string} For example "m1 (Current) · Faster R-CNN MobileNetV3 FPN". */
export function modelOptionText(m) {
  return [m.current ? `m${m.number} (${t('md.current')})` : `m${m.number}`, m.label].filter(Boolean).join(' · ');
}

/** The pre-annotation tab. `hooks` are the callbacks of the Annotations step: `changed()` after a run ends or
 *  is accepted, `reviewCvat(taskId)` and `reviewFiftyone()`. Returns {node, load, choose}. */
export function preannotateBlock(page, hooks) {
  const { projectId } = page;
  const regression = () => !!(page.ctx && page.ctx.project) && page.ctx.project.task === 'regression';   // #619
  const node = el('div', { class: 'pre' });
  const live = el('p', { class: 'sr-only', 'aria-live': 'polite' });
  const form = { model: null, task: null, scope: 'unlabeled', device: 'auto', open: false, more: false, query: '' };
  /* the settings of each model (#86): only what differs from the defaults the server describes. `options` holds
     values by name (a number field keeps a wrong value as typed, to show its error); `classes` null means every
     class of the model. Remembered in this browser, per project and model. */
  const SETTINGS_KEY = 'al-pre-settings';
  const MANY_CLASSES = 12;   // up to this many classes: chips; more: a search and a list
  const LIST_LIMIT = 100;    // classes listed at once in a long list
  let settings = {};
  try { settings = JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}')[projectId] || {}; } catch (err) { settings = {}; }
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
    if (!models.some((m) => m.number === form.model)) form.model = ((models.find((m) => m.current) || models[0]) || {}).number || null;
    const tasks = data.tasks || [];
    if (!tasks.some((x) => x.id === form.task)) form.task = ((tasks.slice().reverse().find((x) => x.unlabeled > 0)) || tasks[tasks.length - 1] || {}).id || null;
    const task = tasks.find((x) => x.id === form.task);
    if (task && !task[form.scope] && task[form.scope === 'unlabeled' ? 'unreviewed' : 'unlabeled']) form.scope = form.scope === 'unlabeled' ? 'unreviewed' : 'unlabeled';
  }

  async function start(button) {
    button.disabled = true;
    try {
      const model = (data.models || []).find((m) => m.number === form.model);
      const s = model ? settingsOf(model) : { options: {}, classes: null };
      const json = { model: form.model, task: form.task, scope: form.scope, device: form.device, options: sentOptions(model) };
      if (s.classes) json.classes = s.classes;
      await api('POST', `/projects/${projectId}/preannotations`, { json });
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
    const modelSel = el('select', { id: 'pre-model', class: 'tr-select', 'aria-describedby': [model && model.value != null ? 'pre-model-detail' : null, gone.length ? 'pre-model-hint' : null].filter(Boolean).join(' ') || null, onchange: (e) => { form.model = Number(e.target.value); form.query = ''; draw(true); } },
      ...models.map((m) => el('option', { value: String(m.number), selected: m.number === form.model },
        modelOptionText(m))),
      ...gone.map((m) => el('option', { value: String(m.number), disabled: true }, `m${m.number} · ${m.label} · ${t('pre.model_missing')}`)));
    const taskSel = el('select', { id: 'pre-task', class: 'tr-select', onchange: (e) => { form.task = Number(e.target.value); draw(true); } },
      ...tasks.map((x) => el('option', { value: String(x.id), selected: x.id === form.task }, `${x.name} · ${tn('pre.task_unlabeled', x.unlabeled)}`)));
    const scopeOpt = (scope, label, count) => {
      const id = `pre-scope-${scope}`;
      const input = el('input', { type: 'radio', name: 'pre-scope', id, class: 'sr-only', checked: form.scope === scope, disabled: !count, onchange: () => { form.scope = scope; draw(true); } });
      return el('label', { for: id, class: `seg-opt${count ? '' : ' is-disabled'}` }, input, label, el('span', { class: 'count' }, num(count)));
    };
    const blocked = model ? settingsProblem(model) : null;
    const go = el('button', { type: 'button', class: 'btn primary', id: 'pre-start', disabled: !n || !!running || !!blocked, 'aria-describedby': !n && !running ? 'pre-start-hint' : blocked ? 'pre-set-problem' : null, onclick: (e) => start(e.currentTarget) }, icon('star', 16), tn('pre.start', n));
    return el('section', { class: 'pre-section', 'aria-labelledby': 'pre-h-form' },
      el('h3', { id: 'pre-h-form' }, t('pre.form_title')),
      el('p', { class: 'muted' }, t('pre.form_intro')),
      el('div', { class: 'pre-form' },
        el('div', { class: 'field' }, el('label', { for: 'pre-model' }, t('pre.model')), modelSel,
          model && model.value != null ? el('p', { class: 'hint', id: 'pre-model-detail' }, t('pre.model_result', { metric: metricName(model.metric), value: nf3.format(model.value) })) : null,
          gone.length ? el('p', { class: 'hint', id: 'pre-model-hint' }, t('pre.model_missing_hint')) : null),
        el('div', { class: 'field' }, el('label', { for: 'pre-task' }, t('pre.task')), taskSel),
        el('fieldset', { class: 'field pre-scope' }, el('legend', null, t('pre.images')),
          el('div', { class: 'seg' }, scopeOpt('unlabeled', t('pre.scope_unlabeled'), task ? task.unlabeled : 0), scopeOpt('unreviewed', t('pre.scope_unreviewed_short'), task ? task.unreviewed : 0))),
        el('div', { class: 'field' }, el('label', { for: 'pre-device' }, t('tr.device_label')),
          el('select', { id: 'pre-device', class: 'tr-select', 'aria-describedby': 'pre-device-hint', onchange: (e) => { form.device = e.target.value; } },
            ...(devices || ['auto', 'cpu']).map((d) => el('option', { value: d, selected: d === form.device }, deviceName(d)))),
          el('p', { class: 'hint', id: 'pre-device-hint' }, t('pre.device_help')))),
      model ? settingsBlock(model) : null,
      el('div', { class: 'pre-actions' }, go, running ? liveLine(running) : (n ? null : el('p', { class: 'hint', id: 'pre-start-hint' }, t('pre.start_none'))) || lastOutcome()));
  }

  /* --- settings of the pre-annotation (#86): classes and the parameters of the model's kind --- */

  const optLabel = (name) => (name === 'confidence' ? t('pre.threshold') : tr(`pre.opt.${name}`, name.replace(/_/g, ' ')));
  const optHint = (model, name) => (name === 'confidence' ? t(model.task_kind === 'classification' ? 'pre.threshold_cls' : 'pre.threshold_det') : tr(`pre.opt_hint.${name}`, ''));
  const fmtValue = (o, v) => (o.type === 'bool' ? t(v ? 'pre.yes' : 'pre.no') : nf4.format(Number(v)));
  function settingsOf(model) {
    if (!settings[model.number]) settings[model.number] = { options: {}, classes: null };
    return settings[model.number];
  }
  function remember() {
    try {
      const all = JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}');
      all[projectId] = settings;
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(all));
    } catch (err) { /* not remembered: the settings still apply to this visit */ }
  }
  const usableClasses = (model) => (model.classes || []).filter((c) => c.in_project).map((c) => c.name);
  /* the classes chosen, in the model's order (null in the settings: all of them) */
  function chosenClasses(model) {
    const usable = usableClasses(model);
    const s = settingsOf(model);
    return s.classes ? usable.filter((c) => s.classes.includes(c)) : usable;
  }
  function setClasses(model, names) {
    const usable = usableClasses(model);
    const picked = usable.filter((c) => names.includes(c));
    settingsOf(model).classes = picked.length === usable.length ? null : picked;
    remember();
    draw(true);
  }
  /* the error of a value, or null; `v` is what the person typed (a string) or a boolean */
  function optionError(o, v) {
    if (o.type === 'bool') return null;
    const x = typeof v === 'number' ? v : (String(v).trim() === '' ? NaN : Number(String(v).replace(',', '.')));
    if (!Number.isFinite(x) || x < o.min || x > o.max) return t('pre.opt_range', { min: nf4.format(o.min), max: nf4.format(o.max) });
    if (o.type === 'int' && !Number.isInteger(x)) return t('pre.opt_whole');
    return null;
  }
  const valueOf = (o, s) => (o.name in s.options ? s.options[o.name] : o.default);
  const changed = (o, s) => o.name in s.options && Number(s.options[o.name]) !== Number(o.default);
  /* the options sent: only those changed, as numbers or booleans (the server fills the defaults) */
  function sentOptions(model) {
    const out = {};
    if (!model) return out;
    const s = settingsOf(model);
    for (const o of model.options || []) {
      if (!(o.name in s.options)) continue;
      const v = s.options[o.name];
      out[o.name] = o.type === 'bool' ? !!v : Number(String(v).replace(',', '.'));
    }
    return out;
  }
  function settingsProblem(model) {
    const s = settingsOf(model);
    if (usableClasses(model).length && !chosenClasses(model).length) return t('pre.classes_none');
    return (model.options || []).some((o) => o.name in s.options && optionError(o, s.options[o.name])) ? t('pre.opt_fix') : null;
  }
  /* one line: what differs from the defaults (the first three, then "+n") */
  function summaryParts(options, schema, classes, total) {
    const parts = [];
    if (classes != null && classes < total) parts.push(t('pre.set_classes', { n: num(classes), total: num(total) }));
    for (const o of schema) {
      if (o.name in options && Number(options[o.name]) !== Number(o.default) && !optionError(o, options[o.name])) parts.push(`${optLabel(o.name)} ${fmtValue(o, options[o.name])}`);
    }
    return parts.length > 4 ? [...parts.slice(0, 3), `+${parts.length - 3}`] : parts;
  }
  function setOption(model, o, value) {
    const s = settingsOf(model);
    if (value === undefined || (o.type === 'bool' ? value === o.default : Number(value) === Number(o.default) && String(value).trim() !== '')) delete s.options[o.name];
    else s.options[o.name] = value;
    remember();
    draw(true);
  }
  function optionField(model, o) {
    const s = settingsOf(model);
    const id = `pre-opt-${o.name}`;
    const v = valueOf(o, s);
    const isChanged = changed(o, s) || (o.name in s.options && optionError(o, s.options[o.name]));
    const error = o.name in s.options ? optionError(o, s.options[o.name]) : null;
    const hintId = `${id}-hint`;
    const reset = isChanged ? el('button', { type: 'button', class: 'pre-reset', 'aria-label': t('pre.opt_reset_one', { name: optLabel(o.name) }), onclick: () => setOption(model, o, undefined) }, t('pre.opt_reset')) : null;
    const hint = el('p', { class: `hint${error ? ' pre-opt-error' : ''}`, id: hintId }, error || `${optHint(model, o.name)} ${t('pre.opt_default', { v: fmtValue(o, o.default) })}`.trim());
    if (o.type === 'bool') {
      return el('div', { class: `field pre-opt${isChanged ? ' is-changed' : ''}` },
        el('div', { class: 'pre-opt-head' }, el('label', { class: 'ds-check', for: id }, el('input', { type: 'checkbox', id, checked: !!v, 'aria-describedby': hintId, onchange: (e) => setOption(model, o, e.target.checked) }), optLabel(o.name)), reset),
        hint);
    }
    return el('div', { class: `field pre-opt${isChanged ? ' is-changed' : ''}` },
      el('div', { class: 'pre-opt-head' }, el('label', { for: id }, optLabel(o.name)), reset),
      /* the browser counts steps from `min`: a min that is not a multiple of the step (0.001 with 0.01) would make
         the default and every value typed "invalid" for it, and the arrows would land on odd values (#663); the
         real limits are checked by optionError and the server */
      el('input', { type: 'number', id, min: String(Math.floor(o.min / o.step + 1e-9) * o.step), max: String(o.max), step: String(o.step), value: String(v), 'aria-describedby': hintId, 'aria-invalid': error ? 'true' : null,
        onchange: (e) => setOption(model, o, e.target.value) }),
      hint);
  }
  function classesBlock(model) {
    const all = model.classes || [];
    const usable = usableClasses(model);
    if (!all.length) return null;
    const chosen = chosenClasses(model);
    const outside = all.filter((c) => !c.in_project).map((c) => c.name);
    const toggle = (name) => setClasses(model, chosen.includes(name) ? chosen.filter((c) => c !== name) : [...chosen, name]);
    let picker;
    if (usable.length <= MANY_CLASSES) {
      picker = el('div', { class: 'pre-chips', role: 'group', 'aria-labelledby': 'pre-classes-h' }, ...usable.map((c) =>
        el('button', { type: 'button', class: `chip-btn${chosen.includes(c) ? ' active' : ''}`, 'aria-pressed': chosen.includes(c) ? 'true' : 'false', onclick: () => toggle(c) }, c)));
    } else {
      const q = form.query.trim().toLowerCase();
      const found = usable.filter((c) => !q || c.toLowerCase().includes(q));
      const ordered = [...found.filter((c) => chosen.includes(c)), ...found.filter((c) => !chosen.includes(c))];
      const shown = ordered.slice(0, LIST_LIMIT);
      picker = el('div', { class: 'pre-class-list' },
        el('input', { type: 'search', id: 'pre-class-search', class: 'pre-class-search', value: form.query, placeholder: t('pre.classes_search'), 'aria-label': t('pre.classes_search'),
          oninput: (e) => { form.query = e.target.value; draw(true); } }),
        el('ul', { class: 'pre-class-items', 'aria-labelledby': 'pre-classes-h' }, ...shown.map((c, i) => {
          const id = `pre-class-${i}`;
          return el('li', null, el('label', { class: 'ds-check', for: id }, el('input', { type: 'checkbox', id, checked: chosen.includes(c), onchange: () => toggle(c) }), c));
        })),
        ordered.length > shown.length ? el('p', { class: 'hint' }, t('pre.classes_partial', { n: num(shown.length), total: num(ordered.length) })) : null,
        !ordered.length ? el('p', { class: 'hint' }, t('pre.classes_no_match')) : null);
    }
    return el('div', { class: 'pre-set-block' },
      el('h4', { id: 'pre-classes-h' }, t('pre.classes_title')),
      el('div', { class: 'pre-set-tools' },
        el('span', { class: 'muted' }, t('pre.classes_count', { n: num(chosen.length), total: num(usable.length) })),
        el('button', { type: 'button', class: 'btn small', onclick: () => setClasses(model, usable) }, t('pre.classes_all')),
        el('button', { type: 'button', class: 'btn small', onclick: () => setClasses(model, []) }, t('pre.classes_none_btn')),
        el('button', { type: 'button', class: 'btn small', onclick: () => setClasses(model, usable.filter((c) => !chosen.includes(c))) }, t('pre.classes_invert'))),
      picker,
      !chosen.length && usable.length ? el('p', { class: 'pre-note warn', id: 'pre-set-problem' }, icon('warning', 16), t('pre.classes_none')) : null,   // the Start button points here
      outside.length ? el('p', { class: 'hint' }, t('pre.classes_outside', { names: outside.slice(0, 10).join(', ') + (outside.length > 10 ? '…' : '') })) : null);
  }
  function settingsBlock(model) {
    const s = settingsOf(model);
    const schema = model.options || [];
    const usable = usableClasses(model);
    if (!schema.length && !usable.length) return null;
    const problem = settingsProblem(model);
    const parts = summaryParts(s.options, schema, chosenClasses(model).length, usable.length);
    const basic = schema.filter((o) => o.level === 'basic');
    const more = schema.filter((o) => o.level !== 'basic');
    const moreChanged = more.some((o) => o.name in s.options);
    const anyChanged = Object.keys(s.options).length || s.classes;
    const open = form.open || !!problem;
    return el('details', { class: 'pre-settings', open, ontoggle: (e) => { form.open = e.currentTarget.open; } },
      el('summary', null,
        el('span', { class: 'pre-settings-title' }, t('pre.settings')),
        el('span', { class: 'muted' }, ` · ${parts.length ? parts.join(' · ') : t('pre.set_default')}`),
        parts.length ? el('span', { class: 'tr-badge pre-changed' }, t('pre.set_changed')) : null),
      el('div', { class: 'pre-settings-body' },
        classesBlock(model),
        schema.length ? el('div', { class: 'pre-set-block' },
          el('h4', null, t('pre.params_title')),
          el('div', { class: 'pre-opts' }, ...basic.map((o) => optionField(model, o))),
          more.length ? el('details', { class: 'pre-more', open: form.more || moreChanged, ontoggle: (e) => { form.more = e.currentTarget.open; } },
            el('summary', null, t('pre.params_more')),
            el('div', { class: 'pre-opts' }, ...more.map((o) => optionField(model, o)))) : null) : null,
        problem && problem !== t('pre.classes_none') ? el('p', { class: 'pre-note warn', id: 'pre-set-problem' }, icon('warning', 16), problem) : null,   // no classes: said once, under them (#663)
        anyChanged ? el('div', { class: 'pre-set-foot' }, el('button', { type: 'button', class: 'btn small', onclick: () => { settings[model.number] = { options: {}, classes: null }; remember(); draw(true); } }, t('pre.set_reset_all'))) : null));
  }
  /* the settings a run used, in one line, and a way to fill the form with them */
  function runSettings(r) {
    const model = (data.models || []).find((m) => m.number === r.model);
    const schema = model ? model.options || [] : [];
    const options = r.options || { confidence: r.threshold };
    const conf = `${t('pre.threshold')} ${nf3.format(options.confidence != null ? options.confidence : r.threshold || 0)}`;
    const rest = summaryParts(options, schema.filter((o) => o.name !== 'confidence'), r.classes ? r.classes.length : null, model ? usableClasses(model).length : (r.classes || []).length);
    return [conf, ...rest].join(' · ');
  }
  function repeatRun(r) {
    const model = (data.models || []).find((m) => m.number === r.model && m.supported);
    if (!model) return;
    const options = {};
    for (const o of model.options || []) if (r.options && o.name in r.options && Number(r.options[o.name]) !== Number(o.default)) options[o.name] = r.options[o.name];
    if (!r.options && r.threshold != null && Number(r.threshold) !== Number((model.options || []).find((o) => o.name === 'confidence')?.default)) options.confidence = r.threshold;
    const usable = usableClasses(model);
    const classes = r.classes ? usable.filter((c) => r.classes.includes(c)) : null;
    settings[model.number] = { options, classes: classes && classes.length < usable.length ? classes : null };
    remember();
    Object.assign(form, { model: model.number, task: r.task, scope: r.scope, open: true });
    draw(true);
    live.textContent = t('pre.repeat_done', { n: r.number });
    const target = document.getElementById('pre-h-form');
    if (target) target.scrollIntoView({ block: 'start', behavior: 'smooth' });
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
        el('thead', null, el('tr', null, ...['#', t('pre.model'), t('pre.task'), t('pre.settings_col'), t('pre.outcome'), t('pre.when'), ''].map((h) => el('th', { scope: 'col' }, h)))),
        el('tbody', null, ...runs.map((r) => el('tr', null,
          el('th', { scope: 'row' }, String(r.number)),
          el('td', null, `m${r.model} · ${r.model_label || ''}`),
          el('td', null, `${r.task_name || r.task} · ${t(r.scope === 'unlabeled' ? 'pre.scope_unlabeled' : 'pre.scope_unreviewed')}`),
          el('td', null, runSettings(r)),
          el('td', { class: r.status === 'failed' ? 'pre-fail' : null }, status(r)),
          el('td', null, fmtDateTime(r.finished || r.created)),
          el('td', null, (data.models || []).some((m) => m.number === r.model && m.supported)
            ? el('button', { type: 'button', class: 'btn small', 'aria-label': t('pre.repeat_aria', { n: r.number }), onclick: () => repeatRun(r) }, t('pre.repeat'))
            : null)))))));
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
    if (regression()) {   // no model fills values yet: one line instead of a form that cannot work
      fill(node, el('p', { class: 'report-line' }, icon('info'), t('pre.regression_later')));
      return;
    }
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
  const choose = (number) => { form.model = number; form.query = ''; };
  return { node, load, choose };
}
