/* Step 5, Training: the runs of the project (list, progress, charts, detail), the new-run dialog (family,
 * architecture and size of the model, hyperparameters, grid search), comparing the chosen runs and saving a run as a model. */
import { compareBar, compareCheck, compareTable } from './compare.js';
import { staleCount } from './datasets.js';
import { defaultModel, modelPicker } from './model_picker.js';
import { deviceName } from './names.js';
import { api, errorParams, jobFailure } from '../shell/api.js';
import { confirmDialog, once, openModal, returnFocus } from '../shell/dialogs.js';
import { add, el, fill, icon, redraw } from '../shell/dom.js';
import { LOCALE, decimalSign, fmtDateTime, listSeparator, nf3, nfExact, lower, parseNumber, runName, splitNumbers, num, t, tn, tr } from '../shell/i18n.js';
import { toast, toastError } from '../shell/toast.js';
import { emptyState } from '../shell/widgets.js';

/** The first warmup or patience that cannot fit in the epochs, over every value of the lists (a grid): same rule as the server's
 *  `_check_schedule`. `reads` maps a parameter to its list of numbers. Returns {field, key, value, epochs} or null. */
export function scheduleClash(reads) {
  const epochs = reads.epochs || [];
  if (!epochs.length) return null;
  const shortest = Math.min(...epochs);
  const warmup = reads.warmup_epochs || [];
  if (warmup.length && Math.max(...warmup) > shortest) return { field: 'warmup_epochs', key: 'error.warmup_exceeds_epochs', value: Math.max(...warmup), epochs: shortest };
  const patience = reads.patience || [];
  if (patience.length && Math.max(...patience) >= shortest && Math.max(...patience) > 0) return { field: 'patience', key: 'error.patience_exceeds_epochs', value: Math.max(...patience), epochs: shortest };
  return null;
}

export function trainingPanel(page) {
  const { ctx, projectId, renderStepper } = page;
  const node = el('section', { class: 'panel', id: 'panel-training', 'aria-labelledby': 'h-training' });
  const panel = { node, mounted: false };
  const listBox = el('div', { class: 'tr-list' });
  const cmpBarBox = el('div', { class: 'cmp-bar-box' });
  const cmpBox = el('div', { class: 'cmp-box' });
  const announcer = el('p', { class: 'sr-only', 'aria-live': 'polite' });
  const MAX_GRID = 50;
  const ACTIVE = (r) => r.status === 'queued' || r.status === 'running';
  let runs = [];
  let catalog = null;
  let timer = null;
  let loaded = false;
  const cards = new Map();   // run number -> {sig, node}
  const lastText = new Map();   // run number -> last announced status text

  // ---- small helpers
  async function ensureCatalog() {
    if (!catalog) catalog = await api('GET', `/training/catalog?type=${encodeURIComponent(ctx.project.task)}`);
    return catalog;
  }
  const modelOf = (id) => ((catalog && catalog.models) || []).find((m) => m.id === id);
  const metricName = (m) => tr(`tr.metric.${m}`, tr(`tr.col.${m}`, String(m).replace(/_/g, ' ')));
  const startName = (s) => tr(`tr.start_${s}`, s);
  const paramName = (n) => tr(`tr.p.${n}`, String(n).replace(/_/g, ' '));
  const choiceName = (c) => tr(`tr.opt.${c}`, String(c));
  const fmtVal = (v) => (typeof v === 'number' ? nf3.format(v) : String(v));
  const plain = (v) => nfExact.format(Number(v));   // hyperparameters are shown exactly (no rounding of 0.0001), with the decimal sign of the language
  const foldCount = (v) => Number((v.options || {}).kfold) || (Array.isArray(v.folds) ? v.folds.length : 0) || (v.sizes && Array.isArray(v.sizes.folds) ? v.sizes.folds.length : 0);
  const foldText = (f) => (f == null ? '' : f === 'all' ? t('tr.fold_all') : t('ds.fold', { n: Number(f) + 1 }));
  const datasetText = (r) => (r.dataset == null ? '' : [t('tr.dataset_short', { n: r.dataset }), foldText(r.fold)].filter(Boolean).join(' · '));
  /* a run whose configuration file is damaged comes without model, dataset or start, with `config_error` instead */
  const configError = (r) => (r.config_error ? el('p', { class: 'tr-fail', title: r.config_error }, icon('error', 16), el('span', null, t('tr.config_error'))) : null);  // the technical detail (a file path) stays in the tooltip

  function fmtDuration(ms) {
    const s = Math.max(0, Math.round(ms / 1000));
    if (s < 60) return `${s}s`;
    const m = Math.floor(s / 60);
    if (m < 60) return `${m}m ${String(s % 60).padStart(2, '0')}s`;
    return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m`;
  }
  function durationOf(r) {
    if (!r.started) return '';
    const end = r.finished ? Date.parse(r.finished) : Date.now();
    return fmtDuration(end - Date.parse(r.started));
  }
  function deviceOf(r) {
    const d = r.device || {};
    /* hardware names (GPU, Apple GPU, CPU) are the same in every language: they stay as they are */
    if (d.cuda) return d.device_name || 'GPU';
    if (d.mps) return 'Apple GPU';
    return d.torch ? 'CPU' : '';
  }
  const childIndex = (r) => (r.current ? (r.children || []).findIndex((c) => c.id === r.current) : -1);
  function phaseText(r) {
    if (r.status === 'queued') return r.queue_position ? t('tr.st_queued_n', { n: r.queue_position }) : t('tr.st_queued');
    const i = childIndex(r);
    if (r.status === 'running' && i >= 0 && r.trainings) {
      const own = r.epoch && r.epochs ? t('tr.ph_epoch', { e: r.epoch, n: r.epochs }) : t('tr.ph_training');
      return `${t('tr.ph_child', { i: i + 1, n: r.trainings })} · ${own}`;
    }
    if (r.status === 'running') {
      if (r.phase === 'environment') return t('tr.ph_environment');
      if (r.phase === 'preparing') return t('tr.ph_preparing');
      if (r.phase === 'evaluating') return t('tr.ph_evaluating');
      return r.epoch && r.epochs ? t('tr.ph_epoch', { e: r.epoch, n: r.epochs }) : t('tr.ph_training');
    }
    const names = { done: t('tr.st_done'), failed: t('tr.st_failed'), cancelled: t('tr.st_cancelled') };
    return names[r.status] || String(r.status);
  }
  function percentOf(r) {
    if (r.status !== 'running' || r.phase !== 'training' || !r.epochs) return null;
    const inEpoch = r.step && r.steps ? r.step / r.steps : 0;
    const done = r.step ? Math.max(0, (r.epoch || 1) - 1) + inEpoch : r.epoch || 0;
    const i = childIndex(r);
    const share = i >= 0 && r.trainings ? (i + done / r.epochs) / r.trainings : done / r.epochs;
    return Math.max(0, Math.min(100, Math.round(share * 100)));
  }
  function badge(r) {
    const glyph = r.status === 'done' ? icon('check', 14) : r.status === 'failed' ? icon('error', 14) : r.status === 'cancelled' ? icon('close', 14) : el('span', { class: 'tr-dot', 'aria-hidden': 'true' });
    return el('span', { class: `tr-badge ${r.status}` }, glyph, phaseText(r));
  }
  /* keeps the keyboard focus on the same button when a card or the detail is redrawn */
  function keepFocus(container, redraw) {
    const a = document.activeElement;
    const act = a && container.contains(a) && a.dataset ? a.dataset.act : null;
    redraw();
    if (act) {
      const again = container.querySelector(`[data-act="${act}"]`);
      if (again) again.focus();
    }
  }
  function announce(r) {
    const text = phaseText(r);
    if (lastText.get(r.number) === text) return;
    const first = !lastText.has(r.number);
    lastText.set(r.number, text);
    if (!first || ACTIVE(r)) announcer.textContent = `${t('tr.run_n', { n: r.number })}: ${text}`;
  }

  /* a run's failure and warnings in the interface language, by their codes; the server's own words for a run made
     before the codes (#560) */
  const runError = (r) => jobFailure(r);
  const warningText = (w) => (w.code ? tr(`tr.warn.${w.code}`, w.message, errorParams(w.params)) : w.message);
  function warningsBlock(r) {
    const list = (r.warnings || []).map(warningText).filter(Boolean);
    return list.length
      ? el('ul', { class: 'tr-warnings', 'aria-label': t('tr.warnings') }, ...list.map((w) => el('li', null, icon('warning', 14), el('span', null, w))))
      : null;
  }
  /* quality alerts of a finished run (#148): never block saving; the text comes from tr.alert.<code> */
  const alertText = (a) => tr(`tr.alert.${a.code}`, a.code, { ...errorParams(a.params), ...(a.params && a.params.metric ? { metric: metricName(a.params.metric) } : {}) });
  const alertsOf = (r) => (r.status === 'done' ? r.alerts || [] : []);
  function alertsBadge(r) {
    const list = alertsOf(r);
    return list.length
      ? el('p', { class: 'tr-alerts' }, el('span', { class: 'tr-badge alert', title: list.map(alertText).join(' · ') }, icon('warning', 14), tn('tr.alerts_badge', list.length)))
      : null;
  }
  function alertsBlock(r) {
    const list = alertsOf(r);
    return list.length
      ? el('section', { class: 'tr-section', 'aria-labelledby': 'trd-alerts' },
        el('h3', { id: 'trd-alerts', class: 'section-title' }, t('tr.alerts')),
        el('p', { class: 'muted' }, t('tr.alerts_hint')),
        el('ul', { class: 'tr-warnings' }, ...list.map((a) => el('li', null, icon('warning', 14), el('span', null, alertText(a))))))
      : null;
  }
  /* a failed or cancelled single run with a checkpoint goes on from it (same number); grids and all-folds runs later */
  const resumable = (r) => (r.status === 'failed' || r.status === 'cancelled') && !!r.checkpoint_epoch && r.mode !== 'grid' && r.fold !== 'all';
  async function resumeRun(r) {
    try {
      await api('POST', `/projects/${projectId}/runs/${r.number}/resume`);
      toast(t('tr.resumed', { n: r.number, epoch: num(r.checkpoint_epoch) }), 'success', 4000);
    } catch (err) { toastError(err); }
    await load(true);
  }

  // ---- list
  function runCard(r) {
    const m = modelOf(r.model);
    const metric = (m && m.default_metric) || '';
    const pct = percentOf(r);
    const id = `run-${r.number}`;
    const best = r.best && r.best.value != null ? `${metricName(metric)} ${fmtVal(r.best.value)}` : '';
    const meta = r.config_error ? [] : [datasetText(r), startName(r.start), r.mode === 'grid' ? t('tr.mode_grid') : null, r.trainings ? tn('tr.trainings', r.trainings) : null].filter(Boolean);
    const facts = [durationOf(r), deviceOf(r)].filter(Boolean);
    const act = (name, label, fn, cls = '') => el('button', { type: 'button', class: `btn small ${cls}`, 'data-act': name, onclick: fn }, label);
    const noWeights = r.status === 'done' && !r.model_number && (r.has_weights === false || ctx.weightless.has(r.number));
    const saveButton = act('save', t('md.save'), () => saveModel(r), 'primary');
    if (noWeights) {
      saveButton.disabled = true;
      saveButton.setAttribute('aria-describedby', `${id}-noweights`);
    }
    return el('li', { class: `tr-run card st-${r.status}`, id, 'aria-labelledby': `${id}-t` },
      el('div', { class: 'tr-run-head' },
        comparableRun(r) ? compareCheck({ id: `${id}-cmp`, label: runName(r.number), checked: chosen.has(r.number), onChange: (on) => choose([r.number], on) }) : null,
        el('h3', { id: `${id}-t` }, runName(r.number)),
        el('span', { class: 'tr-model' }, m ? m.label : r.model),
        badge(r)),
      el('p', { class: 'muted tr-meta' }, meta.join(' · ')),
      pct != null ? el('progress', { class: 'tr-progress', max: '100', value: String(pct), 'aria-label': t('common.progress') }) : null,
      r.status === 'running' && r.message ? el('p', { class: 'muted tr-msg' }, r.message) : null,
      configError(r),
      r.status === 'failed' && runError(r) ? el('p', { class: 'tr-fail' }, icon('error', 16), el('span', null, runError(r))) : null,
      warningsBlock(r),
      alertsBadge(r),
      noWeights ? el('p', { class: 'muted tr-msg tr-noweights', id: `${id}-noweights` }, icon('info', 14), t('md.no_weights')) : null,
      best || facts.length
        ? el('p', { class: 'tr-facts' }, best ? el('strong', null, best) : null, best && facts.length ? ' · ' : null, el('span', { class: 'muted' }, facts.join(' · ')))
        : null,
      el('div', { class: 'tr-actions' },
        act('detail', t('tr.details'), () => openDetail(r.number)),
        r.status === 'done' && r.model_number
          ? el('a', { class: 'btn small tr-saved', href: '#models', 'data-act': 'model' }, icon('star', 14), t('md.saved_as', { n: r.model_number }))
          : null,
        r.status === 'done' && !r.model_number ? saveButton : null,
        ACTIVE(r) ? act('cancel', t('tr.cancel'), () => cancelRun(r), 'danger-outline') : null,
        resumable(r) ? el('button', { type: 'button', class: 'btn small primary', 'data-act': 'resume', title: t('tr.resume_from', { epoch: num(r.checkpoint_epoch) }), onclick: once(() => resumeRun(r)) }, t('tr.resume')) : null,
        r.config_error ? null : act('again', t('tr.reproduce'), once(() => reproduceRun(r))),
        ACTIVE(r) ? null : el('button', { type: 'button', class: 'icon-btn danger tr-del', 'data-act': 'delete', 'aria-label': t('tr.delete_aria', { n: r.number }), title: t('tr.delete'), onclick: () => removeRun(r) }, icon('trash', 18))));
  }

  function draw() {
    const ul = listBox.querySelector('ul.tr-cards');
    newBtn.hidden = !runs.length;
    if (!runs.length) {
      cards.clear();
      fill(cmpBarBox);
      fill(cmpBox);
      fill(listBox, emptyState('chart', t('tr.empty'),
        el('button', { type: 'button', class: 'btn primary', onclick: once(() => openNew(newBtn)) }, icon('plus', 16), t('tr.new'))));
      return;
    }
    const list = ul || el('ul', { class: 'tr-cards' });
    const focused = document.activeElement;
    const act = focused && list.contains(focused) && focused.dataset ? focused.dataset.act : null;
    const owner = act ? [...list.children].findIndex((li) => li.contains(focused)) : -1;
    const ownerNumber = owner >= 0 ? Number(list.children[owner].id.replace('run-', '')) : null;
    const keep = new Set();
    const nodes = runs.map((r) => {
      keep.add(r.number);
      const sig = JSON.stringify(r) + (catalog ? '1' : '0') + (ctx.weightless.has(r.number) ? 'w' : '') + (chosen.has(r.number) ? 'c' : '');
      const old = cards.get(r.number);
      if (old && old.sig === sig) return old.node;
      const fresh = runCard(r);
      cards.set(r.number, { sig, node: fresh });
      return fresh;
    });
    for (const k of [...cards.keys()]) if (!keep.has(k)) cards.delete(k);
    const same = nodes.length === list.children.length && nodes.every((n, i) => list.children[i] === n);
    if (!same) fill(list, ...nodes);
    if (!ul) fill(listBox, list);
    drawCompare().catch(toastError);
    if (act && ownerNumber != null && cards.has(ownerNumber) && !list.contains(document.activeElement)) {
      const again = cards.get(ownerNumber).node.querySelector(`[data-act="${act}"]`);
      if (again) again.focus();
    }
  }

  // ---------------------------------------------------------------- comparing the chosen runs (#214)
  const chosen = new Set();     // run numbers chosen for the comparison
  const details = new Map();    // run number -> {sig, data}: the results read for the comparison
  const comparableRun = (r) => r.status === 'done' && !r.config_error;
  function choose(numbers, on) {
    numbers.forEach((n) => (on ? chosen.add(n) : chosen.delete(n)));
    draw();
  }
  /* the results of the chosen runs are read from each run's detail, once per state of the run */
  async function chosenDetails(cols) {
    await Promise.all(cols.map(async (r) => {
      const sig = JSON.stringify([r.status, r.best, r.finished]);
      if (details.has(r.number) && details.get(r.number).sig === sig) return;
      const data = await api('GET', `/projects/${projectId}/runs/${r.number}`);
      details.set(r.number, { sig, data });
    }));
  }
  let cmpSig = '';
  async function drawCompare() {
    const pool = runs.filter(comparableRun);
    for (const n of [...chosen]) if (!pool.some((r) => r.number === n)) chosen.delete(n);
    for (const n of [...details.keys()]) if (!pool.some((r) => r.number === n)) details.delete(n);
    const focusedAct = document.activeElement && cmpBarBox.contains(document.activeElement) ? document.activeElement.dataset.act : null;
    fill(cmpBarBox, pool.length >= 2
      ? compareBar({ prefix: 'tr-cmp', chosen: chosen.size, total: pool.length, onAll: (on) => choose(pool.map((r) => r.number), on), onClear: () => choose([...chosen], false) })
      : null);
    if (focusedAct) { const again = cmpBarBox.querySelector(`[data-act="${focusedAct}"]`) || cmpBarBox.querySelector('[data-act]'); if (again) again.focus(); }
    const cols = pool.filter((r) => chosen.has(r.number)).sort((a, b) => a.number - b.number);
    const sig = JSON.stringify(cols);
    if (sig === cmpSig) return;
    cmpSig = sig;
    if (cols.length < 2) { fill(cmpBox); return; }
    try { await chosenDetails(cols); } catch (err) { cmpSig = ''; toastError(err); return; }   // the next redraw tries again
    if (sig !== cmpSig) return;   // the choice changed while reading
    fill(cmpBox, compareTable({
      id: 'tr-compare', title: t('tr.compare'), help: t('tr.compare_help'), cols,
      resultsOf: (r) => (details.get(r.number) || {}).data && details.get(r.number).data.results,
      head: (r) => { const m = modelOf(r.model); return [`#${r.number}`, el('span', { class: 'md-arch' }, m ? m.label : r.model)]; },
    }));
  }

  function schedule() {
    clearTimeout(timer);
    if (runs.some(ACTIVE)) timer = setTimeout(load, 2000);
  }

  async function load(quiet = false) {
    try {
      await ensureCatalog().catch(() => null);
      const d = await api('GET', `/projects/${projectId}/runs`);
      runs = d.runs || [];
      loaded = true;
    } catch (err) {
      if (!quiet && !loaded) toastError(err);
      schedule();
      return;
    }
    ctx.runCount = runs.length;
    ctx.runActive = runs.filter(ACTIVE).length;
    runs.forEach(announce);
    if (node.isConnected) draw();
    renderStepper();
    schedule();
  }

  async function cancelRun(r) {
    const ok = await confirmDialog({ title: t('tr.cancel_title', { n: r.number }), text: t('tr.cancel_text'), confirmLabel: t('tr.cancel') });
    if (!ok) return;
    try {
      await api('POST', `/projects/${projectId}/runs/${r.number}/cancel`);
      toast(t('tr.cancelled', { n: r.number }), 'success', 3000);
    } catch (err) { toastError(err); }
    await load(true);
  }
  async function reproduceRun(r) {
    try {
      const made = await api('POST', `/projects/${projectId}/runs/${r.number}/reproduce`);
      toast(t('tr.created', { n: made.number }), 'success', 4000);
    } catch (err) { toastError(err); }
    await load(true);
  }
  function saveModel(r) {
    const spec = modelOf(r.model);
    const canOnnx = !!spec && ['classification', 'semantic'].includes(spec.task_kind);
    const onnx = el('input', { type: 'checkbox', id: 'md-onnx', disabled: !canOnnx });
    const formats = el('fieldset', { class: 'field md-formats' },
      el('legend', null, t('md.formats')),
      el('p', { class: 'md-always' }, icon('check', 14), el('span', null, t('md.format_pt'))),
      el('label', { class: 'ds-check', for: 'md-onnx' }, onnx, el('span', null, t('md.format_onnx'))),
      el('p', { class: 'hint' }, canOnnx ? t('md.format_onnx_help') : t('md.onnx_later')));
    const note = el('textarea', { id: 'md-note', rows: '2', maxlength: '500', placeholder: t('md.note_placeholder') });
    const save = el('button', { type: 'submit', class: 'btn primary' }, icon('star', 16), t('md.save'));
    const dlg = el('dialog', { class: 'dialog', 'aria-labelledby': 'md-save-t' },
      el('form', { class: 'dialog-body form', method: 'dialog' },
        el('h2', { id: 'md-save-t' }, t('md.save_title', { n: r.number })),
        el('p', { class: 'muted' }, t('md.save_help')),
        formats,
        el('div', { class: 'field' }, el('label', { for: 'md-note' }, t('md.note'), el('span', { class: 'optional' }, ` · ${t('step.optional')}`)), note),
        el('div', { class: 'form-actions' }, save, el('button', { type: 'button', class: 'btn', onclick: () => dlg.close() }, t('common.cancel')))));
    dlg.querySelector('form').addEventListener('submit', async (e) => {
      e.preventDefault();
      save.disabled = true;
      try {
        const m = await api('POST', `/projects/${projectId}/models`, { json: { run: r.number, note: note.value } });
        let exportFailed = false;
        if (onnx.checked) {
          try { await api('POST', `/projects/${projectId}/models/${m.number}/export`, { json: { format: 'onnx' } }); } catch (err) { exportFailed = true; toastError(err); }
        }
        dlg.close();
        toast(t(onnx.checked && !exportFailed ? 'md.saved_onnx' : 'md.saved', { n: m.number }), 'success', 4000);
        await load(true);
        ctx.panels.models.count();
      } catch (err) {
        toastError(err);
        if (err.code !== 'run_no_weights') { save.disabled = false; return; }
        ctx.weightless.add(r.number);  // the weights are gone: the card says so instead of offering the save again
        dlg.close();
        await load(true);
      }
    });
    dlg.addEventListener('close', () => dlg.remove());
    document.body.appendChild(dlg);
    openModal(dlg);
    note.focus();
  }

  async function removeRun(r) {
    const ok = await confirmDialog({ title: t('tr.delete_title', { n: r.number }), text: t('tr.delete_text'), confirmLabel: t('tr.delete') });
    if (!ok) return;
    try {
      await api('DELETE', `/projects/${projectId}/runs/${r.number}`, { json: { confirm: true } });
      toast(t('tr.deleted', { n: r.number }), 'success', 3000);
    } catch (err) { toastError(err); }
    await load(true);
  }

  // ---------------------------------------------------------------- new run dialog
  /* What accepting the AGPL-3.0 licence of Ultralytics means, in plain words, with the link to its text and an explicit
     confirmation (#213); `onAccepted` gets the licence as the server keeps it. */
  function openLicence(licence, onAccepted) {
    const check = el('input', { type: 'checkbox', id: 'lic-confirm' });
    const accept = el('button', { type: 'submit', class: 'btn primary', disabled: true }, t('tr.licence_accept'));
    check.addEventListener('change', () => { accept.disabled = !check.checked; });
    const dlg = el('dialog', { class: 'dialog', 'aria-labelledby': 'lic-title' },
      el('form', { class: 'dialog-body form', method: 'dialog', onsubmit: async (e) => {
        e.preventDefault();
        if (!check.checked) return;
        accept.disabled = true;
        try {
          const saved = await api('POST', `/training/licences/${encodeURIComponent(licence.id)}/accept`, { json: { accept: true } });
          dlg.close();
          toast(t('tr.licence_accepted', { name: licence.name, family: tr(`tr.family.${licence.id}`, licence.id) }), 'success', 6000);
          onAccepted(saved);
        } catch (err) { accept.disabled = false; toastError(err); }
      } },
        el('h2', { id: 'lic-title' }, t('tr.licence_title', { name: licence.name.replace(/-/g, '‑'), package: tr(`tr.family.${licence.id}`, licence.package) })),
        el('p', null, t('tr.licence_text_what')),
        el('p', null, t('tr.licence_text_share')),
        el('p', null, t('tr.licence_text_private')),
        el('p', null, t('tr.licence_text_kept')),
        el('p', null, el('a', { href: licence.url, target: '_blank', rel: 'noopener' }, t('tr.licence_link', { name: licence.name }), ' ', icon('external', 14))),
        el('label', { class: 'ds-check', for: 'lic-confirm' }, check, el('span', null, t('tr.licence_confirm', { name: licence.name }))),
        el('div', { class: 'form-actions' }, accept, el('button', { type: 'button', class: 'btn', onclick: () => dlg.close() }, t('common.cancel')))));
    dlg.addEventListener('close', () => dlg.remove());
    document.body.appendChild(dlg);
    openModal(dlg);
    check.focus();
  }

  async function openNew(trigger, version = null) {
    const opener = document.activeElement;   // the empty state's own button when the header one is hidden
    let cat;
    let versions;
    let augCat;
    let licences;
    try {
      cat = await ensureCatalog();
      licences = (await api('GET', '/training/licences')).licences || [];
      augCat = await api('GET', `/training/augmentations?type=${encodeURIComponent(ctx.project.task)}`);
      versions = ((await api('GET', `/projects/${projectId}/datasets`)).versions || []).filter((v) => !v.incomplete).sort((a, b) => b.version - a.version);
    } catch (err) { toastError(err); return; }
    const models = cat.models || [];
    /* a family whose licence was not accepted (Ultralytics, AGPL-3.0) cannot be chosen until it is (#213) */
    const licenceOf = (f) => licences.find((l) => l.id === f) || null;
    const blocked = (f) => !!licenceOf(f) && !licenceOf(f).accepted;
    const defaultArch = cat.default_arch || {};
    const firstFamily = (models.find((m) => !blocked(m.family)) || {}).family;
    const first = firstFamily ? defaultModel(models, firstFamily, defaultArch) : null;
    if (!versions.length || !first) { toast(t('tr.no_options'), 'error'); return; }

    const st = { dataset: (versions.find((v) => v.version === version) || versions[0]).version, fold: 'all', family: first.family, model: first.id, start: 'transfer', mode: 'single', metric: '', vals: {}, sel: {}, aug: {}, augLevel: null, augBad: new Set(), augRange: {}, env: null };
    const versionOf = () => versions.find((v) => v.version === st.dataset);
    const modelNow = () => models.find((m) => m.id === st.model);
    let active = [];       // parameter specs shown now
    let reads = {};        // name -> list of values
    const errEls = {};

    const sel = (id, ...opts) => el('select', { id, class: 'tr-select' }, ...opts);
    const create = el('button', { type: 'submit', class: 'btn primary' }, t('tr.start_run'));
    const formError = el('p', { class: 'tr-form-error', role: 'alert', hidden: true });
    const blockedWhy = el('p', { class: 'tr-form-error tr-blocked', role: 'status', hidden: true });   // why "Start" is off, next to it (#635)

    // data
    const datasetSel = sel('tr-dataset', ...versions.map((v) => el('option', { value: String(v.version) },
      [`v${v.version}`, fmtDateTime(v.created), foldCount(v) ? t('ds.kfold_short', { k: foldCount(v) }) : null, v.note && String(v.note).trim().toLowerCase() !== `v${v.version}` ? String(v.note).slice(0, 40) : null].filter(Boolean).join(' · '))));
    datasetSel.value = String(st.dataset);
    const foldSel = sel('tr-fold');
    const foldField = el('div', { class: 'field', hidden: true }, el('label', { for: 'tr-fold' }, t('tr.fold')), foldSel);
    function drawFolds() {
      const k = foldCount(versionOf());
      foldField.hidden = !k;
      fill(foldSel, el('option', { value: 'all' }, t('tr.fold_all')), ...Array.from({ length: k }, (_, i) => el('option', { value: String(i) }, t('ds.fold', { n: i + 1 }))));
      st.fold = 'all';
    }
    /* a version freezes its labels: say so when the labels changed afterwards (training is not blocked) */
    const staleNote = el('p', { class: 'ds-stale-note', id: 'tr-dataset-stale', role: 'status', hidden: true });
    function drawStale() {
      const n = staleCount(versionOf());
      staleNote.hidden = !n;
      fill(staleNote, ...(n ? [icon('warning', 14), tn('ds.stale_train', n, { n: num(n), version: st.dataset })] : []));
    }
    datasetSel.setAttribute('aria-describedby', 'tr-dataset-stale');
    datasetSel.addEventListener('change', () => { st.dataset = Number(datasetSel.value); drawFolds(); drawStale(); changed(); });
    foldSel.addEventListener('change', () => { st.fold = foldSel.value; changed(); });

    // family, then model (#537): only the families with a model for this project type are listed (#601); one whose
    // licence waits to be accepted (Ultralytics, AGPL-3.0) stays in the list, disabled, with the reason
    const FAMILIES = [...new Set(['torchvision', 'timm', 'ultralytics', ...models.map((m) => m.family)])]
      .filter((f) => models.some((m) => m.family === f));
    const familyReason = (f) => (blocked(f) ? t('tr.family_licence', { name: licenceOf(f).name }) : null);
    const familySel = sel('tr-family');
    const familyNote = el('div', { class: 'hint tr-family-note', id: 'tr-family-note' });
    /* the list and its hint; drawn again when a licence is accepted. The reasons stay in view, not only inside the
       closed list, and a family that waits for its licence offers to read and accept it */
    function drawFamilies() {
      fill(familySel, ...FAMILIES.map((f) => {
        const reason = familyReason(f);
        return el('option', { value: f, disabled: !!reason }, reason ? `${tr(`tr.family.${f}`, f)} (${reason})` : tr(`tr.family.${f}`, f));
      }));
      familySel.value = st.family;
      const unavailable = FAMILIES.filter(familyReason).map((f) => `${tr(`tr.family.${f}`, f)} (${familyReason(f)})`);
      const waiting = FAMILIES.filter(blocked);
      fill(familyNote,
        el('p', null, [t('tr.family_help'), unavailable.length ? t('tr.family_unavailable', { list: unavailable.join(' · ') }) : null].filter(Boolean).join(' ')),
        ...waiting.map((f) => el('button', { type: 'button', class: 'btn small', 'data-act': `licence-${f}`, onclick: () => openLicence(licenceOf(f), (accepted) => {
          Object.assign(licenceOf(f), accepted);
          st.family = f;
          drawFamilies();
          chooseFamily(f);
          familySel.focus();
        }) }, t('tr.licence_open', { name: licenceOf(f).name }))));
    }
    drawFamilies();
    familySel.setAttribute('aria-describedby', 'tr-family-note');
    const modelNote = el('p', { class: 'hint', id: 'tr-model-note' });
    /* what a change of model did to the hyperparameters, in one line; read out by screen readers */
    const modelChanges = el('p', { class: 'hint tr-model-changes', id: 'tr-model-changes', role: 'status' });
    let shownModel = null;   // the model the form holds values of
    let touched = false;     // the person chose a model: the size this computer suits is no longer proposed
    /* architecture (or generation) and size (or variant) of the family (#632); the task is the project's */
    const picker = modelPicker({ models, family: st.family, model: st.model, onChange: (m, sizeOnly) => { st.model = m.id; drawModel(sizeOnly); } });
    picker.archField.addEventListener('change', () => { touched = true; });
    picker.variantField.addEventListener('change', () => { touched = true; });
    /* another family: its default model, with the values that still apply carried over (#312) */
    function chooseFamily(f) {
      st.family = f;
      const m = defaultModel(models, f, defaultArch, st.env && st.env.default_size);
      st.model = m.id;
      picker.setFamily(f, m);
      drawModel();
    }

    // start
    const startHelp = el('p', { class: 'hint', 'aria-live': 'polite' });
    const startInputs = {};
    const startLabels = {};
    const startBox = el('div', { class: 'seg tr-seg', role: 'radiogroup', 'aria-label': t('tr.start') },
      ...['transfer', 'finetune', 'scratch'].map((s) => {
        const input = el('input', { type: 'radio', name: 'tr-start', id: `tr-start-${s}`, value: s, class: 'sr-only', onchange: () => { st.start = s; drawStart(); drawParams(); } });
        startInputs[s] = input;
        startLabels[s] = el('label', { for: input.id, class: 'seg-opt' }, input, startName(s));
        return startLabels[s];
      }));
    function drawStart() {
      const m = modelNow();
      if (!m.supports_pretrained) st.start = 'scratch';
      for (const [s, input] of Object.entries(startInputs)) {
        input.checked = st.start === s;
        startLabels[s].hidden = s !== 'scratch' && !m.supports_pretrained;   // not offered, rather than greyed out
      }
      startHelp.textContent = m.supports_pretrained ? t(`tr.start_${st.start}_help`) : t('tr.start_scratch_only');
    }

    // mode
    const modeBox = el('div', { class: 'seg tr-seg', role: 'radiogroup', 'aria-label': t('tr.mode') },
      ...['single', 'grid'].map((m) => {
        const input = el('input', { type: 'radio', name: 'tr-mode', id: `tr-mode-${m}`, value: m, class: 'sr-only', checked: st.mode === m,
          onchange: () => { setMode(m); } });
        return el('label', { for: input.id, class: 'seg-opt' }, input, t(`tr.mode_${m}`));
      }));
    const modeHelp = el('p', { class: 'hint' });
    const summary = el('p', { class: 'tr-combos', role: 'status' });

    // selection metric
    const metricSel = sel('tr-metric');

    // device (#320): not part of the configuration; the list comes from the training environment
    const deviceSel = sel('tr-device', ...['auto', 'cpu'].map((d) => el('option', { value: d }, deviceName(d))));
    api('GET', '/training/environment').then((env) => {
      st.env = env;
      fill(deviceSel, ...(env.devices || ['auto', 'cpu']).map((d) => el('option', { value: d }, deviceName(d))));
      if (shownModel) drawModelNote(modelNow());
      if (env.default_size) picker.setSuggested(env.default_size, !touched);
    }).catch(() => {});

    // parameters
    const paramsBox = el('div', { class: 'tr-params' });
    /* the settings stay closed: the line on the bar says what differs from the defaults */
    const advLine = el('span', { class: 'muted tr-adv-line' });
    const advBox = el('details', { class: 'tr-adv' }, el('summary', null, el('strong', null, t('tr.params')), advLine));
    const moreBox = el('details', { class: 'tr-more' }, el('summary', null, t('tr.more_params')));
    const moreInner = el('div', { class: 'tr-params' });
    add(moreBox, moreInner);

    function readParam(spec) {
      const grid = st.mode === 'grid';
      const name = spec.name;
      if (spec.kind === 'bool') return { list: [!!st.vals[name]] };
      if (spec.kind === 'choice') {
        const list = grid ? st.sel[name] : st.sel[name].slice(0, 1);
        return list.length ? { list } : { error: t('tr.err_required') };
      }
      const text = String(st.vals[name] == null ? '' : st.vals[name]);
      const tokens = grid ? splitNumbers(text, spec.kind === 'int') : [text.trim()].filter(Boolean);
      if (!tokens.length) return { error: t('tr.err_required') };
      const list = [];
      for (const tok of tokens) {
        const n = parseNumber(tok);
        if (!Number.isFinite(n)) return { error: t('tr.err_number', { v: tok }) };
        if (spec.kind === 'int' && !Number.isInteger(n)) return { error: t('tr.err_int', { v: tok }) };
        if ((spec.min != null && n < spec.min) || (spec.max != null && n > spec.max)) return { error: t('tr.err_range', { min: plain(spec.min), max: plain(spec.max) }) };
        list.push(n);
      }
      if (new Set(list).size !== list.length) return { error: t('tr.err_dup') };
      return { list };
    }

    /* a stored value as the field shows it: a text field of a grid writes a plain number with the language's decimal sign */
    function fieldText(value, grid) {
      const text = String(value == null ? '' : value);
      return grid && /^-?\d+\.\d+$/.test(text) ? text.replace('.', decimalSign()) : text;
    }

    function paramField(spec) {
      const grid = st.mode === 'grid';
      const name = spec.name;
      const id = `tr-p-${name}`;
      const err = el('p', { class: 'tr-err', id: `${id}-err` });
      errEls[name] = err;
      const hint = spec.help ? el('p', { class: 'hint', id: `${id}-hint` }, tr(`tr.ph.${name}`, spec.help)) : null;
      const describedby = `${hint ? `${id}-hint ` : ''}${id}-err`;
      let control;
      if (spec.kind === 'bool') {
        control = el('label', { for: id, class: 'ds-check tr-switch' },
          el('input', { type: 'checkbox', id, role: 'switch', checked: !!st.vals[name], 'aria-describedby': describedby, onchange: (e) => { st.vals[name] = e.target.checked; changed(); } }),
          el('span', null, paramName(name)));
        return el('div', { class: 'field tr-param' }, control, hint, err);
      }
      if (spec.kind === 'choice') {
        if (grid) {
          control = el('fieldset', { class: 'field tr-param' }, el('legend', null, paramName(name)),
            el('div', { class: 'tr-checks' }, ...spec.choices.map((c, i) => el('label', { class: 'ds-check', for: `${id}-${i}` },
              el('input', { type: 'checkbox', id: `${id}-${i}`, value: c, checked: st.sel[name].includes(c),
                onchange: (e) => { st.sel[name] = spec.choices.filter((x) => (x === c ? e.target.checked : st.sel[name].includes(x))); changed(); } }),
              el('span', null, choiceName(c))))), hint, err);
          return control;
        }
        const select = el('select', { id, class: 'tr-select', 'aria-describedby': describedby, onchange: (e) => { st.sel[name] = [e.target.value]; changed(); } },
          ...spec.choices.map((c) => el('option', { value: c, selected: st.sel[name][0] === c }, choiceName(c))));
        return el('div', { class: 'field tr-param' }, el('label', { for: id }, paramName(name)), select, hint, err);
      }
      const input = el('input', {
        id, type: grid ? 'text' : 'number', inputmode: spec.kind === 'int' ? 'numeric' : 'decimal', autocomplete: 'off', value: fieldText(st.vals[name], grid),
        min: !grid && spec.min != null ? String(spec.min) : null, max: !grid && spec.max != null ? String(spec.max) : null, step: !grid ? (spec.kind === 'int' ? '1' : 'any') : null,
        placeholder: grid ? `${plain(spec.default)}${listSeparator()}…` : null, 'aria-describedby': describedby,
      });
      input.addEventListener('input', () => { st.vals[name] = input.value; changed(); });
      return el('div', { class: 'field tr-param' }, el('label', { for: id }, paramName(name)), input, hint, err);
    }

    /* Augmentation techniques of the model's kind of task. A preset level fills the list; ticking a technique
       or changing a setting makes the level "custom", whose list goes in the request. */
    const augBox = el('fieldset', { class: 'field tr-aug', id: 'tr-aug' });
    const augKind = () => (augCat.kinds || {})[modelNow().task_kind] || { techniques: [], presets: {} };
    const augLevels = () => st.sel.augmentation || [];
    function loadPreset(level) {
      st.aug = {};
      st.augBad.clear();
      for (const entry of augKind().presets[level] || []) st.aug[entry.name] = { ...entry.params };
    }
    function augEntries() {
      return augKind().techniques.filter((tq) => st.aug[tq.name]).map((tq) => ({ name: tq.name, params: { ...st.aug[tq.name] } }));
    }
    function dropStaleErrors() {
      for (const key of [...st.augBad]) if (!st.aug[key.split('.')[0]]) st.augBad.delete(key);
    }
    function toCustom() {
      st.sel.augmentation = ['custom'];
      st.augLevel = 'custom';
      const select = document.getElementById('tr-p-augmentation');
      if (select && select.tagName === 'SELECT') select.value = 'custom';
    }
    function toggleAug(tq, on) {
      if (on) {
        st.aug[tq.name] = Object.fromEntries(tq.params.map((p) => [p.name, p.default]));
        if (tq.needs_flip_pairs) st.aug[tq.name].flip_pairs = [];
        for (const other of tq.conflicts) delete st.aug[other];
      } else {
        delete st.aug[tq.name];
      }
      dropStaleErrors();
      toCustom();
      drawAug();
      changed();
      const box = document.getElementById(`tr-aug-${tq.name}`);
      if (box) box.focus();   // the list was drawn again: keep the keyboard where it was
    }
    /* a number setting of a chosen technique; an invalid value blocks "Start" until fixed */
    function augParam(tq, spec) {
      const id = `tr-aug-${tq.name}-${spec.name}`;
      const key = `${tq.name}.${spec.name}`;
      st.augRange[key] = `${num(spec.min)}–${num(spec.max)}`;
      const input = el('input', { id, type: 'number', inputmode: 'decimal', autocomplete: 'off', class: 'tr-aug-num', value: String(st.aug[tq.name][spec.name]),
        min: String(spec.min), max: String(spec.max), step: spec.kind === 'int' ? '1' : 'any', title: tr(`tr.augph.${spec.name}`, spec.help), 'aria-invalid': st.augBad.has(key) ? 'true' : null });
      input.addEventListener('input', () => {
        const n = Number(input.value);
        const ok = input.value.trim() !== '' && Number.isFinite(n) && n >= spec.min && n <= spec.max && (spec.kind !== 'int' || Number.isInteger(n));
        if (ok) { st.aug[tq.name][spec.name] = n; st.augBad.delete(key); input.removeAttribute('aria-invalid'); } else { st.augBad.add(key); input.setAttribute('aria-invalid', 'true'); }
        toCustom();
        changed();
      });
      return el('label', { for: id, class: 'tr-aug-set-item' }, el('span', null, tr(`tr.augp.${spec.name}`, spec.name)), input);
    }
    /* keypoints: the left/right pairs a horizontal flip swaps, written "0-1, 2-3" (points numbered from 0) */
    function augPairs(tq) {
      const id = `tr-aug-${tq.name}-pairs`;
      const key = `${tq.name}.flip_pairs`;
      const input = el('input', { id, type: 'text', autocomplete: 'off', class: 'tr-aug-pairs', placeholder: '0-1, 2-3', 'aria-describedby': `${id}-hint`,
        value: (st.aug[tq.name].flip_pairs || []).map((pair) => pair.join('-')).join(', '), 'aria-invalid': st.augBad.has(key) ? 'true' : null });
      input.addEventListener('input', () => {
        const pairs = input.value.split(',').map((s) => s.trim()).filter(Boolean).map((s) => s.split('-').map((x) => x.trim()));
        const flat = pairs.flat();
        const ok = pairs.every((pair) => pair.length === 2 && pair.every((x) => /^\d+$/.test(x)) && pair[0] !== pair[1]) && new Set(flat).size === flat.length;
        if (ok) { st.aug[tq.name].flip_pairs = pairs.map((pair) => pair.map(Number)); st.augBad.delete(key); input.removeAttribute('aria-invalid'); } else { st.augBad.add(key); input.setAttribute('aria-invalid', 'true'); }
        toCustom();
        changed();
      });
      return el('div', { class: 'tr-aug-set-item wide' }, el('label', { for: id }, t('tr.aug_pairs')), input, el('p', { class: 'hint', id: `${id}-hint` }, t('tr.aug_pairs_help')));
    }
    function augItem(tq) {
      const on = !!st.aug[tq.name];
      const id = `tr-aug-${tq.name}`;
      const settings = on ? tq.params.map((spec) => augParam(tq, spec)) : [];
      if (on && tq.needs_flip_pairs) settings.push(augPairs(tq));
      return el('div', { class: `tr-aug-item${on ? ' on' : ''}` },
        el('label', { for: id, class: 'ds-check' },
          el('input', { type: 'checkbox', id, checked: on, 'aria-describedby': `${id}-hint`, onchange: (e) => toggleAug(tq, e.target.checked) }),
          el('span', null, tr(`tr.aug.${tq.name}`, tq.name))),
        el('p', { class: 'hint', id: `${id}-hint` }, tr(`tr.augh.${tq.name}`, tq.help)),
        settings.length ? el('div', { class: 'tr-aug-set' }, ...settings) : null);
    }
    function drawAug() {
      const kind = augKind();
      augBox.hidden = !kind.techniques.length || augLevels().length !== 1;
      if (augBox.hidden) { fill(augBox); return; }
      fill(augBox, el('legend', null, t('tr.aug_title')),
        el('p', { class: 'hint' }, augEntries().length ? t('tr.aug_help') : t('tr.aug_none')),
        el('div', { class: 'tr-aug-list' }, ...kind.techniques.map(augItem)));
    }
    /* a preset level (re)fills the list for the model now shown; a custom list keeps what this model offers */
    function syncAug() {
      const level = augLevels().length === 1 ? augLevels()[0] : null;
      if (level && level !== 'custom') loadPreset(level);
      if (level === 'custom') {
        const offered = new Set(augKind().techniques.map((tq) => tq.name));
        for (const name of Object.keys(st.aug)) if (!offered.has(name)) delete st.aug[name];
        dropStaleErrors();
      }
      st.augLevel = level;
      drawAug();
    }

    function setDefault(spec) {
      if (spec.kind === 'choice') st.sel[spec.name] = [spec.default];
      else st.vals[spec.name] = spec.kind === 'bool' ? !!spec.default : String(spec.default);
    }
    /* a valid value that is not the default of `spec`: the person chose it */
    function edited(spec) {
      const r = readParam(spec);
      return !r.error && r.list.join() !== String([spec.default]);
    }
    /* Values for the model now shown, coming from `prev` (null on opening): a value the person chose is
       kept when the new model has the parameter and accepts it; the rest takes the new model's default.
       Returns the lines that say what changed. */
    function carryValues(prev) {
      const m = modelNow();
      const notes = [];
      notes.resets = 0;   // values the person chose that the new model does not take
      const kept = [];
      for (const spec of m.params) {
        if (spec.name === 'start') continue;
        const old = prev && prev.params.find((p) => p.name === spec.name);
        const name = paramName(spec.name);
        if (!old) {
          setDefault(spec);
          if (prev) notes.push(t('tr.chg_new', { p: name }));
          continue;
        }
        if (edited(old)) {
          const before = readParam(old).list;
          if (spec.kind === 'choice') st.sel[spec.name] = st.sel[spec.name].filter((c) => spec.choices.includes(c));
          if (!readParam(spec).error) { kept.push(name); continue; }
          setDefault(spec);
          notes.resets += 1;
          notes.push(t('tr.chg_reset', { p: name, v: valueText(spec, spec.default), old: before.map((v) => valueText(spec, v)).join(', ') }));
          continue;
        }
        setDefault(spec);
        if (String(old.default) !== String(spec.default)) notes.push(t('tr.chg_default', { p: name, v: valueText(spec, spec.default) }));
      }
      if (prev) {
        for (const old of prev.params) {
          if (old.name !== 'start' && !m.params.some((p) => p.name === old.name)) notes.push(t('tr.chg_gone', { p: paramName(old.name) }));
        }
      }
      if (kept.length) notes.push(t('tr.chg_kept', { list: kept.join(', ') }));
      return notes;
    }
    function drawParams() {
      const m = modelNow();
      active = m.params.filter((p) => p.name !== 'start' && (p.name !== 'backbone_lr_factor' || st.start === 'finetune'));
      for (const k of Object.keys(errEls)) delete errEls[k];
      const common = active.filter((p) => p.group !== 'model');
      const own = active.filter((p) => p.group === 'model');
      fill(paramsBox, ...common.flatMap((p) => (p.name === 'augmentation' ? [paramField(p), augBox] : [paramField(p)])));
      fill(moreInner, ...own.map(paramField));
      moreBox.hidden = !own.length;
      syncAug();
      changed();
    }
    function setMode(mode) {
      st.mode = mode;
      if (mode === 'single') {
        for (const spec of active) {
          if (spec.kind === 'int' || spec.kind === 'float') {
            const first = splitNumbers(st.vals[spec.name], spec.kind === 'int')[0] || '';
            st.vals[spec.name] = Number.isFinite(parseNumber(first)) ? String(parseNumber(first)) : first;
          }
          if (spec.kind === 'choice') st.sel[spec.name] = st.sel[spec.name].slice(0, 1);
        }
      }
      modeHelp.textContent = t(mode === 'grid' ? 'tr.mode_grid_help' : 'tr.mode_single_help');
      if (mode === 'grid') advBox.open = true;  // a grid search is about the values: show them
      drawParams();
    }
    /* the packages a model needs, said as what this computer has or lacks once the environment answered (#635) */
    function drawModelNote(m) {
      const need = m.packages || [];
      const have = st.env && st.env.exists ? Object.keys(st.env.packages || {}).map((n) => n.toLowerCase().replace(/[-_.]+/g, '-')) : null;
      const missing = have ? need.filter((p) => !have.includes(p.toLowerCase().replace(/[-_.]+/g, '-'))) : need;
      const pkgs = !need.length ? '' : have && !missing.length ? t('tr.packages_ready', { list: need.join(', ') })
        : t('tr.packages_note', { list: missing.join(', ') });
      modelNote.textContent = [tr(`tr.note.${m.family}`, ''), pkgs].filter(Boolean).join(' ');
    }
    /* `sizeOnly`: another size of the same architecture keeps every value the person chose and says nothing,
       unless a value had to go back to its default (#312, #632) */
    function drawModel(sizeOnly = false) {
      const m = modelNow();
      /* the catalogue's own note is English: the screen says it by family, in the interface language */
      drawModelNote(m);
      const notes = carryValues(shownModel);
      if (!m.metrics.includes(st.metric)) {   // a metric the person chose stays while the new model has it
        if (shownModel && st.metric) notes.push(t('tr.chg_default', { p: t('tr.select_metric'), v: metricName(m.default_metric) }));
        st.metric = m.default_metric;
      }
      fill(metricSel, ...m.metrics.map((x) => el('option', { value: x, selected: x === st.metric }, metricName(x))));
      const start = st.start;
      drawStart();
      if (shownModel && st.start !== start) notes.push(t('tr.chg_default', { p: t('tr.start'), v: startName(st.start) }));
      if (shownModel && sizeOnly && !notes.resets) modelChanges.textContent = '';
      else if (shownModel) {
        const shown = notes.length > 4 ? [...notes.slice(0, 4), t('tr.params_more', { n: notes.length - 4 })] : notes;
        modelChanges.textContent = notes.length ? `${t('tr.chg_title', { model: m.label })} ${shown.join(' · ')}` : t('tr.chg_none', { model: m.label });
      }
      shownModel = m;
      setMode(st.mode);
    }
    familySel.addEventListener('change', () => { touched = true; chooseFamily(familySel.value); });
    metricSel.addEventListener('change', () => { st.metric = metricSel.value; });

    function changed() {
      let combos = 1;
      let bad = 0;
      const wrong = [];   // names of the settings to fix, said next to the disabled button
      reads = {};
      for (const spec of active) {
        const r = readParam(spec);
        const box = errEls[spec.name];
        const input = document.getElementById(`tr-p-${spec.name}`);
        if (box) box.textContent = r.error || '';
        if (input) { if (r.error) input.setAttribute('aria-invalid', 'true'); else input.removeAttribute('aria-invalid'); }
        if (r.error) { bad += 1; wrong.push(paramName(spec.name)); continue; }
        reads[spec.name] = r.list;
        combos *= r.list.length;
      }
      const clash = scheduleClash(reads);   // a warmup or a patience longer than the training (the server refuses it too)
      if (clash) {
        const box = errEls[clash.field];
        if (box) box.textContent = t(clash.key, { [clash.field === 'patience' ? 'patience' : 'warmup']: num(clash.value), epochs: num(clash.epochs) });
        const input = document.getElementById(`tr-p-${clash.field}`);
        if (input) input.setAttribute('aria-invalid', 'true');
        bad += 1;
      }
      const levels = reads.augmentation || [];
      if (levels.length > 1 && levels.includes('custom')) {   // a custom list cannot be one value of a grid
        if (errEls.augmentation) errEls.augmentation.textContent = t('tr.aug_custom_alone');
        bad += 1;
        wrong.push(paramName('augmentation'));
      }
      if ((levels.length === 1 ? levels[0] : null) !== st.augLevel) syncAug();
      if (levels.length === 1 && levels[0] === 'custom') {
        bad += st.augBad.size;
        st.augBad.forEach((key) => { const [tech, param] = key.split('.'); wrong.push(`${tr(`tr.aug.${tech}`, tech)}: ${param === 'flip_pairs' ? t('tr.aug_pairs') : `${tr(`tr.augp.${param}`, param)} (${st.augRange[key] || ''})`}`); });
      }
      const grid = st.mode === 'grid';
      const folds = st.fold === 'all' ? Math.max(1, foldCount(versionOf())) : 1;   // a hold-out version has 0 folds: one run per combination
      const total = combos * folds;
      summary.hidden = !grid && folds < 2;
      summary.classList.remove('bad', 'ok');
      let blocked = bad > 0;
      if (grid && !bad) {
        if (combos < 2) { summary.textContent = t('tr.grid_need'); summary.classList.add('bad'); blocked = true; }
        else if (total > MAX_GRID) { summary.textContent = t('tr.trainings_max', { total, n: MAX_GRID }); summary.classList.add('bad'); blocked = true; }
        else { summary.textContent = folds > 1 ? t('tr.combos_folds', { c: tn('tr.combos', combos), k: folds, total }) : tn('tr.combos', combos); summary.classList.add('ok'); }
      } else if (grid) summary.textContent = '';
      else if (folds > 1) { summary.textContent = t('tr.folds_runs', { k: folds }); summary.classList.add('ok'); }
      if (!grid) modeHelp.textContent = folds > 1 ? t('tr.mode_single_folds_help') : t('tr.mode_single_help');   // one run per fold, not one run
      create.disabled = blocked;
      blockedWhy.hidden = !wrong.length;
      blockedWhy.textContent = wrong.length ? t('tr.blocked_fields', { fields: wrong.join(', ') }) : '';
      formError.hidden = true;
      if (bad) advBox.open = true;  // the field with the problem must be in view
      advLine.textContent = bad ? `${changedLine(true)} · ${tn('tr.params_invalid', bad, { n: bad })}` : changedLine();
    }

    function valueText(spec, v) {
      if (spec.kind === 'choice') return choiceName(v);
      if (typeof v === 'boolean') return v ? '✓' : '✗';
      return plain(v);
    }
    /* "epochs 20 · batch size 8": only what is not the default; the grid lists show as they are */
    function changedLine(withErrors = false) {
      const diffs = active.filter((spec) => reads[spec.name] && reads[spec.name].join() !== String([spec.default]))
        .map((spec) => `${paramName(spec.name)} ${reads[spec.name].map((v) => valueText(spec, v)).join(', ')}`);
      if (!diffs.length) return withErrors ? t('tr.params_changed') : t('tr.params_default');
      return diffs.length > 3 ? `${diffs.slice(0, 3).join(' · ')} · ${t('tr.params_more', { n: diffs.length - 3 })}` : diffs.join(' · ');
    }

    function requestBody() {
      const params = {};
      const grid = {};
      for (const spec of active) {
        const list = reads[spec.name];
        if (st.mode === 'grid' && list.length > 1) grid[spec.name] = list; else params[spec.name] = list[0];
      }
      const folds = foldCount(versionOf());
      const body = { dataset: st.dataset, fold: folds ? (st.fold === 'all' ? 'all' : Number(st.fold)) : null, model: st.model, start: st.start, mode: st.mode, metric: st.metric, params, grid, device: deviceSel.value };
      if (params.augmentation === 'custom') body.augmentations = augEntries();
      return body;
    }

    const form = el('form', { class: 'dialog-body form tr-form', method: 'dialog', novalidate: true },
      el('h2', null, t('tr.new_title')),
      el('div', { class: 'tr-form-grid' },
        el('div', { class: 'tr-col' },
          el('div', { class: 'field' }, el('label', { for: 'tr-dataset' }, t('tr.dataset')), datasetSel, staleNote),
          foldField,
          el('div', { class: 'field' }, el('label', { for: 'tr-family' }, t('tr.family')), familySel, familyNote),
          picker.archField,
          picker.variantField,
          el('div', { class: 'tr-model-box' }, el('span', { class: 'sr-only' }, `${t('tr.model')}: `), picker.summary, modelNote, modelChanges)),
        el('div', { class: 'tr-col' },
          el('fieldset', { class: 'field' }, el('legend', null, t('tr.start')), startBox, startHelp),
          el('fieldset', { class: 'field' }, el('legend', null, t('tr.mode')), modeBox, modeHelp, summary),
          el('div', { class: 'field' }, el('label', { for: 'tr-metric' }, t('tr.select_metric')), metricSel, el('p', { class: 'hint' }, t('tr.select_metric_help'))),
          el('div', { class: 'field' }, el('label', { for: 'tr-device' }, t('tr.device_label')), deviceSel, el('p', { class: 'hint' }, t('tr.device_help'))))),
      el('div', { class: 'tr-adv-wrap' }, advBox),
      formError,
      el('div', { class: 'form-actions' }, blockedWhy, create, el('button', { type: 'button', class: 'btn', onclick: () => dlg.close() }, t('common.cancel'))));
    add(advBox, paramsBox, moreBox);
    const dlg = el('dialog', { class: 'dialog wide tr-dialog', 'aria-label': t('tr.new_title') }, form);

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      if (create.disabled) return;
      create.disabled = true;
      formError.hidden = true;
      try {
        const made = await api('POST', `/projects/${projectId}/runs`, { json: requestBody() });
        dlg.close();
        toast(t('tr.created', { n: made.number }), 'success', 4000);
        await load(true);
      } catch (err) {
        formError.textContent = err.message || String(err);
        formError.hidden = false;
        create.disabled = false;
        formError.scrollIntoView({ block: 'nearest' });
      }
    });
    dlg.addEventListener('close', () => { dlg.remove(); returnFocus(trigger, opener, node.querySelector('.ds-empty-actions .btn')); });
    drawFolds();
    drawStale();
    drawModel();
    document.body.appendChild(dlg);
    openModal(dlg);
    datasetSel.focus();
  }

  // ---------------------------------------------------------------- charts and tables (no libraries)
  const SVGNS = 'http://www.w3.org/2000/svg';
  function svg(tag, attrs, ...kids) {
    const n = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs || {})) if (v != null) n.setAttribute(k, String(v));
    for (const k of kids) if (k) n.appendChild(typeof k === 'string' ? document.createTextNode(k) : k);
    return n;
  }
  /* series: [{name, cls, points: [[epoch, value], ...]}]; bestEpoch draws a vertical marker */
  /* `label` writes a value of the axis and of the points (three decimals by default; a learning rate needs significant digits) */
  function lineChart(title, series, bestEpoch, label = (v) => nf3.format(Math.round(v * 1000) / 1000)) {
    const W = 440;
    const H = 210;
    const m = { l: 46, r: 14, t: 12, b: 28 };
    const all = series.flatMap((s) => s.points).filter(([, y]) => Number.isFinite(y));
    if (!all.length) return null;
    const xs = all.map((p) => p[0]);
    let x0 = Math.min(...xs);
    let x1 = Math.max(...xs);
    if (x0 === x1) { x0 -= 1; x1 += 1; }
    let y0 = Math.min(...all.map((p) => p[1]));
    let y1 = Math.max(...all.map((p) => p[1]));
    if (y0 === y1) { y0 -= 0.5; y1 += 0.5; }
    const pad = (y1 - y0) * 0.08;
    y0 = y0 >= 0 ? Math.max(0, y0 - pad) : y0 - pad;   // metrics and losses are never negative
    y1 += pad;
    const X = (x) => m.l + ((x - x0) / (x1 - x0)) * (W - m.l - m.r);
    const Y = (y) => H - m.b - ((y - y0) / (y1 - y0)) * (H - m.t - m.b);
    const kids = [];
    for (let i = 0; i <= 3; i++) {
      const v = y0 + ((y1 - y0) * i) / 3;
      kids.push(svg('line', { class: 'tr-grid', x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }));
      kids.push(svg('text', { class: 'tr-axis', x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end' }, label(v)));
    }
    const step = Math.max(1, Math.ceil((x1 - x0) / 6));
    for (let x = Math.ceil(x0); x <= x1; x += step) kids.push(svg('text', { class: 'tr-axis', x: X(x), y: H - 8, 'text-anchor': 'middle' }, String(x)));
    if (bestEpoch != null && bestEpoch >= x0 && bestEpoch <= x1) kids.push(svg('line', { class: 'tr-best', x1: X(bestEpoch), x2: X(bestEpoch), y1: m.t, y2: H - m.b }));
    series.forEach((s, i) => {
      const pts = s.points.filter(([, y]) => Number.isFinite(y));
      if (!pts.length) return;
      kids.push(svg('polyline', { class: `tr-line s${i + 1}`, fill: 'none', points: pts.map(([x, y]) => `${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join(' ') }));
      if (pts.length <= 40) pts.forEach(([x, y]) => kids.push(svg('circle', { class: `tr-pt s${i + 1}`, cx: X(x), cy: Y(y), r: x === bestEpoch ? 4.5 : 2.8 }, svg('title', null, `${t('tr.epoch_n', { n: x })}: ${s.name} ${label(y)}`))));
    });
    const last = series.map((s) => { const p = s.points[s.points.length - 1]; return p ? `${s.name} ${label(p[1])}` : ''; }).filter(Boolean).join(', ');
    const chart = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'tr-svg', role: 'img', 'aria-label': `${title}: ${last}` }, ...kids);
    return el('figure', { class: 'tr-chart' },
      el('figcaption', null, el('strong', null, title)),
      chart,
      el('ul', { class: 'tr-legend' },
        ...series.map((s, i) => (s.points.some(([, y]) => Number.isFinite(y)) ? [s, i] : null)).filter(Boolean).map(([s, i]) => el('li', null, el('span', { class: `tr-swatch s${i + 1}`, 'aria-hidden': 'true' }), s.name)),
        bestEpoch != null ? el('li', null, el('span', { class: 'tr-swatch best', 'aria-hidden': 'true' }), t('tr.best_epoch_n', { n: bestEpoch })) : null));
  }

  function confusionTable(matrix, classes) {
    const n = classes.length;
    const rowSum = matrix.map((row) => row.reduce((a, b) => a + b, 0));
    const colSum = classes.map((_, j) => matrix.reduce((a, row) => a + (row[j] || 0), 0));
    let keep = classes.map((_, i) => i).filter((i) => rowSum[i] > 0 || colSum[i] > 0);
    if (!keep.length) return null;
    const LIMIT = 40;
    const total = keep.length;
    if (total > LIMIT) keep = keep.sort((a, b) => rowSum[b] + colSum[b] - rowSum[a] - colSum[a]).slice(0, LIMIT).sort((a, b) => a - b);
    const cell = (i, j) => {
      const v = (matrix[i] || [])[j] || 0;
      const share = rowSum[i] ? v / rowSum[i] : 0;
      return el('td', { class: `tr-cell${i === j ? ' diag' : ''}${v === 0 ? ' zero' : ''}`, style: v ? `--share:${Math.round(share * 55)}%` : null, title: `${classes[i]} → ${classes[j]}: ${v}` }, v ? num(v) : '·');
    };
    return el('div', { class: 'tr-block' },
      el('h4', null, t('tr.confusion')),
      el('p', { class: 'muted tr-note' }, t('tr.confusion_help'), total > LIMIT ? ` ${t('tr.shown_of', { n: LIMIT, total: num(total) })}` : ''),
      el('div', { class: 'tr-table-wrap' }, el('table', { class: 'tr-table tr-confusion' },
        el('caption', { class: 'sr-only' }, t('tr.confusion')),
        el('thead', null, el('tr', null, el('th', { scope: 'col', class: 'corner' }, t('tr.actual_vs_pred')), ...keep.map((j) => el('th', { scope: 'col', title: classes[j] }, classes[j])))),
        el('tbody', null, ...keep.map((i) => el('tr', null, el('th', { scope: 'row', title: classes[i] }, classes[i]), ...keep.map((j) => cell(i, j))))))),
      n > keep.length && total <= LIMIT ? el('p', { class: 'muted tr-note' }, tn('tr.hidden_classes', n - keep.length)) : null);
  }

  function perClassTable(per) {
    const rows = Object.entries(per || {}).filter(([, v]) => v && typeof v === 'object');
    if (!rows.length) return null;
    const order = ['precision', 'recall', 'f1', 'support'];
    const keys = [];
    rows.forEach(([, v]) => Object.keys(v).forEach((k) => { if (!keys.includes(k) && typeof v[k] === 'number') keys.push(k); }));
    keys.sort((a, b) => (order.indexOf(a) + 1 || 99) - (order.indexOf(b) + 1 || 99));
    const shown = rows.filter(([, v]) => !('support' in v) || v.support > 0 || keys.some((k) => k !== 'support' && v[k] > 0));
    const fmtCell = (k, v) => (k === 'support' || (Number.isInteger(v) && v > 1) ? num(v) : nf3.format(v));
    return el('div', { class: 'tr-block' },
      el('h4', null, t('tr.per_class')),
      el('div', { class: 'tr-table-wrap' }, el('table', { class: 'tr-table' },
        el('thead', null, el('tr', null, el('th', { scope: 'col' }, t('tr.class')), ...keys.map((k) => el('th', { scope: 'col' }, k === 'support' ? t(keys.includes('iou') ? 'tr.support.pixels' : keys.includes('ap50') ? 'tr.support.objects' : 'tr.metric.support') : metricName(k))))),
        el('tbody', null, ...shown.map(([name, v]) => el('tr', null, el('th', { scope: 'row', title: name }, name), ...keys.map((k) => el('td', null, v[k] == null ? '' : fmtCell(k, v[k])))))))));
  }

  function partResults(part) {
    const scalars = Object.entries(part).filter(([, v]) => typeof v === 'number');
    const classes = part.classes || Object.keys(part.per_class || {});
    return el('div', { class: 'tr-part' },
      scalars.length ? el('dl', { class: 'tr-stats' }, ...scalars.map(([k, v]) => el('div', { class: 'tr-stat' }, el('dt', null, metricName(k)), el('dd', null, k === 'images' ? num(v) : nf3.format(v))))) : null,
      Array.isArray(part.confusion_matrix) ? confusionTable(part.confusion_matrix, classes) : null,
      perClassTable(part.per_class));
  }

  // ---------------------------------------------------------------- YAML (built here from the config object)
  function yamlScalar(v) {
    if (v === null || v === undefined) return 'null';
    if (typeof v === 'boolean' || typeof v === 'number') return String(v);
    const s = String(v);
    return /^[A-Za-z][\w.-]*$/.test(s) && !/^(true|false|null|yes|no|on|off|y|n)$/i.test(s) ? s : JSON.stringify(s);
  }
  function toYaml(obj, pad = '') {
    const lines = [];
    for (const [k, v] of Object.entries(obj)) {
      if (Array.isArray(v)) lines.push(`${pad}${k}: [${v.map(yamlScalar).join(', ')}]`);
      else if (v && typeof v === 'object') {
        if (Object.keys(v).length) { lines.push(`${pad}${k}:`); lines.push(toYaml(v, `${pad}  `)); } else lines.push(`${pad}${k}: {}`);
      } else lines.push(`${pad}${k}: ${yamlScalar(v)}`);
    }
    return lines.join('\n');
  }
  async function copyText(text) {
    try { await navigator.clipboard.writeText(text); return true; } catch (e) { /* falls back below */ }
    const area = el('textarea', { class: 'sr-only', 'aria-hidden': 'true', tabindex: '-1' }, text);
    document.body.appendChild(area);
    area.select();
    let ok = false;
    try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
    area.remove();
    return ok;
  }

  // ---------------------------------------------------------------- run detail
  function openDetail(number) {
    const body = el('div', { class: 'dialog-body tr-detail' });
    const live = el('p', { class: 'sr-only', 'aria-live': 'polite' });
    const dlg = el('dialog', { class: 'dialog wide tr-dialog', 'aria-label': t('tr.detail_title', { n: number }) }, body, live);
    let data = null;
    let sig = '';
    let part = null;
    let dtimer = null;
    let lastLive = '';

    function configBlock(r) {
      if (r.config_error) return el('section', { class: 'tr-section', 'aria-labelledby': 'trd-config' }, el('h3', { id: 'trd-config', class: 'section-title' }, t('tr.config')), configError(r));
      const c = r.config || {};
      const m = modelOf(c.model || r.model);
      const params = Object.entries(c.params || {});
      const grid = Object.entries(c.grid || {});
      const row = (k, v) => (v ? el('div', { class: 'tr-kv' }, el('dt', null, k), el('dd', null, v)) : null);
      const yaml = toYaml(c);
      const pkgs = Object.entries(r.packages || {}).map(([k, v]) => `${k} ${v}`).join(', ');
      return el('section', { class: 'tr-section', 'aria-labelledby': 'trd-config' },
        el('h3', { id: 'trd-config', class: 'section-title' }, t('tr.config')),
        el('dl', { class: 'tr-dl' },
          row(t('tr.model'), m ? m.label : c.model),
          row(t('tr.dataset'), datasetText(r)),
          row(t('tr.start'), startName(c.start || r.start)),
          row(t('tr.mode'), c.mode === 'grid' ? t('tr.mode_grid') : t('tr.mode_single')),
          row(t('tr.select_metric'), metricName(c.metric || (m && m.default_metric) || '')),
          row(t('tr.device'), r.setup && r.setup.device ? '' : deviceOf(r)),   // else in "what the training used"
          row(t('tr.duration'), durationOf(r)),
          row(t('tr.started'), r.started ? fmtDateTime(r.started) : ''),
          pkgs ? el('div', { class: 'tr-kv wide' }, el('dt', null, t('tr.packages')), el('dd', { class: 'tr-pkgs' }, pkgs)) : null),
        params.length || grid.length
          ? el('ul', { class: 'tr-chips', 'aria-label': t('tr.params') },
            ...params.map(([k, v]) => el('li', null, el('span', { class: 'muted' }, `${paramName(k)} `), typeof v === 'string' ? choiceName(v) : plain(v))),
            ...grid.map(([k, v]) => el('li', { class: 'grid' }, el('span', { class: 'muted' }, `${paramName(k)} `), `[${v.map((x) => (typeof x === 'string' ? choiceName(x) : plain(x))).join(', ')}]`)))
          : null,
        el('details', { class: 'tr-yaml' },
          el('summary', null, t('tr.yaml')),
          el('div', { class: 'tr-yaml-bar' },
            el('button', { type: 'button', class: 'btn small', 'data-act': 'copy', onclick: async () => { toast(t((await copyText(yaml)) ? 'tr.yaml_copied' : 'tr.yaml_failed'), 'info', 2500); } }, icon('copy', 14), t('tr.copy_yaml'))),
          el('pre', { class: 'tr-pre', tabindex: '0', 'aria-label': t('tr.yaml') }, yaml)));
    }

    /* what the training really used, which may differ from what was asked (a halved batch, no mixed precision on
       the CPU): recorded by the worker as `setup` (#529) */
    function setupBlock(r) {
      const su = r.setup || {};
      if (!Object.keys(su).length && !r.resize) return null;
      const asked = ((r.config || {}).params || {}).batch_size;
      const device = !su.device ? '' : su.device.startsWith('cuda') ? t('tr.device_gpu', { n: su.device.split(':')[1] || '0' })
        : su.device === 'mps' ? t('tr.device_mps') : su.device === 'cpu' ? t('tr.device_cpu') : su.device;
      const batch = su.batch_size == null ? ''
        : asked != null && Number(asked) !== Number(su.batch_size) ? t('tr.setup_batch_asked', { used: num(su.batch_size), asked: num(asked) }) : num(su.batch_size);
      const row = (k, v) => (v ? el('div', { class: 'tr-kv' }, el('dt', null, k), el('dd', null, v)) : null);
      return el('section', { class: 'tr-section', 'aria-labelledby': 'trd-setup' },
        el('h3', { id: 'trd-setup', class: 'section-title' }, t('tr.setup')),
        el('dl', { class: 'tr-dl' },
          row(t('tr.device'), device),
          row(paramName('amp'), su.amp == null ? '' : t(su.amp ? 'tr.setup_amp_on' : 'tr.setup_amp_off')),
          row(paramName('batch_size'), batch),
          row(paramName('resize'), r.resize ? choiceName(r.resize) : '')));
    }

    function progressBlock(r) {
      if (!ACTIVE(r)) return el('div', null, r.status === 'failed' && runError(r) ? el('p', { class: 'tr-fail' }, icon('error', 16), el('span', null, runError(r))) : null, warningsBlock(r));
      const pct = percentOf(r);
      return el('div', { class: 'tr-live' },
        el('p', { class: 'tr-live-line' }, el('strong', null, phaseText(r)), r.message ? el('span', { class: 'muted' }, ` · ${r.message}`) : null),
        pct != null ? el('progress', { class: 'tr-progress', max: '100', value: String(pct), 'aria-label': t('common.progress') }) : null);
    }

    let childSel = null;   // training (child) whose curves and results are shown; null = the run's own
    let childData = null;
    const valueText = (k, v) => (k === 'start' ? startName(v) : typeof v === 'string' ? choiceName(v) : plain(v));
    const childLabel = (c) => [
      Object.entries(c.values || {}).map(([k, v]) => `${k === 'start' ? t('tr.start') : paramName(k)} ${valueText(k, v)}`).join(', '),
      c.fold != null ? t('ds.fold', { n: Number(c.fold) + 1 }) : ''].filter(Boolean).join(' · ') || c.id;

    async function showChild(id) {
      if (!id) { childSel = null; childData = null; sig = ''; render(); return; }
      try {
        childData = await api('GET', `/projects/${projectId}/runs/${number}/children/${id}`);
        childSel = id;
        sig = '';
        render();
      } catch (err) { toastError(err); }
    }

    function trainingsBlock(r) {
      const kids = r.children || [];
      if (!kids.length) return null;
      const res = r.results || {};
      const metric = (r.config && r.config.metric) || res.metric || '';
      const names = Object.keys(kids[0].values || {});
      const perCombo = kids.filter((c) => c.combo === kids[0].combo).length;
      const shown = childSel || (ACTIVE(r) ? r.current : r.best_child);
      const viewBtn = (id, label) => el('button', {
        type: 'button', class: `btn small${shown === id ? ' primary' : ''}`, 'aria-pressed': String(shown === id),
        'aria-label': t('tr.show_curves_of', { name: label }),
        onclick: () => showChild(id === (ACTIVE(r) ? r.current : r.best_child) ? null : id) }, t(shown === id ? 'tr.showing' : 'tr.show_curves'));
      const head = el('tr', null,
        el('th', { scope: 'col' }, '#'),
        ...names.map((k) => el('th', { scope: 'col' }, k === 'start' ? t('tr.start') : paramName(k))),
        perCombo > 1 ? el('th', { scope: 'col' }, t('tr.per_fold')) : null,
        el('th', { scope: 'col' }, res.ranking && perCombo > 1 ? t('tr.mean_std', { m: metricName(metric) }) : metricName(metric)),
        el('th', { scope: 'col' }, el('span', { class: 'sr-only' }, t('tr.curves'))));
      let rows;
      if (res.ranking) {
        rows = res.ranking.map((row) => {
          const ids = row.children || [];
          const scores = row.scores || {};
          const pick = ids.filter((id) => id in scores).sort((a, b) => scores[b] - scores[a])[0] || null;
          const entry = kids.find((c) => c.id === (pick || ids[0])) || { values: row.values };
          const failed = ids.map((id) => kids.find((c) => c.id === id)).filter((c) => c && c.status === 'failed');
          const mean = row.mean == null ? '—' : perCombo > 1 ? `${nf3.format(row.mean)} ± ${nf3.format(row.std || 0)}` : nf3.format(row.mean);
          return el('tr', { class: row.combo === res.best_combo ? 'best' : null },
            el('th', { scope: 'row' }, row.mean == null ? '—' : String(row.rank)),
            ...names.map((k) => el('td', null, valueText(k, (row.values || {})[k]))),
            perCombo > 1 ? el('td', { class: 'muted' }, ids.map((id) => (id in scores ? nf3.format(scores[id]) : '—')).join(' · ')) : null,
            el('td', { class: 'num', title: failed.length ? failed.map((c) => c.error).join('\n') : null },
              el('strong', null, mean), failed.length ? el('span', { class: 'tr-fail-note' }, ` ${tn('tr.failed_n', failed.length)}`) : null),
            el('td', null, pick ? viewBtn(pick, childLabel(entry)) : null));
        });
      } else {
        rows = kids.map((c, i) => el('tr', { class: c.id === r.current ? 'current' : null },
          el('th', { scope: 'row' }, String(i + 1)),
          ...names.map((k) => el('td', null, valueText(k, (c.values || {})[k]))),
          perCombo > 1 ? el('td', null, t('ds.fold', { n: Number(c.fold) + 1 })) : null,
          el('td', { class: 'num', title: c.error || null }, c.best && c.best.value != null ? nf3.format(c.best.value) : el('span', { class: 'muted' }, phaseText(c))),
          el('td', null, c.status !== 'queued' ? viewBtn(c.id, childLabel(c)) : null)));
      }
      return el('section', { class: 'tr-section', 'aria-labelledby': 'trd-children' },
        el('h3', { id: 'trd-children', class: 'section-title' }, res.ranking ? t('tr.ranking') : t('tr.trainings_title')),
        el('p', { class: 'muted tr-note' }, res.ranking ? t(perCombo > 1 ? 'tr.ranking_help_folds' : 'tr.ranking_help', { k: perCombo }) : tn('tr.trainings', kids.length)),
        el('div', { class: 'tr-table-wrap' }, el('table', { class: 'tr-table tr-rank' },
          el('caption', { class: 'sr-only' }, res.ranking ? t('tr.ranking') : t('tr.trainings_title')),
          el('thead', null, head), el('tbody', null, ...rows))));
    }

    function viewingNote(r) {
      const kids = r.children || [];
      if (!kids.length) return null;
      const id = childSel || (ACTIVE(r) ? r.current : r.best_child);
      const c = kids.find((x) => x.id === id);
      if (!c) return null;
      const key = childSel ? 'tr.viewing' : ACTIVE(r) ? 'tr.viewing_current' : 'tr.viewing_best';
      return el('p', { class: 'tr-viewing' }, icon('chart', 16), el('span', null, t(key, { name: childLabel(c) })),
        childSel ? el('button', { type: 'button', class: 'btn small', onclick: () => showChild(null) }, t(ACTIVE(r) ? 'tr.view_current' : 'tr.view_best')) : null);
    }

    function chartsBlock(r) {
      const rows = r.metrics || [];
      if (!rows.length) return null;
      const c = r.config || {};
      const m = modelOf(c.model || r.model);
      const metric = c.metric || (r.results && r.results.metric) || (m && m.default_metric) || '';
      const keyOf = rows.some((x) => `val_${metric}` in x) ? `val_${metric}` : Object.keys(rows[rows.length - 1]).find((k) => k.startsWith('val_') && k !== 'val_loss');
      const bestRow = [...rows].reverse().find((x) => x.best);
      const best = (r.results && r.results.best_epoch) || (r.best && r.best.epoch) || (bestRow && bestRow.epoch) || null;
      const pts = (k) => rows.filter((x) => x[k] != null).map((x) => [x.epoch, x[k]]);
      const loss = lineChart(t('tr.chart_loss'), [{ name: t('tr.train'), points: pts('train_loss') }, { name: t('tr.val'), points: pts('val_loss') }], best);
      const score = keyOf ? lineChart(`${metricName(metric || keyOf.slice(4))} (${t('tr.val').toLowerCase()})`, [{ name: metricName(keyOf.slice(4)), points: pts(keyOf) }], best) : null;
      /* the learning rate of each epoch, as the schedule and the warmup set it (#529) */
      /* learning rates are small numbers: scientific notation keeps the axis labels short (1,5E-4) */
      const sig = new Intl.NumberFormat(LOCALE, { notation: 'scientific', maximumSignificantDigits: 2 });
      const lr = pts('lr').length ? lineChart(paramName('lr'), [{ name: paramName('lr'), points: pts('lr') }], null, (v) => (v === 0 ? '0' : sig.format(v))) : null;
      return el('section', { class: 'tr-section', 'aria-labelledby': 'trd-curves' },
        el('h3', { id: 'trd-curves', class: 'section-title' }, t('tr.curves')),
        el('div', { class: 'tr-charts' }, loss, score, lr));
    }

    function resultsBlock(r) {
      const res = r.results;
      if (!res) return null;
      const parts = ['val', 'test'].filter((k) => res[k] && typeof res[k] === 'object');
      if (!parts.length) return null;
      if (!parts.includes(part)) part = parts[0];
      const pick = parts.length > 1
        ? el('div', { class: 'seg tr-seg', role: 'radiogroup', 'aria-label': t('tr.results') },
          ...parts.map((k) => {
            const input = el('input', { type: 'radio', name: 'trd-part', id: `trd-part-${k}`, value: k, class: 'sr-only', checked: part === k, onchange: () => { part = k; sig = ''; render(); } });
            return el('label', { for: input.id, class: 'seg-opt' }, input, k === 'val' ? t('tr.val') : t('tr.test'));
          }))
        : null;
      return el('section', { class: 'tr-section', 'aria-labelledby': 'trd-results' },
        el('h3', { id: 'trd-results', class: 'section-title' }, t('tr.results')),
        res.best_epoch != null ? el('p', { class: 'muted' }, t('tr.best_summary', { best: res.best_epoch, n: res.epochs_run != null ? res.epochs_run : res.best_epoch }), res.stopped_early ? ` ${t('tr.stopped_early')}` : '') : null,
        pick, partResults(res[part]));
    }

    function render() {
      const r = data;
      const view = childSel && childData ? { ...childData, number: r.number } : r;
      const s = JSON.stringify(r) + part + (childSel ? JSON.stringify(childData) : '');
      if (s === sig) return;
      sig = s;
      const top = body.scrollTop;
      keepFocus(body, () => fill(body,
        el('div', { class: 'tr-detail-head' },
          el('h2', null, t('tr.detail_title', { n: r.number })),
          badge(r),
          el('button', { type: 'button', class: 'icon-btn', 'data-act': 'close', 'aria-label': t('common.close'), onclick: () => dlg.close() }, icon('close', 18))),
        progressBlock(r), alertsBlock(r), trainingsBlock(r), viewingNote(r), chartsBlock(view), resultsBlock(view), setupBlock(view), configBlock(r)));
      body.scrollTop = top;
      const text = phaseText(r);
      if (text !== lastLive) { lastLive = text; live.textContent = text; }
    }

    async function refresh() {
      clearTimeout(dtimer);
      try {
        data = await api('GET', `/projects/${projectId}/runs/${number}`);
        if (childSel) childData = await api('GET', `/projects/${projectId}/runs/${number}/children/${childSel}`);
      } catch (err) {
        if (err.status === 404) { dlg.close(); toastError(err); return; }
        if (!data) { dlg.close(); toastError(err); return; }
      }
      if (!dlg.isConnected) return;
      render();
      if (ACTIVE(data)) dtimer = setTimeout(refresh, 2000);
    }

    dlg.addEventListener('close', () => { clearTimeout(dtimer); dlg.remove(); load(true); });
    document.body.appendChild(dlg);
    openModal(dlg);
    add(body, el('p', { class: 'muted' }, t('common.loading')));
    refresh().then(() => { const close = dlg.querySelector('[data-act="close"]'); if (close) close.focus(); });
  }

  // ---------------------------------------------------------------- assemble
  const newBtn = el('button', { type: 'button', class: 'btn primary', id: 'tr-new', onclick: once(() => openNew(newBtn)) }, icon('plus', 16), t('tr.new'));
  add(node,
    el('div', { class: 'ds-head' },
      el('div', null, el('h2', { id: 'h-training' }, t('step.training')), el('p', { class: 'muted' }, t('tr.intro'))),
      newBtn),
    announcer,
    cmpBarBox,
    listBox,
    cmpBox);

  panel.mount = () => load();
  panel.show = () => load(true);
  /** Step 4 sends the person here with a dataset version to train on. */
  panel.startWith = (version) => openNew(newBtn, version);
  /** The project changed (its task type may have): the models on offer and their names are asked again. */
  panel.refresh = () => {
    catalog = null;
    cards.clear();
    if (panel.mounted) load(true);
  };
  /** Run counts for the step's status line, before the panel is opened. */
  panel.count = async () => {
    try {
      runs = (await api('GET', `/projects/${projectId}/runs`)).runs || [];
      ctx.runCount = runs.length;
      ctx.runActive = runs.filter(ACTIVE).length;
      renderStepper();
    } catch (err) { /* the status stays "no runs" */ }
  };
  return panel;
}
