/* Step 3, Curation: measure the images, flag the ones that may hurt the dataset, exclude or restore them, and
 * the thresholds of the checks. */
import { imageGrid, openLightbox } from './grid.js';
import { ISSUES, issueName } from './names.js';
import { similarityBlock } from './similarity.js';
import { api } from '../shell/api.js';
import { confirmDialog, once } from '../shell/dialogs.js';
import { add, el, esc, fill, icon, redraw } from '../shell/dom.js';
import { fmt, nf1, num, t, tn, tr } from '../shell/i18n.js';
import { toast, toastError } from '../shell/toast.js';

export function curationPanel(page) {
  const { ctx, jobStarted, kindActive, projectId, refreshProject, renderStepper } = page;
  const node = el('section', { class: 'panel', id: 'panel-curation', hidden: true, 'aria-labelledby': 'h-curation' });
  const panel = { node, mounted: false };
  let summary = null;
  let issue = '';
  let view = 'active';
  let limitsOpen = false;
  let firstLoad = true;
  const ANY = '*';  // the quick filter "only with issues": the server filter issue=any
  const noticeBox = el('div', { class: 'cur-notice' });
  const chipsBox = el('div', { class: 'issue-chips', role: 'group', 'aria-label': t('cur.filter') });
  const histBox = el('div', { class: 'hists' });
  const advBox = el('div', { class: 'advanced', id: 'cur-limits', hidden: true });
  const actionsBar = el('div', { class: 'cur-actions' });
  const toolbar = el('div', { class: 'toolbar' });
  const tabs = el('div', { class: 'tabs', role: 'group', 'aria-label': t('cur.view') });
  const countText = el('span', { class: 'muted', 'aria-live': 'polite' });
  const sim = similarityBlock(page);

  const grid = imageGrid({
    projectId,
    selectable: true,
    query: () => ({ status: view, issue: view === 'active' ? (issue === ANY ? 'any' : issue) : '', emptyText: view === 'excluded' ? t('cur.no_excluded') : t('cur.no_match') }),
    onSelect: (sel) => { countText.textContent = sel.size ? tn('cur.selected', sel.size) : ''; drawToolbar(); },
    onOpen: (items, i) => openLightbox(projectId, items, i),
  });

  ctx.curationSelected = () => [...grid.selected];

  const analyzeBtn = (recompute) =>
    el('button', { type: 'button', class: `btn${recompute ? '' : ' primary'}`, disabled: kindActive('analyze'), onclick: once(() => analyze(recompute), () => kindActive('analyze')) },
      recompute ? t('cur.reanalyze') : t('cur.analyze'));
  let analyzing = false;
  ctx.watchers.push(() => {
    if (panel.mounted && analyzing !== kindActive('analyze')) { analyzing = !analyzing; drawNotice(); }
  });

  async function analyze(recompute) {
    try {
      const job = await api('POST', `/projects/${projectId}/curation/analyze`, { json: { recompute: !!recompute } });
      toast(t('cur.started'), 'info', 3000);
      await jobStarted(job);
    } catch (err) {
      toastError(err);
    }
  }

  async function resetAnalysis() {
    const ok = await confirmDialog({ title: t('cur.reset_title'), text: t('cur.reset_text'), confirmLabel: t('cur.reset') });
    if (!ok) return;
    try {
      await api('DELETE', `/projects/${projectId}/curation`);
      toast(t('cur.reset_done'), 'success', 4000);
      await refreshProject();
      load();
    } catch (err) { toastError(err); }
  }

  async function load() {
    try {
      summary = await api('GET', `/projects/${projectId}/curation`);
    } catch (err) {
      summary = null;
      if (err.status !== 404) toastError(err);
    }
    if (summary && ctx.project) {
      const ids = new Set();
      Object.values(summary.issue_ids || {}).forEach((l) => l.forEach((x) => ids.add(x)));
                ctx.project.counts = { ...(ctx.project.counts || {}), analyzed: (summary.totals || {}).measured || 0, with_issues: ids.size };
      renderStepper();
    }
    if (firstLoad && analyzed()) { firstLoad = false; if (flaggedIds()) issue = ANY; }  // problems first
    else if (issue === ANY && !flaggedIds()) issue = '';
    draw();
    grid.reload();
  }

  const analyzed = () => summary && ((summary.totals || {}).measured || 0) > 0;
  /* ids of every image with an issue (the union of the per-issue lists), or null when there is none */
  function flaggedIds() {
    const ids = new Set();
    Object.values((summary && summary.issue_ids) || {}).forEach((l) => l.forEach((x) => ids.add(x)));
    return ids.size ? [...ids] : null;
  }

  function draw() {
    drawNotice();
    drawActions();
    drawChips();
    drawHists();
    drawAdvanced();
    drawTabs();
    drawToolbar();
    histBox.hidden = !analyzed();
    chipsBox.hidden = !analyzed() || view !== 'active';
    advBox.hidden = !limitsOpen || !summary || !summary.thresholds;
  }

  function drawNotice() {
    if (analyzed()) {
      const totals = summary.totals || {};
      fill(noticeBox, el('p', { class: 'muted' }, t('cur.measured', { measured: num(totals.measured || 0), total: num(totals.active || 0) })));
    } else {
      fill(noticeBox, el('div', { class: 'callout' }, icon('info'),
        el('div', null, el('strong', null, t('cur.not_analyzed')), el('p', { class: 'muted' }, t('cur.intro')), analyzeBtn(false))));
    }
  }

  /* the actions of the step, at the top: the safe ones together, "reset" apart and quiet */
  function drawActions() {
    const limitsBtn = summary && summary.thresholds ? el('button', {
      type: 'button', class: 'btn', 'aria-expanded': String(limitsOpen), 'aria-controls': 'cur-limits',
      onclick: () => { limitsOpen = !limitsOpen; draw(); },
    }, t('cur.limits')) : null;
    redraw(actionsBar,
      analyzed() ? analyzeBtn(true) : null, openFoBtn, pullFoBtn, limitsBtn,
      analyzed() ? el('button', { type: 'button', class: 'btn quiet-danger cur-reset', onclick: resetAnalysis }, t('cur.reset')) : null);
  }

  function drawChips() {
    const totals = (summary && summary.totals) || {};
    const counts = (summary && summary.issues) || {};
    const mk = (key, label, n, help) =>
      el('button', {
        type: 'button', class: `chip-btn${issue === key ? ' active' : ''}`, 'aria-pressed': issue === key ? 'true' : 'false',
        title: help || null, 'data-issue': key, 'data-act': key,
        onclick: () => { issue = key; drawChips(); grid.reload(); },
      }, el('span', null, label), el('span', { class: 'count' }, num(n)));
    const flagged = flaggedIds();
    redraw(chipsBox,
      mk('', t('cur.all'), totals.active || 0, t('cur.all.help')),
      flagged ? mk(ANY, t('cur.only_issues'), flagged.length, t('cur.only_issues.help')) : null,
      ...ISSUES.concat(Object.keys(counts).filter((k) => !ISSUES.includes(k))).map((k) => mk(k, issueName(k), counts[k] || 0, tr(`issue.${k}.help`, ''))));
  }

  function drawHists() {
    const h = (summary && summary.histograms) || {};
    const shown = ['brightness', 'contrast', 'sharpness'].filter((m) => h[m] && (h[m].counts || []).length);
    fill(histBox, ...shown.map((m) => histogram(m, h[m])), shown.length ? el('p', { class: 'muted hist-legend' }, t('cur.hist_legend')) : null);
  }

  function histogram(metric, data) {
    const counts = data.counts;
    const edges = data.edges || [];
    const max = Math.max(1, ...counts);
    const w = 200;
    const hgt = 60;
    const bw = w / counts.length;
    const bars = counts.map((c, i) => {
      const bh = Math.round((c / max) * (hgt - 4));
      return `<rect x="${(i * bw + 0.5).toFixed(1)}" y="${hgt - bh}" width="${(bw - 1).toFixed(1)}" height="${bh}" rx="1"></rect>`;
    }).join('');
    const lo = edges.length ? edges[0] : null;
    const hi = edges.length ? edges[edges.length - 1] : null;
    const fmt = (x) => (x == null ? '' : nf1.format(x));
    const name = tr(`metric.${metric}`, metric);
    const label = `${name}: ${fmt(lo)} – ${fmt(hi)}; ${t('cur.hist_count', { n: num(max) })}`;
    return el('figure', { class: 'hist' },
      el('figcaption', null, el('strong', null, name), el('span', { class: 'muted' }, t('cur.hist_count', { n: num(max) }))),
      el('div', { html: `<svg viewBox="0 0 ${w} ${hgt}" width="100%" role="img" aria-label="${esc(label)}" preserveAspectRatio="none">${bars}</svg>` }),
      el('div', { class: 'hist-axis', 'aria-hidden': 'true' }, el('span', null, fmt(lo)), el('span', null, lo != null ? fmt((lo + hi) / 2) : ''), el('span', null, fmt(hi))));
  }

  function drawAdvanced() {
    const th = summary && summary.thresholds;
    if (!th) { fill(advBox); return; }
    const inputs = Object.entries(th).map(([k, v]) => {
      const id = `th-${k}`;
      return el('div', { class: 'field inline' }, el('label', { for: id }, tr(`threshold.${k}`, k)), el('input', { id, name: k, type: 'number', step: 'any', value: String(v) }));
    });
    const formError = el('p', { class: 'tr-form-error', role: 'alert', hidden: true });
    const form = el('form', { class: 'th-form', novalidate: true }, el('p', { class: 'muted' }, t('cur.thresholds_help')), el('div', { class: 'th-grid' }, ...inputs), formError,
      el('div', { class: 'form-actions' }, el('button', { type: 'submit', class: 'btn primary' }, t('cur.thresholds_apply'))));
    const fail = (message) => { formError.textContent = message; formError.hidden = false; };
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      formError.hidden = true;
      const body = {};
      for (const input of form.querySelectorAll('input')) {
        input.removeAttribute('aria-invalid');
        const value = input.value.trim();
        if (value === '' || !Number.isFinite(Number(value))) {  /* Number('') is 0: an emptied field must not become a threshold */
          input.setAttribute('aria-invalid', 'true');
          input.focus();
          fail(t('cur.threshold_number', { name: tr(`threshold.${input.name}`, input.name) }));
          return;
        }
        body[input.name] = Number(value);
      }
      try {
        summary = await api('PUT', `/projects/${projectId}/curation/thresholds`, { json: body });
        toast(t('cur.thresholds_saved'), 'success', 3000);
        await load();
      } catch (err) {
        fail(err.message);  /* the server explains a refused value (out of range) next to the fields */
      }
    });
    fill(advBox, form);
  }

  function drawTabs() {
    const excluded = ((ctx.project || {}).counts || {}).excluded || 0;
    redraw(tabs,
      ...['active', 'excluded'].map((v) =>
        el('button', {
          type: 'button', class: `tab${view === v ? ' active' : ''}`, 'aria-pressed': view === v ? 'true' : 'false', 'data-view': v, 'data-act': v,
          onclick: () => { view = v; draw(); grid.reload(); },
        }, v === 'active' ? t('cur.tab_active') : `${t('cur.tab_excluded')} (${num(excluded)})`)));
  }

  function drawToolbar() {
    const n = grid.selected.size;
    const buttons = [];
    /* the actions on a selection appear once something is selected (no greyed-out buttons) */
    if (view === 'active') {
      buttons.push(
        el('button', { type: 'button', class: 'btn', 'data-act': 'all', title: t('cur.select_all'), onclick: selectAll }, t('cur.select_all_short')),
        el('button', { type: 'button', class: 'btn', 'data-act': 'suggested', disabled: !analyzed(), 'aria-describedby': analyzed() ? null : 'cur-suggested-hint', title: t('cur.select_suggested'), onclick: selectSuggested }, t('cur.select_suggested_short')));
      if (n) {
        buttons.push(
          el('button', { type: 'button', class: 'btn', 'data-act': 'clear', onclick: () => grid.setSelected([]) }, t('cur.clear')),
          el('button', { type: 'button', class: 'btn primary', 'data-act': 'exclude', onclick: once(exclude) }, t('cur.exclude')));
      }
    } else {
      buttons.push(el('button', { type: 'button', class: 'btn', 'data-act': 'all', onclick: selectAll }, t('cur.select_all_excluded')));
      if (n) {
        buttons.push(
          el('button', { type: 'button', class: 'btn', 'data-act': 'clear', onclick: () => grid.setSelected([]) }, t('cur.clear')),
          el('button', { type: 'button', class: 'btn primary', 'data-act': 'restore', onclick: once(restore) }, t('cur.restore')),
          el('button', { type: 'button', class: 'btn danger-outline', 'data-act': 'delete', onclick: once(remove) }, t('cur.delete')));
      }
    }
    const needsAnalysis = view === 'active' && !analyzed();  // "suggested" comes from the analysis
    const hint = n && !needsAnalysis ? null : el('span', { class: 'muted cur-select-hint', id: needsAnalysis ? 'cur-suggested-hint' : null },
      t(needsAnalysis ? 'cur.suggested_needs_analysis' : view === 'active' ? 'cur.select_hint' : 'cur.select_hint_excluded'));
    redraw(toolbar, ...buttons, hint, countText);
  }

  async function selectAll() {
    try { grid.setSelected(await grid.allIds()); } catch (err) { toastError(err); }
  }
  async function selectSuggested() {
    try {
      const data = await api('GET', `/projects/${projectId}/curation/suggested`);
      const ids = Array.isArray(data) ? data : (data && data.ids) || [];
      grid.setSelected(ids);
      toast(ids.length ? tn('cur.suggested_n', ids.length) : t('cur.suggested_none'), 'info', 4000);
    } catch (err) { toastError(err); }
  }
  const after = async () => { await refreshProject(); await load(); };
  async function exclude() {
    const ids = [...grid.selected];
    try {
      const r = await api('POST', `/projects/${projectId}/images/exclude`, { json: { ids, reason: issue || 'manual' } });
      toast(tn('cur.excluded_n', (r && r.changed) ?? ids.length), 'success', 5000);
      await after();
    } catch (err) { toastError(err); }
  }
  async function restore() {
    const ids = [...grid.selected];
    try {
      const r = await api('POST', `/projects/${projectId}/images/restore`, { json: { ids } });
      toast(tn('cur.restored_n', (r && r.changed) ?? ids.length), 'success', 5000);
      await after();
    } catch (err) { toastError(err); }
  }
  async function remove() {
    const ids = [...grid.selected];
    const ok = await confirmDialog({
      title: t('cur.delete_title'), text: tn('cur.delete_text', ids.length), confirmLabel: t('cur.delete_confirm'),
    });
    if (!ok) return;
    try {
      const r = await api('POST', `/projects/${projectId}/images/delete`, { json: { ids, confirm: true } });
      toast(tn('cur.deleted_n', (r && r.deleted) ?? ids.length), 'success', 5000);
      await after();
    } catch (err) { toastError(err); }
  }

  async function openFiftyOne() {
    const view_ = issue ? `curation - ${issue}` : 'curation - quality issues';
    try {
      const job = await api('POST', `/projects/${projectId}/fiftyone`, { json: { view: view_ } });
      if (job && job.id) ctx.openWhenDone.add(job.id);
      toast(t('cur.fo_starting'), 'info', 4000);
      await jobStarted(job);
    } catch (err) { toastError(err); }
  }
  async function pullExclusions() {
    try {
      const r = await api('POST', `/projects/${projectId}/fiftyone/pull-exclusions`);
      toast(tn('cur.pulled_n', (r && r.excluded) || 0), 'success', 5000);
      await after();
    } catch (err) { toastError(err); }
  }

  const openFoBtn = el('button', { type: 'button', class: 'btn', onclick: once(openFiftyOne, () => kindActive('fiftyone')) }, t('cur.open_fo'));
  ctx.watchers.push(() => { openFoBtn.disabled = kindActive('fiftyone'); });
  const pullFoBtn = el('button', { type: 'button', class: 'btn', title: t('cur.fo_hint'), onclick: once(pullExclusions) }, t('cur.pull_fo'));

  add(node,
    el('h2', { id: 'h-curation' }, t('step.curation')),
    noticeBox,
    actionsBar,
    advBox,
    histBox,
    sim.node,
    el('div', { class: 'cur-main' },
      tabs, chipsBox, toolbar, grid.node));
  panel.mount = () => { load(); sim.load(); };
  panel.refresh = () => { load(); sim.load(); };
  return panel;
}
