/* Step 6, Models: the saved models (from training runs and public ones), comparing the chosen ones, downloading,
 * choosing the current one and pre-annotating with one. */
import { compareBar, compareCheck, compareTable } from './compare.js';
import { api } from '../shell/api.js';
import { confirmDialog, openModal } from '../shell/dialogs.js';
import { add, el, fill, icon, redraw } from '../shell/dom.js';
import { LOCALE, fmtBytes, fmtDate, num, t, tn, tr } from '../shell/i18n.js';
import { toast, toastError } from '../shell/toast.js';
import { emptyState } from '../shell/widgets.js';

export function modelsPanel(page) {
  const { ctx, projectId, renderStepper, showTab } = page;
  const node = el('section', { class: 'panel', id: 'panel-models', 'aria-labelledby': 'h-models' });
  const panel = { node, mounted: false };
  const box = el('div', { class: 'md-body' });
  const announcer = el('p', { class: 'sr-only', 'aria-live': 'polite' });
  const nf3 = new Intl.NumberFormat(LOCALE, { maximumFractionDigits: 3 });
  const labels = {};
  let models = [];
  let timer = null;
  let sig = '';
  const metricName = (m) => tr(`tr.metric.${m}`, String(m).replace(/_/g, ' '));
  const startName = (s) => tr(`tr.start_${s}`, s);
  const archName = (m) => m.label || labels[m.model] || m.model;
  const isPublic = (m) => m.source === 'public';
  const busy = (m) => Object.values(m.exports || {}).some((e) => e && (e.status === 'queued' || e.status === 'running'));
  const download = (m, kind) => `/api/projects/${projectId}/models/${m.number}/download/${kind}`;

  /* whether the catalogue has public models for this project type (asked once; null = not known yet) */
  let publicAvailable = null;
  async function ensurePublic() {
    if (publicAvailable !== null) return;
    try {
      publicAvailable = ((await api('GET', `/projects/${projectId}/models/public-catalog`)).models || []).length > 0;
    } catch (err) { publicAvailable = true; }  // the button stays and says why when it is used
  }

  async function ensureLabels() {
    if (Object.keys(labels).length) return;
    try {
      const cat = await api('GET', `/training/catalog?type=${encodeURIComponent(ctx.project.task)}`);
      (cat.models || []).forEach((x) => { labels[x.id] = x.label; });
    } catch (err) { /* the catalogue ids are shown instead */ }
  }

  async function change(m, body, message) {
    try {
      await api('PATCH', `/projects/${projectId}/models/${m.number}`, { json: body });
      if (message) { toast(message, 'success', 3000); announcer.textContent = message; }
    } catch (err) { toastError(err); }
    await load(true);
  }

  function editNote(m) {
    const input = el('textarea', { id: 'md-edit-note', rows: '3', maxlength: '500' }, m.note || '');
    const save = el('button', { type: 'submit', class: 'btn primary' }, t('common.save'));
    const dlg = el('dialog', { class: 'dialog', 'aria-labelledby': 'md-note-t' },
      el('form', { class: 'dialog-body form', method: 'dialog' },
        el('h2', { id: 'md-note-t' }, t('md.note_title', { n: m.number })),
        el('div', { class: 'field' }, el('label', { for: 'md-edit-note' }, t('md.note')), input),
        el('div', { class: 'form-actions' }, save, el('button', { type: 'button', class: 'btn', onclick: () => dlg.close() }, t('common.cancel')))));
    dlg.querySelector('form').addEventListener('submit', async (e) => {
      e.preventDefault();
      dlg.close();
      await change(m, { note: input.value }, t('md.note_saved'));
    });
    dlg.addEventListener('close', () => dlg.remove());
    document.body.appendChild(dlg);
    openModal(dlg);
    input.focus();
  }

  async function exportOnnx(m) {
    try {
      await api('POST', `/projects/${projectId}/models/${m.number}/export`, { json: { format: 'onnx' } });
      announcer.textContent = t('md.export_queued');
    } catch (err) { toastError(err); }
    await load(true);
  }

  async function removeModel(m) {
    const text = [isPublic(m) ? t('md.delete_text_public') : t('md.delete_text'), m.current ? t('md.delete_text_current') : null].filter(Boolean).join(' ');
    const ok = await confirmDialog({ title: t('md.delete_title', { n: m.number }), text, confirmLabel: t('md.delete') });
    if (!ok) return;
    try {
      await api('DELETE', `/projects/${projectId}/models/${m.number}`, { json: { confirm: true } });
      if (m.run != null) ctx.weightless.add(m.run);
      toast(t('md.deleted', { n: m.number }), 'success', 3000);
    } catch (err) { toastError(err); }
    await load(true);
    ctx.panels.training.count();
    ctx.panels.training.refresh();
  }

  const ONNX_KINDS = ['classification', 'semantic'];   // detection graphs are not reliable in ONNX yet
  function onnxBlock(m) {
    const e = (m.exports || {}).onnx || {};
    if (!ONNX_KINDS.includes(m.task_kind) && e.status !== 'done') return el('span', { class: 'muted md-export' }, t('md.onnx_later'));
    if (e.status === 'queued' || e.status === 'running') {
      const text = e.status === 'queued' ? t('md.export_queued') : e.message && e.message.startsWith('train_env_install:') ? t('md.installing', { pkg: e.message.split(':')[1] }) : t('md.exporting');
      return el('span', { class: 'md-export busy' }, el('span', { class: 'tr-dot', 'aria-hidden': 'true' }), text);
    }
    if (e.status === 'done') {
      return el('a', { class: 'btn small', href: download(m, 'onnx'), download: '', 'data-act': 'onnx' }, icon('download', 14), t('md.download_onnx'), el('span', { class: 'muted' }, ` · ${fmtBytes(e.size || 0)}`));
    }
    return el('span', { class: 'md-export' },
      el('button', { type: 'button', class: 'btn small', 'data-act': 'export', onclick: () => exportOnnx(m) }, e.status === 'failed' ? t('md.retry_onnx') : t('md.export_onnx')),
      e.status === 'failed' ? el('span', { class: 'tr-fail' }, icon('error', 14), el('span', null, e.error === 'interrupted' ? t('md.interrupted') : e.error || '')) : null);
  }

  function card(m) {
    const res = m.results || {};
    const val = (res.val || {})[m.metric];
    const test = (res.test || {})[m.metric];
    const id = `model-${m.number}`;
    return el('li', { class: `tr-run card md-model${m.current ? ' current' : ''}`, id, 'aria-labelledby': `${id}-t` },
      el('div', { class: 'tr-run-head' },
        isPublic(m) ? null : compareCheck({ id: `${id}-cmp`, label: `m${m.number}`, checked: chosen.has(m.number), onChange: (on) => choose([m.number], on) }),
        el('h3', { id: `${id}-t` }, `m${m.number}`),
        el('span', { class: 'tr-model' }, archName(m)),
        m.current
          ? el('span', { class: 'tr-badge done md-current' }, icon('star', 14), t('md.current'))
          : el('button', { type: 'button', class: 'btn small', 'data-act': 'current', onclick: () => change(m, { current: true }, t('md.made_current', { n: m.number })) }, t('md.make_current'))),
      isPublic(m)
        ? el('p', { class: 'muted tr-meta' }, el('span', { class: 'md-public' }, t('md.public_badge')), ` ${t('md.public_meta', { date: fmtDate(m.created) })}`)
        : el('p', { class: 'muted tr-meta' }, [t('tr.dataset_short', { n: m.dataset }), startName(m.start), t('md.from_run', { n: m.run }), fmtDate(m.created)].join(' · ')),
      isPublic(m)
        ? el('details', { class: 'md-mapping' }, el('summary', null, tn('md.classes_n', Object.keys(m.mapping || {}).length)),
          el('p', null, el('span', { class: 'muted' }, `${t('md.mapping')}: `),
            Object.entries(m.mapping || {}).map(([from, to]) => (from === to ? from : `${from} → ${to}`)).join(', ')))
        : el('p', { class: 'tr-facts' },
          val != null ? el('strong', null, `${metricName(m.metric)} ${nf3.format(val)}`) : null,
          test != null ? el('span', null, ` · ${t('md.test_value', { v: nf3.format(test) })}`) : null,
          el('span', { class: 'muted' }, ` · ${fmtBytes(m.size || 0)}`)),
      el('p', { class: 'md-note' },
        m.note ? el('span', null, m.note) : el('span', { class: 'muted' }, t('md.no_note')),
        el('button', { type: 'button', class: 'icon-btn', 'data-act': 'note', 'aria-label': t('md.edit_note_aria', { n: m.number }), title: t('md.edit_note'), onclick: () => editNote(m) }, icon('edit', 16))),
      el('div', { class: 'tr-actions' },
        busy(m) ? null : el('button', { type: 'button', class: 'btn small', 'data-act': 'preannotate', onclick: () => { showTab('annotations'); ctx.panels.annotations.preannotateWith(m.number); } }, icon('star', 14), t('md.preannotate')),
        isPublic(m) ? el('span', { class: 'muted md-export' }, t('md.public_weights')) : null,
        isPublic(m) ? null : el('a', { class: 'btn small', href: download(m, 'pytorch'), download: '', 'data-act': 'pt' }, icon('download', 14), t('md.download_pt')),
        isPublic(m) ? null : onnxBlock(m),
        busy(m) ? null : el('button', { type: 'button', class: 'icon-btn danger tr-del', 'data-act': 'delete', 'aria-label': t('md.delete_aria', { n: m.number }), title: t('md.delete'), onclick: () => removeModel(m) }, icon('trash', 18))));
  }

  /* the models chosen for the comparison (#214): only trained ones have metrics to compare */
  const chosen = new Set();
  const comparable = () => models.filter((m) => !isPublic(m));
  function choose(numbers, on) {
    numbers.forEach((n) => (on ? chosen.add(n) : chosen.delete(n)));
    sig = '';
    draw();
  }
  function compareBlock() {
    const cols = comparable().filter((m) => chosen.has(m.number)).sort((a, b) => a.number - b.number);
    if (cols.length < 2) return null;
    return compareTable({
      id: 'md-compare', title: t('md.compare'), help: t('md.compare_help'), cols, resultsOf: (m) => m.results, sizeOf: (m) => m.size,
      head: (m) => [`m${m.number}`, m.current ? el('span', { class: 'md-star', title: t('md.current') }, icon('star', 12)) : null, el('span', { class: 'md-arch' }, archName(m))],
    });
  }

  /* a redraw keeps the keyboard focus on the same button of the same model */
  function refocus(redraw) {
    const a = document.activeElement;
    const act = a && box.contains(a) && a.dataset ? a.dataset.act : null;
    const owner = act ? a.closest('li[id]') : null;
    const ownerId = owner ? owner.id : null;
    redraw();
    if (!act) return;
    const scope = (ownerId && document.getElementById(ownerId)) || box;
    const again = scope.querySelector(`[data-act="${act}"]`) || scope.querySelector('[data-act]');
    if (again) again.focus();
  }

  async function addPublic(trigger) {
    let data;
    try { data = await api('GET', `/projects/${projectId}/models/public-catalog`); } catch (err) { toastError(err); return; }
    if (!(data.models || []).length) { toast(t('md.public_none'), 'info', 5000); return; }
    const st = { id: data.models[0].id, map: {} };
    const modelOf = () => data.models.find((x) => x.id === st.id);
    const choices = el('div', { class: 'md-public-list', role: 'radiogroup', 'aria-label': t('md.public_model') });
    const mapBox = el('div', { class: 'md-map' });
    const note = el('textarea', { id: 'md-public-note', rows: '2', maxlength: '500' });
    const save = el('button', { type: 'submit', class: 'btn primary' }, t('md.public_add'));
    const err = el('p', { class: 'tr-form-error', role: 'alert', hidden: true });
    /* st.map: project class -> the public categories chosen for it (several may name one class). Returns the first category chosen for two project classes ({category, classes}), or null: a category goes to one class only */
    function clash() {
      const by = {};
      for (const [cls, froms] of Object.entries(st.map)) for (const from of froms) (by[from] = by[from] || []).push(cls);
      const found = Object.entries(by).find(([, classes]) => classes.length > 1);
      return found ? { category: found[0], classes: found[1] } : null;
    }
    function showClash() {
      const c = clash();
      err.textContent = c ? t('md.mapping_duplicate', { category: c.category, classes: c.classes.join(', ') }) : '';
      err.hidden = !c;
      return !!c;
    }
    function suggested() {
      /* project class -> the public categories suggested for it (same name) */
      const out = {};
      for (const [from, to] of Object.entries(modelOf().suggested || {})) (out[to] = out[to] || []).push(from);
      return out;
    }
    function drawChoices() {
      fill(choices, ...data.models.map((m) => {
        const id = `md-pub-${m.id}`;
        const input = el('input', { type: 'radio', name: 'md-pub', id, value: m.id, checked: m.id === st.id, onchange: () => { st.id = m.id; st.map = suggested(); drawMap(); showClash(); } });
        return el('label', { class: `md-public-choice${m.id === st.id ? ' selected' : ''}`, for: id }, input,
          el('span', { class: 'md-public-text' }, el('strong', null, m.label),
            el('span', { class: 'muted' }, `${tr(`md.pub.${m.id}`, m.note)} · ${t('md.public_stats', { map: nf3.format(m.box_map), mb: num(m.size_mb) })}`)));
      }));
    }
    let mapOpen = false;
    const linked = () => data.project_classes.filter((cls) => (st.map[cls] || []).length).length;
    const mapLine = () => t('md.mapping_summary', { n: num(linked()), total: num(data.project_classes.length) });
    function drawMap() {
      drawChoices();
      const m = modelOf();
      const line = el('span', null, mapLine());
      const rows = el('div', { class: 'md-map-rows' }, ...data.project_classes.map((cls) => {
          const id = `md-map-${cls}`;
          const sel = el('select', { id, class: 'tr-select', multiple: true, size: '4', onchange: (e) => { st.map[cls] = [...e.target.selectedOptions].map((o) => o.value); line.textContent = mapLine(); showClash(); } },
            ...m.classes.map((c) => el('option', { value: c, selected: (st.map[cls] || []).includes(c) }, c)));
          return el('div', { class: 'md-map-row' }, el('label', { for: id }, cls), sel);
        }));
      const box = el('details', { class: 'md-map', open: mapOpen || !linked() },
        el('summary', null, line, el('span', { class: 'md-map-review' }, t('md.mapping_review'))), rows);
      box.addEventListener('toggle', () => { mapOpen = box.open; });
      fill(mapBox, el('p', { class: 'hint' }, t('md.mapping_help')), box);
    }
    st.map = suggested();
    drawMap();
    const dlg = el('dialog', { class: 'dialog wide', 'aria-labelledby': 'md-pub-t' },
      el('form', { class: 'dialog-body form', method: 'dialog' },
        el('h2', { id: 'md-pub-t' }, t('md.public_title')),
        el('p', { class: 'muted' }, t('md.public_intro')),
        el('fieldset', { class: 'field' }, el('legend', null, t('md.public_model')), choices),
        el('fieldset', { class: 'field' }, el('legend', null, t('md.mapping')), mapBox),
        el('div', { class: 'field' }, el('label', { for: 'md-public-note' }, t('md.note'), el('span', { class: 'optional' }, ` · ${t('step.optional')}`)), note),
        el('p', { class: 'hint' }, t('md.public_license')),
        err,
        el('div', { class: 'form-actions' }, save, el('button', { type: 'button', class: 'btn', onclick: () => dlg.close() }, t('common.cancel')))));
    dlg.querySelector('form').addEventListener('submit', async (e) => {
      e.preventDefault();
      if (showClash()) return;
      const mapping = {};
      for (const [cls, froms] of Object.entries(st.map)) for (const from of froms) mapping[from] = cls;
      if (!Object.keys(mapping).length) { err.textContent = t('md.mapping_empty'); err.hidden = false; return; }
      save.disabled = true;
      try {
        const m = await api('POST', `/projects/${projectId}/models/public`, { json: { public: st.id, mapping, note: note.value } });
        dlg.close();
        toast(t('md.public_added', { n: m.number }), 'success', 4000);
        await load(true);
      } catch (e2) { save.disabled = false; err.textContent = e2.message; err.hidden = false; }
    });
    dlg.addEventListener('close', () => { dlg.remove(); if (trigger) trigger.focus(); });
    document.body.appendChild(dlg);
    openModal(dlg);
  }

  function draw() {
    const s = JSON.stringify(models) + publicAvailable + [...chosen].join(',');
    if (s === sig) return;
    sig = s;
    publicBtn.hidden = publicAvailable === false;  // no public model for this type: nothing to add
    if (!models.length) {
      mdActions.hidden = true;  // the two ways to get a model are offered in the empty state instead
      fill(box, emptyState('star', publicAvailable === false ? t('md.empty_no_public') : t('md.empty'),
        publicAvailable === false ? null : el('button', { type: 'button', class: 'btn primary', onclick: (e) => addPublic(e.currentTarget) }, t('md.public_add_button')),
        el('a', { class: publicAvailable === false ? 'btn primary' : 'btn', href: '#training' }, t('md.go_training'))));
      return;
    }
    mdActions.hidden = false;
    refocus(() => fill(box, ...[
      comparable().length >= 2
        ? compareBar({ prefix: 'md-cmp', chosen: comparable().filter((m) => chosen.has(m.number)).length, total: comparable().length,
          onAll: (on) => choose(comparable().map((m) => m.number), on), onClear: () => choose([...chosen], false) })
        : null,
      el('ul', { class: 'tr-cards' }, ...models.map(card)),
      compareBlock(),
      models.some((m) => !isPublic(m)) ? el('p', { class: 'muted md-help' }, icon('info', 14), t('md.export_help')) : null,
    ].filter(Boolean)));
  }

  async function load(quiet = false) {
    clearTimeout(timer);
    try {
      await ensureLabels();
      await ensurePublic();
      const d = await api('GET', `/projects/${projectId}/models`);
      models = d.models || [];
      ctx.modelCount = models.length;
      ctx.currentModel = d.current;
      renderStepper();
      draw();
    } catch (err) { if (!quiet) toastError(err); }
    if (!node.hidden && models.some(busy)) timer = setTimeout(() => load(true), 2000);
  }

  const publicBtn = el('button', { type: 'button', class: 'btn', onclick: (e) => addPublic(e.currentTarget) }, t('md.public_add_button'));
  const mdActions = el('div', { class: 'md-head-actions' },
    publicBtn,
    el('a', { class: 'btn', href: '#training' }, t('md.go_training')));
  add(node,
    el('div', { class: 'ds-head' },
      el('div', null, el('h2', { id: 'h-models' }, t('step.models')), el('p', { class: 'muted' }, t('md.intro'))),
      mdActions),
    announcer,
    box);
  panel.mount = () => load();
  panel.show = () => load(true);
  /** The project changed (its task type may have): model names come from the new catalogue. */
  panel.refresh = () => {
    for (const id of Object.keys(labels)) delete labels[id];
    publicAvailable = null;
    if (panel.mounted) load(true);
  };
  panel.count = async () => {
    try {
      const d = await api('GET', `/projects/${projectId}/models`);
      ctx.modelCount = (d.models || []).length;
      ctx.currentModel = d.current;
      renderStepper();
    } catch (err) { /* the status stays "no models" */ }
  };
  return panel;
}
