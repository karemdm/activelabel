/* The page that imports a whole project from CVAT: choose the server and project, check, then import. */
import { taskBadge, taskPicker } from '../cv/names.js';
import { jobMessage, openToolSetup, statusText, toolName } from '../integrations/tools.js';
import { api, jobFailure } from '../shell/api.js';
import { el, fill, icon } from '../shell/dom.js';
import { fmtDate, num, t, tn, tr } from '../shell/i18n.js';
import { toastError } from '../shell/toast.js';
import { classList, problemRow, progressBar } from '../shell/widgets.js';

export function initImport() {
  const root = document.getElementById('import-root');
  const SOURCES = ['cvat'];
  const state = { tools: {}, tool: null, items: null, error: null, search: '', selected: null, job: null, timer: null };
  const ready = (tool) => !!(state.tools[tool] && state.tools[tool].configured);

  const sourceBox = el('section', { class: 'card import-source', 'aria-labelledby': 'h-source' });
  const listBox = el('section', { class: 'import-main', 'aria-live': 'polite' });
  const jobBox = el('section', { class: 'card import-job', 'aria-live': 'polite', hidden: true });
  fill(root, sourceBox, listBox, jobBox);

  async function loadTools() {
    try {
      const list = await api('GET', '/annotation-tools');
      state.tools = {};
      (Array.isArray(list) ? list : []).forEach((x) => { state.tools[x.tool] = x; });
    } catch (err) { toastError(err); }
    drawSource();
  }

  function drawSource() {
    const focused = document.activeElement && document.activeElement.name === 'import-tool' ? document.activeElement.value : null;
    fill(sourceBox,
      el('h2', { id: 'h-source' }, t('import.step_source')),
      el('div', { class: 'task-cards tool-cards', role: 'radiogroup', 'aria-labelledby': 'h-source' }, ...SOURCES.map((tool) => {
        const id = `import-tool-${tool}`;
        const ok = ready(tool);
        const input = el('input', { type: 'radio', name: 'import-tool', id, value: tool, class: 'sr-only' });
        input.checked = tool === state.tool;
        input.addEventListener('change', () => choose(tool));
        return el('div', { class: `tool-card${tool === state.tool ? ' selected' : ''}`, 'data-tool': tool },
          el('label', { for: id, class: 'tool-card-main' },
            input, icon(tool, 28),
            el('span', { class: 'task-card-title' }, toolName(tool)),
            el('span', { class: 'task-card-help' }, t(`import.tool.${tool}.help`))),
          el('div', { class: 'tool-card-foot' },
            el('span', { class: `ready-badge ${ok ? 'ok' : 'todo'}` }, icon(ok ? 'check' : 'info', 14), ok ? t('annotate.ready') : t('annotate.not_ready')),
            el('button', { type: 'button', class: 'btn small', 'aria-label': t('annotate.setup_for', { tool: toolName(tool) }), onclick: () => setup(tool) },
              ok ? t('annotate.settings') : t('annotate.setup'))));
      })));
    if (focused) { const f = sourceBox.querySelector(`input[value="${focused}"]`); if (f) f.focus(); }
  }

  function setup(tool) {
    openToolSetup(tool, state.tools[tool], async () => {
      await loadTools();
      if (state.tool === tool) await choose(tool);
    });
  }

  async function choose(tool) {
    state.tool = tool;
    drawSource();
    await loadItems();
  }

  async function loadItems() {
    const tool = state.tool;
    state.items = null;
    state.error = null;
    state.selected = null;
    state.search = '';
    if (!ready(tool)) {
      state.error = { status: 400 };
      drawMain();
      return;
    }
    drawMain();
    try {
      const data = await api('GET', `/import/${encodeURIComponent(tool)}/projects`);
      if (tool !== state.tool) return;
      state.items = (data && data.items) || [];
    } catch (err) {
      if (tool !== state.tool) return;
      state.error = err;
    }
    drawMain();
  }

  // ---- list + side panel
  const visible = () => {
    const q = state.search.trim().toLowerCase();
    return (state.items || []).filter((i) => !q || String(i.name).toLowerCase().includes(q));
  };
  const itemKey = (i) => `${i.kind || 'project'}:${i.id}`;

  function classCell(classes) {
    const list = classes || [];
    if (!list.length) return el('span', { class: 'muted' }, '—');
    return el('span', { class: 'class-cell' },
      ...list.slice(0, 3).map((c) => el('span', { class: 'chip static' }, c)),
      list.length > 3 ? el('span', { class: 'muted' }, t('import.more_classes', { n: num(list.length - 3) })) : null);
  }

  function drawMain() {
    if (!state.tool) { fill(listBox); return; }
    const name = toolName(state.tool);
    if (state.error) {
      const notSet = state.error.status === 400;
      fill(listBox, el('div', { class: 'card empty-note import-error', role: 'alert' }, icon('warning'),
        el('div', null,
          el('p', null, notSet ? t('import.not_configured', { tool: name }) : state.error.message),
          notSet && ready(state.tool) && state.error.message ? el('p', { class: 'muted' }, state.error.message) : null,
          ready(state.tool) ? el('p', { class: 'form-actions' }, el('button', { type: 'button', class: 'btn', onclick: loadItems }, t('import.retry'))) : null)));   // the card above has the one Configurar
      return;
    }
    if (state.items === null) {
      fill(listBox, el('div', { class: 'card' }, el('p', { class: 'muted', role: 'status' }, t('import.loading', { tool: name }))));
      return;
    }
    const search = el('input', { type: 'search', id: 'import-search', autocomplete: 'off', value: state.search, 'aria-label': t('import.search_label'), placeholder: t('import.search_label') });
    const tableBox = el('div', { class: 'import-table-wrap' });
    const hasKind = state.tool === 'cvat' && state.items.some((i) => i.kind);
    const table = (rows) => {
      const sel = state.selected && itemKey(state.selected);
      return el('table', { class: 'import-table' },
        el('caption', { class: 'sr-only' }, t('import.list_label', { tool: name })),
        el('thead', null, el('tr', null,
          el('th', { scope: 'col' }, t('field.name')),
          hasKind ? el('th', { scope: 'col' }, t('import.col_kind')) : null,
          el('th', { scope: 'col', class: 'num' }, t('project.images')),
          el('th', { scope: 'col' }, t('field.classes')),
          el('th', { scope: 'col' }, t('project.created')))),
        el('tbody', null, ...rows.map((i) => {
          const id = `import-row-${itemKey(i).replace(/\W/g, '_')}`;
          const input = el('input', { type: 'radio', name: 'import-item', id, value: itemKey(i) });
          input.checked = itemKey(i) === sel;
          input.addEventListener('change', () => select(i));
          return el('tr', { class: itemKey(i) === sel ? 'selected' : '', onclick: (e) => { if (e.target !== input) { input.checked = true; select(i); } } },
            el('td', { class: 'name', 'data-label': t('field.name') }, el('label', { for: id }, input, el('span', null, i.name))),
            hasKind ? el('td', { 'data-label': t('import.col_kind') }, i.kind ? el('span', { class: 'badge' }, tr(`import.kind.${i.kind}`, i.kind)) : null) : null,
            el('td', { class: 'num', 'data-label': t('project.images') }, num(i.images)),
            el('td', { 'data-label': t('field.classes') }, classCell(i.classes)),
            el('td', { 'data-label': t('project.created') }, i.created ? fmtDate(i.created) : '—'));
        })));
    };
    const draw = () => {
      const rows = visible();
      fill(tableBox, rows.length ? table(rows) : el('div', { class: 'empty-note' }, icon('info'),
        el('p', null, state.items.length ? t('import.empty_search') : t('import.empty', { tool: name }))));
    };
    search.addEventListener('input', () => { state.search = search.value; draw(); });
    draw();
    const listCard = el('div', { class: 'card import-list' },
      el('div', { class: 'import-list-head' }, el('h2', null, t('import.step_project')), search), tableBox);
    const side = el('aside', { class: 'card import-side', id: 'import-side', 'aria-label': t('import.panel_title') });
    fill(listBox, el('div', { class: 'import-grid' }, listCard, side));
    drawSide(side);
  }

  function select(item) {
    state.selected = item;
    listBox.querySelectorAll('.import-table tbody tr').forEach((r) => {
      const input = r.querySelector('input');
      r.classList.toggle('selected', !!input && input.value === itemKey(item));
    });
    const side = listBox.querySelector('#import-side');
    drawSide(side);
    if (window.matchMedia('(max-width: 1000px)').matches && side) side.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }

  function drawSide(side) {
    const item = state.selected;
    if (!item) {
      fill(side, el('h2', null, t('import.panel_title')), el('div', { class: 'empty-note' }, icon('info'), el('p', null, t('import.select_hint'))));
      return;
    }
    const tool = toolName(state.tool);
    const nameInput = el('input', { id: 'import-name', type: 'text', maxlength: '100', autocomplete: 'off', value: item.name });
    const descInput = el('textarea', { id: 'import-desc', rows: '2', maxlength: '2000', placeholder: t('import.description_ph', { tool, name: item.name }) });
    const needTask = !(item.classes || []).length;
    const picker = needTask ? taskPicker('import-task', 'detection') : null;
    const go = el('button', { type: 'submit', class: 'btn primary big' }, t('import.start'));
    const form = el('form', { class: 'form', novalidate: true },
      el('div', { class: 'field' }, el('label', { for: 'import-name' }, t('field.name')), nameInput),
      el('div', { class: 'field' }, el('label', { for: 'import-desc' }, t('field.description')), descInput),
      el('p', { class: 'callout-line' }, icon('info'), el('span', null, t('import.summary', { tool }))),
      picker ? el('fieldset', { class: 'field' }, el('legend', null, t('import.task_title')), el('p', { class: 'hint' }, t('import.task_help')), picker.node) : null,
      el('div', { class: 'form-actions' }, go));
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const name = nameInput.value.trim();
      if (!name) { toastError(new Error(t('new.name_required'))); nameInput.focus(); return; }
      go.disabled = true;
      const json = { tool: state.tool, remote_id: item.id, kind: item.kind || 'project', name };
      /* an empty description becomes the (translated) example shown in the field */
      json.description = descInput.value.trim() || descInput.placeholder;
      if (picker) json.task = picker.get();
      try {
        const job = await api('POST', '/import', { json });
        startJob(job, name);
      } catch (err) {
        go.disabled = false;
        toastError(err);
      }
    });
    fill(side, el('h2', null, t('import.panel_title')), el('p', { class: 'import-picked' }, icon(state.tool, 20), el('strong', null, item.name)), form);
  }

  // ---- job

  function showJobView(on) {
    sourceBox.hidden = on;
    listBox.hidden = on;
    jobBox.hidden = !on;
  }

  function startJob(job, name) {
    state.job = { id: job.id, name };
    showJobView(true);
    drawJob(job);
    clearTimeout(state.timer);
    state.timer = setTimeout(pollJob, 2000);
  }

  async function pollJob() {
    let job;
    try {
      job = await api('GET', `/jobs/${encodeURIComponent(state.job.id)}`);
    } catch (err) {
      if (err.status === 404) { drawFailed(err.message); return; }
      toastError(err);
      state.timer = setTimeout(pollJob, 4000);
      return;
    }
    drawJob(job);
    if (job.status === 'queued' || job.status === 'running') state.timer = setTimeout(pollJob, 2000);
  }

  async function cancel(button) {
    button.disabled = true;
    button.textContent = t('import.cancelling');
    try {
      await api('POST', `/jobs/${encodeURIComponent(state.job.id)}/cancel`);
    } catch (err) {
      toastError(err);
      button.disabled = false;
      button.textContent = t('jobs.stop');
    }
  }

  function backToList() {
    clearTimeout(state.timer);
    state.job = null;
    showJobView(false);
    loadItems();
  }

  function drawFailed(message) {
    fill(jobBox,
      el('h2', { class: 'bad-title' }, icon('error', 22), t('import.failed_title')),
      el('p', { class: 'job-error', role: 'alert' }, message || ''),
      el('div', { class: 'form-actions' }, el('button', { type: 'button', class: 'btn primary', onclick: backToList }, t('import.back'))));
  }

  function drawJob(job) {
    const name = state.job ? state.job.name : '';
    if (job.status === 'queued' || job.status === 'running') {
      const cancelBtn = el('button', { type: 'button', class: 'btn danger-outline' }, t('jobs.stop'));
      cancelBtn.addEventListener('click', () => cancel(cancelBtn));
      fill(jobBox,
        el('h2', null, t('import.progress_title', { name })),
        el('p', { class: 'muted' }, t('import.progress_help')),
        el('div', { class: 'job-progress' }, progressBar(job.done, job.total, t('import.progress_title', { name })),
          el('span', { class: 'muted' }, job.total ? `${num(job.done)} / ${num(job.total)}` : '')),
        el('p', { class: 'job-msg', role: 'status' }, job.message ? jobMessage(job) : statusText(job.status)),
        el('div', { class: 'form-actions' }, cancelBtn));
      return;
    }
    if (job.status === 'done') { drawResult(job.result || {}); return; }
    if (job.status === 'cancelled') {
      fill(jobBox, el('p', { class: 'empty-note' }, icon('info'), t('import.cancelled')),
        el('div', { class: 'form-actions' }, el('button', { type: 'button', class: 'btn primary', onclick: backToList }, t('import.back'))));
      return;
    }
    drawFailed(jobFailure(job) || statusText('failed'));
  }

  function drawResult(r) {
    const classes = Array.isArray(r.classes) ? r.classes.length : Number(r.classes) || 0;
    const fact = (label, value) => el('div', null, el('dt', null, label), el('dd', null, value));
    const problems = r.problems || [];
    fill(jobBox, ...[
      el('h2', { class: 'ok-title' }, icon('check', 22), t('import.done_title')),
      el('p', null, el('strong', null, r.name || (state.job && state.job.name) || '')),
      el('dl', { class: 'facts result-facts' },
        fact(t('import.detected_task'), r.task ? taskBadge(r.task) : '—'),
        fact(t('project.images'), [tn('images.added', r.images_added || 0), r.images_rejected ? ` · ${tn('images.rejected', r.images_rejected)}` : '']),
        fact(t('project.annotations'), tn('import.labels', r.labels || 0)),
        classes ? fact(t('field.classes'), tn('import.classes', classes)) : null),
      Array.isArray(r.classes) && r.classes.length ? classList(r.classes) : null,
      r.linked === undefined ? null : el('p', { class: `report-line ${r.linked ? 'ok' : 'warn'}` }, icon(r.linked ? 'check' : 'warning'), t(r.linked ? 'import.linked' : 'import.not_linked')),   // what the link allows right away (docs/SYNC.md)
      problems.length ? el('div', { class: 'problems-box' }, el('h3', null, t('import.problems')), el('ul', { class: 'problems plain' }, ...problems.map(problemRow))) : null,
      el('div', { class: 'form-actions' },
        r.project_id ? el('a', { class: 'btn primary big', href: `/projects/${encodeURIComponent(r.project_id)}` }, t('import.open')) : null,
        el('button', { type: 'button', class: 'btn', onclick: backToList }, t('import.another'))),
    ].filter(Boolean));
  }

  drawSource();
  loadTools().then(() => choose('cvat'));
}
