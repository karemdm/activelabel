/* Step 1, Images: uploading images and folders, the gallery, the tasks and jobs of the project (with the
 * new-task dialog) and the videos. */
import { imageGrid, openLightbox } from './grid.js';
import { CHUNK, SKIP_FILES, nextTaskNumber, partName } from './names.js';
import { CVAT_FIXABLE, cvatTarget } from '../integrations/open.js';
import { jobMessage, toolName } from '../integrations/tools.js';
import { api, errorMessage } from '../shell/api.js';
import { fiftyoneError, fiftyoneMissing, sayFiftyoneMissing } from '../shell/features.js';
import { confirmDialog, menuButton } from '../shell/dialogs.js';
import { add, el, fill, icon } from '../shell/dom.js';
import { fmtDateTime, nf1, num, t, tn, tr } from '../shell/i18n.js';
import { dropTab, reserveTab } from '../shell/newtab.js';
import { toast, toastError } from '../shell/toast.js';
import { progressBar } from '../shell/widgets.js';

export function imagesPanel(page) {
  const { ctx, hasImages, jobStarted, projectId, refreshCounts, refreshProject, showTab, tabsView } = page;
  const node = el('section', { class: 'panel', id: 'panel-images', 'aria-labelledby': 'h-images' });
  const panel = { node, mounted: false };
  let uploading = false;
  let controller = null;

  const fileInput = el('input', { type: 'file', multiple: true, class: 'sr-only', id: 'file-input', accept: 'image/*,.zip,application/zip', tabindex: '-1' });
  const folderInput = el('input', { type: 'file', multiple: true, class: 'sr-only', id: 'folder-input', tabindex: '-1' });
  folderInput.setAttribute('webkitdirectory', '');
  folderInput.setAttribute('directory', '');
  const progressBox = el('div', { class: 'upload-progress', hidden: true });
  const reportBox = el('div', { class: 'report', 'aria-live': 'polite' });
  let staged = null; // files dropped on the empty state, waiting for "Create task"
  const usable = (entries) => entries.filter((e) => !SKIP_FILES.has(e.name.split('/').pop().toLowerCase()));

  const drop = el('div', { class: 'dropzone', id: 'dropzone' },
    icon('upload', 36),
    el('p', { class: 'drop-title' }, t('images.drop_title')),
    el('p', { class: 'muted' }, t('images.drop_help')),
    el('div', { class: 'drop-actions' },
      el('button', { type: 'button', class: 'btn primary', onclick: () => fileInput.click() }, t('images.choose_files')),
      el('button', { type: 'button', class: 'btn', onclick: () => folderInput.click() }, t('images.choose_folder'))));

  /** Makes `zone` accept dropped files and folders; calls onEntries(entries). */
  function wireDrop(zone, onEntries) {
    ['dragenter', 'dragover'].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add('over'); }));
    ['dragleave', 'drop'].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove('over'); }));
    zone.addEventListener('drop', async (e) => onEntries(await filesFromDrop(e.dataTransfer)));
  }
  wireDrop(drop, (entries) => startUpload(entries));
  fileInput.addEventListener('change', () => { startUpload([...fileInput.files].map((f) => ({ file: f, name: f.webkitRelativePath || f.name }))); fileInput.value = ''; });
  folderInput.addEventListener('change', () => { startUpload([...folderInput.files].map((f) => ({ file: f, name: f.webkitRelativePath || f.name }))); folderInput.value = ''; });

  async function filesFromDrop(dt) {
    const out = [];
    const items = dt.items ? [...dt.items] : [];
    if (items.length && items[0].webkitGetAsEntry) {
      const roots = items.map((i) => i.webkitGetAsEntry()).filter(Boolean); // before any await
      const walk = async (entry, prefix) => {
        if (entry.isFile) {
          const file = await new Promise((res, rej) => entry.file(res, rej));
          out.push({ file, name: prefix + entry.name });
        } else if (entry.isDirectory) {
          const reader = entry.createReader();
          for (;;) {
            const batch = await new Promise((res, rej) => reader.readEntries(res, rej));
            if (!batch.length) break;
            for (const child of batch) await walk(child, `${prefix}${entry.name}/`);
          }
        }
      };
      for (const r of roots) await walk(r, '');
      return out;
    }
    return [...dt.files].map((f) => ({ file: f, name: f.name }));
  }

  const emptyReport = () => ({ added: 0, duplicates: [], duplicates_count: 0, rejected: [], rejected_count: 0, other_files: [], other_files_count: 0 });
  const mergeReport = (acc, r) => {
    acc.added += Number(r.added) || 0;
    for (const key of ['duplicates', 'rejected', 'other_files']) {
      const list = r[key] || [];
      acc[key].push(...list);
      acc[`${key}_count`] += Number(r[`${key}_count`]) || list.length;
    }
  };

  /** Codes of the server's answers that say the name of the new task is the problem. */
  const NAME_ERRORS = new Set(['task_name_empty', 'task_name_taken', 'task_name_too_long', 'control_characters']);

  async function startUpload(entries) {
    const files = usable(entries);
    if (!files.length || uploading) return;
    const jobSize = jobSizeInput.value.trim();
    if (jobSize && !/^[1-9]\d*$/.test(jobSize)) {
      jobSizeInput.setAttribute('aria-invalid', 'true');
      jobSizeInput.focus();
      toast(t('tasks.job_size_invalid'), 'error');
      return;
    }
    jobSizeInput.removeAttribute('aria-invalid');
    uploading = true;
    ctx.busy.add('upload');
    controller = new AbortController();
    fill(reportBox);
    if (!newDlg.open) newDlg.showModal();
    setStage('uploading');
    const merged = emptyReport();
    let sent = 0;
    let cancelled = false;
    let created = null; // the task made by the first chunk
    let failure = null; // why the upload stopped (a rejected name, a server error)
    const draw = () => {
      progressBox.hidden = false;
      fill(progressBox,
        el('div', { class: 'upload-line' },
          el('strong', null, t('images.uploading', { done: num(sent), total: num(files.length) })),
          el('button', { type: 'button', class: 'btn small', id: 'upload-cancel', onclick: () => controller.abort() }, t('common.cancel'))),
        progressBar(sent, files.length, t('images.upload_progress')));
    };
    draw();
    for (let i = 0; i < files.length; i += CHUNK) {
      const chunk = files.slice(i, i + CHUNK);
      const form = new FormData();
      chunk.forEach((e) => form.append('files', e.file, e.name));
      try {
        if (!created) {
          form.append('name', nameInput.value.trim());
          form.append('job_size', jobSize);
          form.append('open', 'true');
          const res = await api('POST', `/projects/${projectId}/tasks`, { body: form, signal: controller.signal });
          created = res.task;
          mergeReport(merged, res.report || {});
        } else {
          mergeReport(merged, await api('POST', `/projects/${projectId}/tasks/${created.id}/images`, { body: form, signal: controller.signal }));
        }
      } catch (err) {
        if (err.name === 'AbortError') cancelled = true; else failure = err;  // the reason is shown in the result card
        break;
      }
      sent += chunk.length;
      draw();
    }
    if (created) {
      // close the task (its images become fixed), also after a cancel or an error: best effort
      let removed = false;
      try {
        const closed = await api('PATCH', `/projects/${projectId}/tasks/${created.id}`, { json: { open: false } });
        removed = !!(closed && closed.removed); // an upload that added nothing leaves no task
      } catch (e) { /* the next task closes it */ }
      if (removed) {
        created = null;
      } else {
        nameInput.value = '';
        jobSizeInput.value = '';
        toast(t('tasks.created', { name: created.name }), 'success', 4000);
      }
    }
    uploading = false;
    ctx.busy.delete('upload');
    const badName = !created && failure && NAME_ERRORS.has(failure.code);   // nothing was sent: the files wait for another name
    staged = badName ? files : null;
    drawStaged();
    setStage(created ? 'done' : 'form'); // nothing was created: back to the form, the report says why
    if (cancelled) toast(t('images.cancelled', { done: num(sent) }), 'info');
    if (badName) {
      showNameError(failure.message);
      nameInput.focus();
    } else {
      renderReport(merged, sent < files.length, failure);
    }
    await refreshProject();
    await loadTasks();
    grid.reload();
    if (ctx.panels.curation.mounted) ctx.panels.curation.refresh();
  }

  /* the server's rejection reasons are short English phrases; "too many entries (more than N)" carries a number */
  const reasonText = (reason) => {
    const text = String(reason);
    const key = /^too many entries/i.test(text) ? 'too_many_entries' : text.toLowerCase().replace(/[^a-z]+/g, '_');
    return tr(`reason.${key}`, text);
  };
  const nameOf = (x) => (typeof x === 'string' ? x : (x && x.name) || '');

  function expandable(title, rows) {
    return el('details', { class: 'expand' }, el('summary', null, title), el('ul', { class: 'plain scroll-box' }, ...rows));
  }

  function renderReport(r, incomplete, failure = null) {
    const rows = [];
    if (r.added) rows.push(el('p', { class: 'report-line ok' }, icon('check'), tn('images.added', r.added)));
    else rows.push(el('p', { class: 'report-line bad' }, icon('warning'), t('images.none_added')));  // no green mark, no task
    if (failure) rows.push(el('p', { class: 'report-line bad', role: 'alert' }, icon('warning'), String(failure.message || failure)));
    if (r.duplicates_count) {
      rows.push(el('div', { class: 'report-line' }, icon('info'),
        el('div', null, tn('images.duplicates', r.duplicates_count), ': ', el('span', { class: 'muted' }, t('images.duplicates_note')),
          expandable(t('images.show_list'), r.duplicates.map((d) => el('li', null, nameOf(d)))))));
    }
    if (r.rejected_count) {
      rows.push(el('div', { class: 'report-line bad' }, icon('warning'),
        el('div', null, tn('images.rejected', r.rejected_count),
          expandable(t('images.show_list'), r.rejected.map((x) => el('li', null, el('span', null, nameOf(x)), ' ', el('span', { class: 'muted' }, `— ${reasonText(x.reason || '')}`)))))));
    }
    if (incomplete) rows.push(el('p', { class: 'muted' }, t('images.incomplete')));
    const annotationLike = (r.other_files || []).some((f) => /\.(json|txt|xml|ya?ml|csv)$/i.test(nameOf(f)));
    if (r.other_files_count && annotationLike) {
      rows.push(el('div', { class: 'callout' }, icon('info'),
        el('div', null,
          el('strong', null, t('images.annotations_found')),
          el('p', { class: 'muted' }, tn('images.annotations_found_n', r.other_files_count)),
          el('button', { type: 'button', class: 'btn primary', onclick: () => { newDlg.close(); showTab('annotations', true); } }, t('images.go_annotations')))));
    }
    fill(reportBox, el('div', { class: 'card report-card' }, el('h3', null, t('images.result')), ...rows));
  }

  // ---- tasks (cards), and the filter of the grid
  let tasks = [];
  let filter = { task: null, job: null, dataset: null, ids: null }; // dataset: {n, part|fold} set from step 4; ids: {list, model} the worst images of a model (step 6)
  let renaming = null; // id of the task being renamed
  let renameDraft = null; // text typed so far (kept if the cards are redrawn)
  const expanded = new Set();
  let focusAfter = null; // selector to focus once the cards are drawn

  const tasksBox = el('div', { class: 'tasks-wrap' });
  const filterBox = el('div', { class: 'task-filter' });
  const taskById = (id) => tasks.find((x) => x.id === id);

  const grid = imageGrid({
    projectId,
    query: () => ({
      status: 'active',
      task: filter.task,
      job: filter.job,
      dataset: filter.dataset,
      ids: filter.ids ? filter.ids.list : null,
      emptyText: filter.task != null || filter.dataset || filter.ids ? t('tasks.filter_empty') : t('grid.no_images'),
    }),
    onOpen: (items, i) => openLightbox(projectId, items, i, { edit: canEditInApp() ? editImage : null }),
    targets: () => (ctx.project && ctx.project.task === 'regression' ? ctx.project.classes : null),
    onValue: () => refreshCounts(),
  });
  const gridTitle = el('h3', { class: 'section-title', id: 'h-images-grid', tabindex: '-1' }, t('images.grid_title'));

  /** Reads the tasks; `refresh` makes the server ask CVAT for the job states again instead of reusing a read of the last minute. */
  async function loadTasks(refresh = false) {
    try {
      tasks = (await api('GET', `/projects/${projectId}/tasks${refresh ? '?refresh=true' : ''}`)).tasks || [];
    } catch (err) {
      toastError(err);
      return;
    }
    const current = taskById(filter.task);
    if (!current) filter = { task: null, job: null, dataset: filter.dataset, ids: filter.ids };
    else if (filter.job != null && !current.jobs.some((j) => j.n === filter.job)) filter.job = null;
    if (!tasks.some((x) => x.id === renaming)) renaming = null;
    nameInput.placeholder = t('tasks.name_placeholder', { id: nextTaskNumber(tasks) });
    drawTasks();
    drawFilter();
    imageTabs.setCount('tasks', tasks.length);
  }

  function setFilter(task, job, { reveal = false, dataset = null } = {}) {
    filter = { task, job, dataset };
    drawFilter();
    grid.reload();
    if (reveal) {
      imageTabs.select('gallery');
      gridTitle.scrollIntoView({ behavior: 'smooth', block: 'start' });
      gridTitle.focus({ preventScroll: true });
    }
  }

  function drawFilter() {
    const task = taskById(filter.task);
    const focused = document.activeElement && document.activeElement.closest && document.activeElement.closest('.task-filter') ? document.activeElement.id : null;
    const taskSelect = el('select', {
      id: 'filter-task',
      onchange: (e) => setFilter(e.target.value === '' ? null : Number(e.target.value), null, { dataset: filter.dataset }),
    },
    el('option', { value: '' }, t('tasks.filter_all_tasks')),
    ...tasks.map((x) => el('option', { value: String(x.id), selected: x.id === filter.task }, x.name)));
    const jobSelect = task && task.jobs.length > 1
      ? el('select', {
        id: 'filter-job',
        onchange: (e) => setFilter(filter.task, e.target.value === '' ? null : Number(e.target.value), { dataset: filter.dataset }),
      },
      el('option', { value: '' }, t('tasks.filter_all_jobs')),
      ...task.jobs.map((j) => el('option', { value: String(j.n), selected: j.n === filter.job }, t('tasks.job_option', { n: j.n, first: num(j.first), last: num(j.last) }))))
      : null;
    const chipText = task
      ? (filter.job != null ? t('tasks.filter_chip_job', { name: task.name, n: filter.job }) : t('tasks.filter_chip', { name: task.name }))
      : null;
    fill(filterBox,
      el('div', { class: 'task-filter-row' },
        el('div', { class: 'field inline' }, el('label', { for: 'filter-task', class: 'sr-only' }, t('tasks.filter_task')), taskSelect),
        jobSelect ? el('div', { class: 'field inline' }, el('label', { for: 'filter-job', class: 'sr-only' }, t('tasks.filter_job')), jobSelect) : null,
        chipText
          ? el('span', { class: 'chip filter-chip', role: 'status' }, chipText,
            el('button', { type: 'button', class: 'chip-x', id: 'filter-clear', 'aria-label': t('tasks.filter_clear'), title: t('tasks.filter_clear'), onclick: () => { setFilter(null, null, { dataset: filter.dataset }); const s = document.getElementById('filter-task'); if (s) s.focus(); } }, icon('close', 14)))
          : null,
        filter.dataset
          ? el('span', { class: 'chip filter-chip', role: 'status', id: 'filter-dataset-chip' },
            t('ds.filter_chip', { n: filter.dataset.n, part: filter.dataset.fold != null ? t('ds.fold', { n: filter.dataset.fold + 1 }) : partName(filter.dataset.part) }),
            el('button', { type: 'button', class: 'chip-x', id: 'filter-dataset-clear', 'aria-label': t('ds.filter_clear'), title: t('ds.filter_clear'), onclick: () => { setFilter(filter.task, filter.job); const s = document.getElementById('filter-task'); if (s) s.focus(); } }, icon('close', 14)))
          : null,
        filter.ids
          ? el('span', { class: 'chip filter-chip', role: 'status', id: 'filter-ids-chip' },
            t('md.eval_worst_chip', { n: filter.ids.model, count: filter.ids.list.length }),
            el('button', { type: 'button', class: 'chip-x', id: 'filter-ids-clear', 'aria-label': t('md.eval_worst_clear'), title: t('md.eval_worst_clear'), onclick: () => { setFilter(filter.task, filter.job, { dataset: filter.dataset }); const s = document.getElementById('filter-task'); if (s) s.focus(); } }, icon('close', 14)))
          : null));
    if (focused) { const f = document.getElementById(focused); if (f) f.focus(); }
  }

  function taskCard(task) {
    const cid = `task-${task.id}`;
    const src = task.source && task.source.video ? task.source : null;
    const isRenaming = renaming === task.id;
    const filtered = filter.task === task.id;
    const manyJobs = task.jobs.length > 1;
    const open = manyJobs && expanded.has(task.id);
    const thinBar = (done, total, label) => { const b = progressBar(done, total, label); b.classList.add('thin'); return b; };

    let title;
    if (isRenaming) {
      const input = el('input', { type: 'text', id: `${cid}-name`, value: renameDraft == null ? task.name : renameDraft, 'aria-label': t('tasks.rename_label', { name: task.name }), autocomplete: 'off', maxlength: '100' });
      input.addEventListener('input', () => { renameDraft = input.value; });
      const cancel = () => { renaming = null; renameDraft = null; focusAfter = `#${cid}-rename`; drawTasks(); };
      const form = el('form', { class: 'task-rename', novalidate: true,
        onsubmit: async (e) => {
          e.preventDefault();
          const name = input.value.trim();
          if (!name) { toastError(new Error(t('error.task_name_empty'))); input.focus(); return; }
          if (name === task.name) { cancel(); return; }
          try {
            await api('PATCH', `/projects/${projectId}/tasks/${task.id}`, { json: { name } });
            toast(t('tasks.renamed'), 'success', 3000);
            renaming = null;
            renameDraft = null;
            focusAfter = `#${cid}-rename`;
            await loadTasks();
          } catch (err) { toastError(err); if (!(await reloadIfChanged(err))) input.focus(); }
        } },
      input,
      el('button', { type: 'submit', class: 'btn small primary' }, t('common.save')),
      el('button', { type: 'button', class: 'btn small', onclick: cancel }, t('common.cancel')));
      input.addEventListener('keydown', (e) => { if (e.key === 'Escape') { e.preventDefault(); cancel(); } });
      title = form;
      focusAfter = `#${cid}-name`;
    } else {
      title = el('h4', { class: 'task-name' },
        el('button', { type: 'button', class: 'task-title', title: t('tasks.see_images_task', { name: task.name }), 'aria-pressed': filtered ? 'true' : 'false', onclick: () => taskSelectSync(task.id, null) },
          src ? el('span', { class: 'task-video-icon', title: t('video.mode_video') }, icon('video', 16)) : null, task.name));
    }

    /* the open icon of a task or of one of its jobs: a menu with the tools it can be opened in */
    const openMenu = (job) => menuButton('external',
      job ? t('tasks.open_menu_job', { n: job, name: task.name }) : t('tasks.open_menu', { name: task.name }),
      () => openItems(task, job));
    const iconButton = (name, label, extra, onclick, id) => el('button', { type: 'button', class: `icon-btn small${extra ? ` ${extra}` : ''}`, id, 'aria-label': label, title: label, onclick }, icon(name, 16));
    let everyText = null;
    if (src && src.step) {
      everyText = src.fps
        ? t('video.meta_every_seconds', { s: nf1.format(Math.round((src.step / src.fps) * 100) / 100) })
        : tn('video.meta_every_frames', src.step);
    } else if (src && src.every_seconds) {
      everyText = t('video.meta_every_seconds', { s: nf1.format(src.every_seconds) });
    }
    const meta = [
      tn('tasks.meta_images', task.active),
      task.images > task.active ? tn('tasks.meta_excluded', task.images - task.active) : null,
      everyText,
      task.job_size ? t('tasks.meta_job_size', { n: num(task.job_size) }) : null,
      tn('tasks.meta_jobs', task.jobs.length),
    ].filter(Boolean).join(' · ');
    const progressText = t('tasks.progress', { done: num(task.annotated), total: num(task.active) });

    const jobsList = manyJobs
      ? el('ul', { class: 'plain task-jobs', id: `${cid}-jobs`, hidden: !open },
        ...task.jobs.map((j) => {
          const on = filtered && filter.job === j.n;
          /* state of the job in the linked CVAT project (new, in progress, completed, rejected) */
          const rj = task.remote && (task.remote.jobs || []).find((x) => x.n === j.n);
          const state = rj && rj.state ? String(rj.state).replace(/\s+/g, '_') : '';
          const badge = state
            ? el('span', { class: `job-state state-${state}` }, tr(`tasks.job_state.${state}`, rj.state))
            : null;
          /* who works on it and at which stage, as text under the row */
          const who = [rj && rj.stage ? tr(`tasks.job_stage.${rj.stage}`, rj.stage) : null,
            rj && rj.assignee ? t('tasks.job_assignee', { name: rj.assignee }) : null].filter(Boolean).join(' · ');
          return el('li', { class: 'job-li' },
            el('button', { type: 'button', class: `job-row${on ? ' is-filtered' : ''}${badge ? ' has-state' : ''}`, 'aria-pressed': on ? 'true' : 'false',
              'aria-label': t('tasks.see_images_job', { n: j.n, name: task.name }), onclick: () => taskSelectSync(task.id, j.n) },
            el('span', { class: 'job-name' }, t('tasks.job_name', { n: j.n })),
            el('span', { class: 'job-range muted' }, `${num(j.first)}–${num(j.last)}`),
            thinBar(j.annotated, j.active, t('tasks.progress', { done: num(j.annotated), total: num(j.active) })),
            el('span', { class: 'job-count' }, `${num(j.annotated)}/${num(j.active)}`),
            badge,
            j.images > j.active ? el('span', { class: 'job-excluded muted' }, tn('tasks.job_excluded', j.images - j.active)) : null,
            who ? el('span', { class: 'job-who muted small' }, who) : null),
            openMenu(j.n));
        }))
      : null;

    return el('li', { class: `task-item card${filtered ? ' is-filtered' : ''}`, id: cid, 'aria-label': task.name },
      el('div', { class: 'task-head' },
        title,
        isRenaming ? null : el('div', { class: 'task-actions' },
          openMenu(null),
          iconButton('edit', `${t('tasks.rename')}: ${task.name}`, '', () => { renaming = task.id; renameDraft = null; drawTasks(); }, `${cid}-rename`),
          iconButton('trash', `${t('tasks.delete')}: ${task.name}`, 'danger', () => deleteTask(task)),
          manyJobs
            ? el('button', { type: 'button', class: 'icon-btn small task-toggle', 'aria-expanded': open ? 'true' : 'false', 'aria-controls': `${cid}-jobs`,
              'aria-label': `${open ? t('tasks.hide_jobs') : t('tasks.show_jobs')}: ${task.name}`, title: open ? t('tasks.hide_jobs') : t('tasks.show_jobs'),
              onclick: () => {
                if (expanded.has(task.id)) expanded.delete(task.id); else expanded.add(task.id);
                focusAfter = `#${cid} .task-toggle`;
                drawTasks();
              } }, icon('chevron', 16))
            : null)),
      el('p', { class: 'task-meta muted' }, meta),
      /* the last edit session of the task: tool, when, and how it ended (#526) */
      task.last_session ? el('p', { class: 'task-last-session muted small' }, t('tasks.last_session', {
        tool: toolName(task.last_session.tool), when: fmtDateTime(task.last_session.at),
        status: tr(`annotate.status.${task.last_session.status}`, task.last_session.status).toLowerCase() })) : null,
      /* an upload cut off midway: the app closed the task, some images may be missing (#434) */
      task.incomplete ? el('p', { class: 'task-incomplete', title: t('tasks.incomplete_help') }, icon('error', 14), t('tasks.incomplete')) : null,
      el('div', { class: 'task-progress' },
        thinBar(task.annotated, task.active, progressText),
        el('span', { class: 'task-progress-text' }, progressText)),
      jobsList);
  }

  const taskSelectSync = (task, job) => setFilter(task, job, { reveal: true });

  /** The Update button: reads the tasks again, CVAT job states included, and keeps the focus on it. */
  async function refreshTasks() {
    focusAfter = '#tasks-refresh';
    await loadTasks(true);
    toast(t('tasks.refreshed'), 'success', 2500);   // a visible answer even when nothing changed
  }

  function drawTasks() {
    const head = el('div', { class: 'tasks-head' },
      el('h3', { class: 'tasks-title sr-only' }, t('tasks.title'), tasks.length ? el('span', { class: 'count-badge' }, num(tasks.length)) : null),   // the tab already names and counts them
      tasks.length ? el('div', { class: 'tasks-head-actions' },
        tasks.some((x) => x.remote)   // only tasks linked to CVAT have job states to read again
          ? el('button', { type: 'button', class: 'btn', id: 'tasks-refresh', title: t('tasks.refresh_help'), onclick: refreshTasks }, icon('refresh', 16), t('tasks.refresh'))
          : null,
        el('button', { type: 'button', class: 'btn primary', id: 'new-task-btn', onclick: () => openNewTask() }, icon('plus', 16), t('tasks.new_title'))) : null);
    let body;
    if (tasks.length) {
      body = el('ul', { class: 'plain task-list' }, ...tasks.map(taskCard));
    } else {
      body = el('div', { class: 'dropzone empty-tasks' },
        icon('images', 40),
        el('p', { class: 'drop-title' }, t('tasks.none')),
        el('button', { type: 'button', class: 'btn primary', id: 'new-task-btn', onclick: () => openNewTask() }, icon('plus', 16), t('tasks.new_title')));
      wireDrop(body, (entries) => openNewTask(entries));
    }
    fill(tasksBox, head, body);
    if (focusAfter) {
      const f = tasksBox.querySelector(focusAfter);
      focusAfter = null;
      if (f) f.focus();
    }
  }

  /** What a task (or one of its jobs) can be opened in. A disabled item says why. */
  function openItems(task, job) {
    const kind = ctx.project && ctx.project.task;
    const regression = kind === 'regression';   // numbers per image: FiftyOne to look at them, no CVAT (#619)
    const viewOnly = kind === 'keypoints' || regression;
    const link = (ctx.links || []).find((l) => l.tool === 'cvat') || null;
    /* a task always says its own reason (not the project's), linked or not */
    const target = cvatTarget({ link: link || {}, configured: true, task, job: job || null });
    const items = [viewOnly
      ? { label: t('tasks.menu_edit_fo'), disabled: true, reason: t(regression ? 'tasks.fo_no_regression' : 'tasks.fo_no_keypoints') }
      : { label: t('tasks.menu_edit_fo'), onclick: () => editInFiftyOne(task, job) }];
    if (viewOnly) items.push({ label: t('tasks.menu_view_fo'), onclick: () => openTaskInFiftyOne(task, job) });
    if (regression) return items;
    /* what a comparison can fix (not in CVAT yet, deleted there) leads to the CVAT tab with the reason; a
       connection not set up or CVAT down cannot be fixed there, so the item stays disabled with why (#684) */
    if (target.url) items.push({ label: t('tasks.menu_cvat'), href: target.url });
    else if (CVAT_FIXABLE.has(task.cvat)) {
      items.push({ label: t('tasks.menu_cvat_go'), reason: t(target.reason), onclick: () => goToCvat(task) });
    } else items.push({ label: t('tasks.menu_cvat'), disabled: true, reason: t(target.reason) });
    return items;
  }

  /** Annotations → CVAT with this task chosen: the comparison there creates it in CVAT, or decides what to do
   *  with what was deleted there. */
  function goToCvat(task) {
    showTab('annotations', true);
    const panel = ctx.panels && ctx.panels.annotations;
    if (panel && panel.focusCvatTask) panel.focusCvatTask(task.id);
    if (panel && panel.reveal) panel.reveal('sync');
  }

  /** The FiftyOne view of a task or job, to look at (a video task shows each video as an ordered sequence). */
  async function openTaskInFiftyOne(task, job) {
    if (sayFiftyoneMissing()) return;   // #675
    const tab = reserveTab(t('cur.fo_waiting'));
    try {
      const started = await api('POST', `/projects/${projectId}/fiftyone`, { json: job ? { task: task.id, job } : { task: task.id } });
      if (started && started.id) ctx.openWhenDone.set(started.id, tab); else dropTab(tab);
      toast(t('cur.fo_starting'), 'info', 4000);
      await jobStarted(started);
    } catch (err) { dropTab(tab); if (!fiftyoneError(err)) toastError(err); }
  }

  /** The App can edit this project's labels (not keypoints) and FiftyOne is installed. */
  function canEditInApp() {
    return !fiftyoneMissing() && !!ctx.project && ctx.project.task !== 'keypoints';
  }

  /** "Edit in FiftyOne" from the lightbox (#706): an edit session on this one image; the App opens on it. */
  async function editImage(img) {
    if (sayFiftyoneMissing()) return;
    const tab = reserveTab(t('cur.fo_waiting'));
    try {
      const started = await api('POST', `/projects/${projectId}/annotate`, { json: { tool: 'fiftyone', scope: 'selected', ids: [img.id] } });
      if (started && started.id) ctx.openWhenDone.set(started.id, tab); else dropTab(tab);
      toast(t('annotate.send_started', { tool: toolName('fiftyone') }), 'info', 4000);
      await jobStarted(started);
    } catch (err) { dropTab(tab); if (!fiftyoneError(err)) toastError(err); }
  }

  /** An edit session on exactly the active images of the task or job (its card is in Annotations). */
  async function editInFiftyOne(task, job) {
    if (sayFiftyoneMissing()) return;   // #675
    const tab = reserveTab(t('cur.fo_waiting'));
    try {
      const started = await api('POST', `/projects/${projectId}/annotate`, { json: { tool: 'fiftyone', scope: 'task', task: task.id, ...(job ? { job } : {}) } });
      if (started && started.id) ctx.openWhenDone.set(started.id, tab); else dropTab(tab);
      toast(t('annotate.send_started', { tool: toolName('fiftyone') }), 'info', 4000);
      await jobStarted(started);
    } catch (err) { dropTab(tab); if (!fiftyoneError(err)) toastError(err); }
  }

  /** A 404/409 means the task changed elsewhere (another tab): show the real state instead of a dead card. */
  async function reloadIfChanged(err) {
    if (err.status !== 404 && err.status !== 409) return false;
    renaming = null;
    renameDraft = null;
    await refreshProject();
    await loadTasks();
    grid.reload();
    return true;
  }

  async function deleteTask(task) {
    const ok = await confirmDialog({
      title: t('tasks.delete_title', { name: task.name }),
      /* a task that is also in CVAT stays there unless the comparison with CVAT is told to delete it too (#684) */
      text: [tn('tasks.delete_text', task.images), task.remote || task.cvat === 'ready' ? t('tasks.delete_text_cvat') : null].filter(Boolean).join(' '),
      confirmLabel: t('tasks.delete_confirm'),
    });
    if (!ok) return;
    try {
      const r = await api('DELETE', `/projects/${projectId}/tasks/${task.id}`, { json: { confirm: true } });
      toast(tn('tasks.deleted', Number(r && r.deleted) || 0), 'success', 5000);
    } catch (err) { toastError(err); await reloadIfChanged(err); return; }
    expanded.delete(task.id);
    await refreshProject();
    await loadTasks();
    grid.reload();
    if (ctx.panels.curation.mounted) ctx.panels.curation.refresh();
  }

  // ---- "new task" dialog (name, images per job, then the files; upload progress and report)
  const nameInput = el('input', { type: 'text', id: 'task-name', autocomplete: 'off', maxlength: '100' });  // the server's limit
  const jobSizeInput = el('input', { type: 'number', id: 'task-job-size', min: '1', step: '1', inputmode: 'numeric', 'aria-describedby': 'task-job-size-hint' });
  jobSizeInput.addEventListener('input', () => jobSizeInput.removeAttribute('aria-invalid'));
  /* a rejected name is said once, under the field (#342); typing a new name clears it */
  const nameError = el('p', { class: 'field-error', id: 'task-name-error', role: 'alert', hidden: true });
  const showNameError = (message) => {
    nameError.textContent = message || '';
    nameError.hidden = !message;
    if (message) { nameInput.setAttribute('aria-invalid', 'true'); nameInput.setAttribute('aria-describedby', nameError.id); } else { nameInput.removeAttribute('aria-invalid'); nameInput.removeAttribute('aria-describedby'); }
  };
  nameInput.addEventListener('input', () => showNameError(''));
  const stagedBox = el('div', { class: 'staged', hidden: true });
  const imagesBox = el('div', { class: 'nt-images' }, drop, stagedBox);
  const videoBox = el('div', { class: 'nt-video', hidden: true });
  const modeBtns = {};
  const modeTabs = el('div', { class: 'tabs nt-modes', role: 'group', 'aria-label': t('tasks.new_title') },
    ...['images', 'video'].map((m) => (modeBtns[m] = el('button', { type: 'button', class: `tab${m === 'images' ? ' active' : ''}`, 'aria-pressed': m === 'images' ? 'true' : 'false', id: `nt-mode-${m}`, onclick: () => setMode(m) }, t(`video.mode_${m}`)))));
  const fieldsBox = el('div', { class: 'new-task-fields' },
    el('div', { class: 'field' }, el('label', { for: 'task-name' }, t('tasks.name')), nameInput, nameError),
    el('div', { class: 'field' },
      el('label', { for: 'task-job-size' }, t('tasks.job_size')), jobSizeInput,
      el('p', { class: 'hint', id: 'task-job-size-hint' }, t('tasks.job_size_hint'))));
  const formBox = el('div', { class: 'nt-form' }, modeTabs, fieldsBox, imagesBox, videoBox);
  let mode = 'images';
  function setMode(m) {
    if (uploading || vid.busy) return;
    mode = m;
    for (const k of Object.keys(modeBtns)) {
      modeBtns[k].classList.toggle('active', k === m);
      modeBtns[k].setAttribute('aria-pressed', k === m ? 'true' : 'false');
    }
    imagesBox.hidden = m !== 'images';
    videoBox.hidden = m !== 'video';
  }

  // ---- video tab
  const POLL_LOST = 3;  // failed polls in a row before the person is told and can leave
  const vid = { busy: false, xhr: null, video: null, taskMade: false, plan: null, seq: 0, timer: null, unit: 'frames', polling: null, jobId: null, pollFails: 0 };
  const VIDEO_EXT = /\.(mp4|avi|mov|mkv|webm|m4v)$/i;
  const vFile = el('input', { type: 'file', class: 'sr-only', id: 'video-input', accept: '.mp4,.avi,.mov,.mkv,.webm,.m4v,video/*', tabindex: '-1' });
  const vDrop = el('div', { class: 'dropzone', id: 'video-dropzone' },
    icon('video', 36),
    el('p', { class: 'drop-title' }, t('video.drop_title')),
    el('p', { class: 'muted' }, t('video.drop_help')),
    el('div', { class: 'drop-actions' }, el('button', { type: 'button', class: 'btn primary', onclick: () => vFile.click() }, t('video.choose'))));
  const vUp = el('div', { class: 'upload-progress', hidden: true });
  const vSel = el('div', { class: 'video-sel', hidden: true });
  const vRun = el('div', { class: 'video-run', hidden: true, 'aria-live': 'polite' });
  add(videoBox, vFile, vDrop, vUp, vSel, vRun);
  wireDrop(vDrop, (entries) => pickVideo(entries.length ? entries[0].file : null));
  vFile.addEventListener('change', () => { pickVideo(vFile.files[0] || null); vFile.value = ''; });

  const pad2 = (n) => String(n).padStart(2, '0');
  const fmtClock = (sec) => {
    const s = Math.max(0, Math.round(Number(sec) || 0));
    const h = Math.floor(s / 3600);
    return h ? `${h}:${pad2(Math.floor((s % 3600) / 60))}:${pad2(s % 60)}` : `${pad2(Math.floor(s / 60))}:${pad2(s % 60)}`;
  };
  /** "120" -> frame 120; "1:30" or "01:02:03" -> frame at that time. null when invalid. */
  function parseFrame(text, fps) {
    const s = String(text).trim();
    if (/^\d+$/.test(s)) return Number(s);
    const m = /^(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)$/.exec(s);
    if (!m) return null;
    const sec = Number(m[1] || 0) * 3600 + Number(m[2]) * 60 + Number(m[3]);
    return Math.round(sec * fps);
  }
  const vStart = el('input', { type: 'text', id: 'video-start', autocomplete: 'off', inputmode: 'text', 'aria-describedby': 'video-start-alt' });
  const vStop = el('input', { type: 'text', id: 'video-stop', autocomplete: 'off', inputmode: 'text', 'aria-describedby': 'video-stop-alt' });
  const vEvery = el('input', { type: 'number', id: 'video-every', min: '0', step: 'any', inputmode: 'decimal', value: '1' });
  const vStartAlt = el('span', { class: 'muted alt', id: 'video-start-alt' });
  const vStopAlt = el('span', { class: 'muted alt', id: 'video-stop-alt' });
  const unitBtns = {};
  const unitSwitch = el('div', { class: 'tabs', role: 'group', 'aria-label': t('video.every') },
    ...['frames', 'seconds'].map((u) => (unitBtns[u] = el('button', { type: 'button', class: `tab${u === 'frames' ? ' active' : ''}`, 'aria-pressed': u === 'frames' ? 'true' : 'false', onclick: () => setUnit(u) }, t(`video.unit_${u}`)))));
  const vInfo = el('p', { class: 'video-info muted' });
  const vSummary = el('p', { class: 'video-summary', 'aria-live': 'polite' });
  const vStrip = el('ul', { class: 'plain video-strip' });
  const vCreate = el('button', { type: 'button', class: 'btn primary', id: 'video-create', onclick: () => createFromVideo() }, t('tasks.create'));
  const vAnother = el('button', { type: 'button', class: 'btn', onclick: () => discardVideo() }, t('video.change'));
  add(vSel,
    vInfo,
    el('div', { class: 'new-task-fields' },
      el('div', { class: 'field' }, el('label', { for: 'video-start' }, t('video.start')), vStart, vStartAlt),
      el('div', { class: 'field' }, el('label', { for: 'video-stop' }, t('video.end')), vStop, vStopAlt)),
    el('div', { class: 'field video-every' },
      el('label', { for: 'video-every' }, t('video.every')),
      el('div', { class: 'video-every-row' }, vEvery, unitSwitch)),
    vSummary, vStrip,
    el('div', { class: 'drop-actions' }, vCreate, vAnother));
  for (const i of [vStart, vStop, vEvery]) i.addEventListener('input', () => { i.removeAttribute('aria-invalid'); schedulePlan(); });

  function setUnit(u) {
    vid.unit = u;
    for (const k of Object.keys(unitBtns)) {
      unitBtns[k].classList.toggle('active', k === u);
      unitBtns[k].setAttribute('aria-pressed', k === u ? 'true' : 'false');
    }
    const v = vid.video;
    if (v && v.fps) {
      const cur = Number(vEvery.value);
      if (cur > 0) vEvery.value = String(u === 'seconds' ? Math.round((cur / v.fps) * 100) / 100 : Math.max(1, Math.round(cur * v.fps)));
    }
    schedulePlan();
  }

  /** Reads the form; returns the request body for the plan/create calls, or null when invalid. */
  function selection(mark) {
    const v = vid.video;
    if (!v) return null;
    const start = parseFrame(vStart.value, v.fps);
    const stop = parseFrame(vStop.value, v.fps);
    const every = Number(vEvery.value);
    const bad = [];
    if (start == null || start >= v.frames) bad.push(vStart);
    if (stop == null || stop >= v.frames || (start != null && stop < start)) bad.push(vStop);
    if (!(every > 0) || (vid.unit === 'frames' && !Number.isInteger(every))) bad.push(vEvery);
    if (mark) bad.forEach((i) => i.setAttribute('aria-invalid', 'true'));
    if (bad.length) return null;
    return vid.unit === 'seconds' ? { start, stop, every_seconds: every } : { start, stop, step: every };
  }

  function updateAlt() {
    const v = vid.video;
    const alt = (input, out) => {
      const f = parseFrame(input.value, v.fps);
      const isTime = /:/.test(input.value);
      out.textContent = f == null ? '' : isTime ? t('video.as_frame', { n: num(f) }) : t('video.as_time', { time: fmtClock(f / v.fps) });
    };
    alt(vStart, vStartAlt);
    alt(vStop, vStopAlt);
  }

  function schedulePlan() {
    if (!vid.video) return;
    updateAlt();
    clearTimeout(vid.timer);
    vid.plan = null;
    vCreate.disabled = true;
    vid.timer = setTimeout(runPlan, 300);
  }

  async function runPlan() {
    const body = selection(true);
    const my = ++vid.seq;
    if (!body) { vSummary.textContent = t('video.invalid'); fill(vStrip); return; }
    try {
      const p = await api('POST', `/projects/${projectId}/videos/${vid.video.id}/plan`, { json: body });
      if (my !== vid.seq || !vid.video) return;
      vid.plan = p;
      vCreate.disabled = !(p.frames > 0);
      const stepText = vid.unit === 'seconds'
        ? t('video.every_seconds_text', { s: num(Number(vEvery.value)), n: num(p.step) })
        : tn('video.every_frames_text', p.step);
      vSummary.textContent = p.frames > 0 ? tn('video.summary', p.frames, { step: stepText }) : t('video.invalid');
      drawStrip(p);
    } catch (err) {
      if (my !== vid.seq) return;
      vSummary.textContent = err.message;
      fill(vStrip);
    }
  }

  function drawStrip(p) {
    const v = vid.video;
    const k = Math.min(8, p.frames);
    const items = [];
    for (let j = 0; j < k; j++) {
      const idx = k === 1 ? p.first : p.first + p.step * Math.round((j * (p.frames - 1)) / (k - 1));
      items.push(el('li', null,
        el('img', { src: `/api/projects/${projectId}/videos/${v.id}/frame?n=${idx}&w=160`, alt: t('video.frame_alt', { n: num(idx) }), loading: 'lazy' }),
        el('span', { class: 'muted' }, fmtClock(idx / v.fps))));
    }
    fill(vStrip, ...items);
  }

  function showVideo(v) {
    vid.video = v;
    vid.taskMade = false;
    vDrop.hidden = true;
    vUp.hidden = true;
    vSel.hidden = false;
    vInfo.textContent = [fmtClock(v.duration), t('video.fps_n', { n: nf1.format(v.fps) }), `${v.width}×${v.height}`, tn('video.frames_n', v.frames)].join(' · ');
    vStart.value = '0';
    vStop.value = String(Math.max(0, v.frames - 1));
    vEvery.value = '1';
    vid.unit = 'frames';
    setUnit('frames');
    updateAlt();
    runPlan();
  }

  function pickVideo(file) {
    if (!file || vid.busy) return;
    if (!VIDEO_EXT.test(file.name)) { toast(t('video.bad_type'), 'error'); return; }
    vid.busy = true;
    vDrop.hidden = true;
    const form = new FormData();
    form.append('file', file, file.name);
    const xhr = new XMLHttpRequest();
    vid.xhr = xhr;
    const draw = (done, total) => {
      vUp.hidden = false;
      fill(vUp,
        el('div', { class: 'upload-line' },
          el('strong', null, t('video.uploading', { name: file.name })),
          el('button', { type: 'button', class: 'btn small', onclick: () => xhr.abort() }, t('common.cancel'))),
        progressBar(done, total, t('images.upload_progress')));
    };
    draw(0, 1);
    const back = () => { vid.busy = false; vid.xhr = null; vUp.hidden = true; vDrop.hidden = !!vid.video; };
    xhr.upload.addEventListener('progress', (e) => { if (e.lengthComputable) draw(e.loaded, e.total); });
    xhr.addEventListener('abort', back);
    xhr.addEventListener('error', () => { back(); toast(t('common.network_error'), 'error', 10000); });
    xhr.addEventListener('load', () => {
      let data = null;
      try { data = JSON.parse(xhr.responseText); } catch (e) { data = null; }
      back();
      if (xhr.status >= 200 && xhr.status < 300 && data) showVideo(data);
      else toastError(new Error(errorMessage(data, xhr.status)));
    });
    xhr.open('POST', `/api/projects/${projectId}/videos`);
    xhr.setRequestHeader('Accept', 'application/json');
    xhr.send(form);
  }

  /** Drops the uploaded video that no task uses (best effort; 409 = a task uses it). */
  async function discardVideo(quiet) {
    const v = vid.video;
    clearTimeout(vid.timer);
    vid.seq += 1;
    vid.video = null;
    vSel.hidden = true;
    vSummary.textContent = '';
    fill(vStrip);
    if (!quiet) vDrop.hidden = false;
    if (v && !vid.taskMade) {
      try { await api('DELETE', `/projects/${projectId}/videos/${v.id}`); } catch (e) { /* in use or already gone */ }
    }
  }

  function resetVideo() {
    if (vid.xhr) vid.xhr.abort();
    clearTimeout(vid.polling);
    vid.polling = null;
    vid.busy = false;
    vid.jobId = null;
    vid.pollFails = 0;
    vRun.hidden = true;
    fill(vRun);
    modeTabs.hidden = false;
    fieldsBox.hidden = false;
    vUp.hidden = true;
    vDrop.hidden = false;
    vSel.hidden = true;
  }

  async function createFromVideo() {
    if (vid.busy || !vid.video) return;
    const sel = selection(true);
    if (!sel) return;
    const jobSize = jobSizeInput.value.trim();
    if (jobSize && !/^[1-9]\d*$/.test(jobSize)) {
      jobSizeInput.setAttribute('aria-invalid', 'true');
      jobSizeInput.focus();
      toast(t('tasks.job_size_invalid'), 'error');
      return;
    }
    const body = { video: vid.video.id, name: nameInput.value.trim(), ...sel };
    if (jobSize) body.job_size = Number(jobSize);
    vid.busy = true;
    vid.taskMade = true; // from now on the server owns the video
    modeTabs.hidden = true;
    fieldsBox.hidden = true;
    vSel.hidden = true;
    vRun.hidden = false;
    closeBtn.hidden = true;
    closeX.hidden = true;
    fill(vRun, el('p', { class: 'muted' }, t('video.starting')));
    try {
      const job = await api('POST', `/projects/${projectId}/tasks/from-video`, { json: body });
      vid.jobId = job.id;
      vid.pollFails = 0;
      drawRun(job);  // the Cancel button is there from the start, not after the first poll
      pollJob(job.id);
    } catch (err) {
      vid.taskMade = false;
      showRunEnd(false, err.message);
    }
  }

  function drawRun(job) {
    fill(vRun,
      el('div', { class: 'upload-line' },
        el('strong', null, t('video.creating')),
        el('button', { type: 'button', class: 'btn small', onclick: async () => { try { await api('POST', `/jobs/${job.id}/cancel`); } catch (e) { toastError(e); } } }, t('jobs.stop'))),
      progressBar(job.done, job.total, t('video.creating')),
      job.message ? el('p', { class: 'muted job-msg' }, jobMessage(job)) : null);
  }

  async function pollJob(id) {
    let job;
    try {
      job = await api('GET', `/jobs/${id}`);
    } catch (err) {
      vid.pollFails += 1;
      if (vid.pollFails >= POLL_LOST && !vRun.querySelector('.poll-lost')) {
        /* the server does not answer: say so, and let the person leave (the job goes on and shows in the activity bar) */
        add(vRun, el('p', { class: 'report-line bad poll-lost', role: 'alert' }, icon('warning'), t('video.poll_lost')));
        closeBtn.textContent = t('common.close');
        closeBtn.hidden = false;
        closeX.hidden = false;
      }
      vid.polling = setTimeout(() => pollJob(id), 1500);
      return;
    }
    if (vid.jobId !== id) return;
    if (job.status === 'queued' || job.status === 'running') {
      if (vid.pollFails >= POLL_LOST) { closeBtn.hidden = true; closeX.hidden = true; }
      vid.pollFails = 0;
      drawRun(job);
      vid.polling = setTimeout(() => pollJob(id), 1500);
      return;
    }
    vid.polling = null;
    const frames = (job.result && (job.result.frames || job.result.images)) || (vid.plan && vid.plan.frames) || 0;
    if (job.status === 'done') {
      nameInput.value = '';
      jobSizeInput.value = '';
      showRunEnd(true, tn('video.created', frames));
    } else {
      vid.taskMade = !!(job.result && job.result.task);
      showRunEnd(false, job.status === 'cancelled' ? t('video.cancelled') : job.error || tr(`job.status.${job.status}`, job.status));
    }
    await refreshProject();
    await loadTasks();
    grid.reload();
    if (ctx.panels.curation.mounted) ctx.panels.curation.refresh();
  }

  function showRunEnd(ok, text) {
    vid.busy = false;
    fill(vRun,
      el('p', { class: `report-line ${ok ? 'ok' : 'bad'}` }, icon(ok ? 'check' : 'warning'), text),
      ok ? null : el('div', { class: 'drop-actions' }, el('button', { type: 'button', class: 'btn', onclick: () => {
        vRun.hidden = true; modeTabs.hidden = false; fieldsBox.hidden = false;
        if (vid.video) { vSel.hidden = false; schedulePlan(); } else vDrop.hidden = false;
      } }, t('video.back'))));
    if (ok) { vid.video = null; vid.taskMade = true; }
    closeBtn.textContent = t(ok ? 'common.close' : 'common.cancel');
    closeBtn.hidden = false;
    closeX.hidden = false;
  }
  const closeBtn = el('button', { type: 'button', class: 'btn', id: 'new-task-close', onclick: () => newDlg.close() }, t('common.cancel'));
  const closeX = el('button', { type: 'button', class: 'icon-btn', 'aria-label': t('common.close'), title: t('common.close'), onclick: () => newDlg.close() }, icon('close', 18));
  const newDlg = el('dialog', { class: 'dialog wide new-task-dialog', 'aria-label': t('tasks.new_title') },
    el('div', { class: 'dialog-body' },
      el('div', { class: 'dlg-head' }, el('h2', null, t('tasks.new_title')), closeX),
      formBox, progressBox, reportBox,
      el('div', { class: 'form-actions' }, closeBtn)));
  newDlg.addEventListener('cancel', (e) => { // Esc must not hide a running upload: say how to stop it
    if (uploading) {
      e.preventDefault();
      const stop = document.getElementById('upload-cancel');
      if (stop) stop.focus();
      toast(t('images.upload_running'), 'info', 4000);
    } else if (vid.jobId && vid.polling && vid.pollFails < POLL_LOST) e.preventDefault();
  });
  newDlg.addEventListener('close', () => {
    if (uploading) { newDlg.showModal(); return; } // a second Esc closes a dialog whose first one was refused
    // a video uploaded but never used by a task is removed again
    if (vid.jobId && vid.polling) return;
    resetVideo();
    if (vid.video) discardVideo(true);
    setMode('images');
  });

  function setStage(stage) {
    formBox.hidden = stage !== 'form';
    progressBox.hidden = stage !== 'uploading';
    closeBtn.hidden = stage === 'uploading';
    closeBtn.textContent = t(stage === 'done' ? 'common.close' : 'common.cancel');  // "Fechar" only once something was done
    closeX.hidden = stage === 'uploading';
  }

  function drawStaged() {
    stagedBox.hidden = !staged;
    drop.hidden = !!staged;
    if (!staged) { fill(stagedBox); return; }
    fill(stagedBox,
      icon('images', 28),
      el('strong', null, tn('images.staged', staged.length)),
      el('div', { class: 'drop-actions' },
        el('button', { type: 'button', class: 'btn primary', id: 'new-task-create', onclick: () => startUpload(staged) }, t('tasks.create')),
        el('button', { type: 'button', class: 'btn', onclick: () => { staged = null; drawStaged(); } }, t('images.change_files'))));
  }

  function openNewTask(entries) {
    if (!uploading && !(vid.jobId && vid.polling)) {
      resetVideo();
      if (entries && entries.length) setMode('images');
      fill(reportBox);
      const files = entries ? usable(entries) : [];
      staged = files.length ? files : null;
      drawStaged();
      setStage('form');
      nameInput.value = '';   // a form that was cancelled starts empty again (#342)
      jobSizeInput.value = '';
      jobSizeInput.removeAttribute('aria-invalid');
      showNameError('');
    }
    if (!newDlg.open) newDlg.showModal();
    nameInput.focus();
  }

  const imageTabs = tabsView('images', t('step.images'), [
    { id: 'gallery', label: t('images.tab_gallery'), nodes: [el('div', { class: 'grid-head' }, gridTitle, filterBox), grid.node] },
    { id: 'tasks', label: t('images.tab_tasks'), nodes: [tasksBox] },
  ], 'gallery');
  add(node,
    el('h2', { id: 'h-images' }, t('step.images')),
    imageTabs.node,
    newDlg, fileInput, folderInput);

  const reload = () => { loadTasks(); grid.reload(); };
  /* a project without images opens on its tasks, where images are added (unless the person chose a tab before) */
  panel.mount = () => { if (!imageTabs.remembered && !hasImages()) imageTabs.select('tasks'); reload(); };
  panel.refresh = reload;
  panel.show = () => { refreshCounts(); loadTasks(); };  // coming back from another step (an import, an apply, a sync changed the counts)
  /** Step 4 sends the person here to see one part of a dataset version. */
  panel.showDataset = (n, part, fold = null) => {
    filter = { task: null, job: null, dataset: { n, part, fold } };
    imageTabs.select('gallery');
    drawFilter();
    grid.reload();
  };
  /** Step 6 sends the person here to see the worst images of a model's evaluation (#611). */
  panel.showIds = (ids, model) => {
    filter = { task: null, job: null, dataset: null, ids: { list: ids, model } };
    imageTabs.select('gallery');
    drawFilter();
    grid.reload();
  };
  /** A deleted dataset version can no longer filter the gallery (every load would answer 404). */
  panel.forgetDataset = (n) => {
    if (!filter.dataset || filter.dataset.n !== n) return;
    filter = { ...filter, dataset: null };
    drawFilter();
    if (panel.mounted) grid.reload();
  };
  return panel;
}
