/* Annotating in an external tool from the Annotations step: CVAT tasks and FiftyOne edit sessions, their
 * setup, and bringing the annotations back. */
import { editedNotice, editedSession, refreshEdited, sessionEdited } from './edited.js';
import { cvatOpenRow, cvatProjectGone, fiftyoneOpenRow } from './open.js';
import { fiftyoneButton, fiftyoneError, sayFiftyoneMissing } from '../shell/features.js';
import {
  SESSION_TOOLS, TOOLS, imagesOf, isHttp, openCvatAudit, openToolSetup, serverName, statusLabel, toolName,
} from './tools.js';
import { api } from '../shell/api.js';
import { confirmDialog, openModal } from '../shell/dialogs.js';
import { el, fill, icon } from '../shell/dom.js';
import { fmtDateTime, num, t, tn, tr } from '../shell/i18n.js';
import { dropTab, reserveTab } from '../shell/newtab.js';
import { toast, toastError } from '../shell/toast.js';

/** The CVAT and FiftyOne tabs of the Annotations step (tasks to annotate, edit sessions, the sync block).
 *  `page` is the object of the project page; it also brings diffView and issueName from the layers above.
 *  Returns the block (its node and the functions the step calls). */
export function annotateBlock(page) {
  const { ctx, diffView, isActive, issueName, jobStarted, projectId, refreshProject, startSync, syncBlock } = page;
  const state = { tools: {}, sessions: [], issues: {}, scope: null, issue: '', loaded: false, cvat: { scope: 'project', task: null, job: null, direction: 'both' } };

  const cvatScopeBox = el('div', { class: 'scope-wrap' });
  const cvatInfoBox = el('div', { class: 'cvat-info' });
  const cvatDirBox = el('div', { class: 'cvat-dir' });
  const cvatSendBox = el('div', { class: 'send-row' });
  const cvatCtaBox = el('div', { class: 'cvat-cta card', hidden: true });
  const cvatOpenBox = el('div', { class: 'cvat-open' });
  const cvatCard = el('div', { class: 'card cvat-card', hidden: true }, cvatInfoBox, cvatScopeBox, cvatOpenBox, cvatDirBox, cvatSendBox);
  const foOpenBox = el('div', { class: 'fo-open' });
  const editedBox = el('div', { class: 'fo-edited-box' });   // "N images edited in FiftyOne, not in the project yet" (#700)
  /* "Revisar antes de aplicar" (#706): off, the App's "Send to ActiveLabel" applies at once when nothing needs a look */
  const reviewFirst = el('input', { type: 'checkbox', id: 'annotate-review-first', onchange: (e) => saveReviewFirst(e.target) });
  const prefsBox = el('p', { class: 'fo-prefs' },
    el('label', { for: 'annotate-review-first' }, reviewFirst, ' ', t('annotate.review_first')),
    el('span', { class: 'muted small', id: 'annotate-review-first-help' }, ' ', t('annotate.review_first_help')));
  reviewFirst.setAttribute('aria-describedby', 'annotate-review-first-help');
  const scopeBox = el('div', { class: 'scope-wrap' });
  const chooseBox = el('details', { class: 'choose-images', id: 'annotate-choose' },
    el('summary', null, t('annotate.choose')), scopeBox);
  const sessionsBox = el('div', { class: 'sessions' });
  const sync = syncBlock();
  const cvatSection = el('section', { class: 'annot-section', id: 'annotate-cvat', 'aria-labelledby': 'h-cvat' },
    el('h3', { id: 'h-cvat', class: 'sr-only' }, toolName('cvat')),
    cvatCtaBox, cvatCard, sync.node);
  const fiftyoneSection = el('section', { class: 'annot-section', id: 'annotate-fiftyone', 'aria-labelledby': 'h-fiftyone' },
    el('h3', { id: 'h-fiftyone', class: 'sr-only' }, t('annotate.edit_title', { tool: toolName('fiftyone') })),
    foOpenBox, editedBox, prefsBox, chooseBox,
    el('div', { class: 'sessions-section', id: 'annotate-sessions', role: 'region', 'aria-labelledby': 'h-sessions' },
      el('h4', { id: 'h-sessions' }, t('annotate.sessions')),
      sessionsBox));

  const counts = () => (ctx.project && ctx.project.counts) || {};
  const isReady = (tool) => tool === 'fiftyone' || !!(state.tools[tool] && state.tools[tool].configured);
  const selectedIds = () => (ctx.curationSelected ? ctx.curationSelected() : []);
  const unlabeledCount = () => Math.max(0, (counts().active || 0) - (counts().annotated || 0) - (counts().empty || 0) - (counts().preannotated || 0));
  const issueEntries = () => Object.entries(state.issues).filter(([, n]) => n > 0);
  const scopeCount = (scope) => {
    if (scope === 'all') return counts().active || 0;
    if (scope === 'unlabeled') return unlabeledCount();
    if (scope === 'selected') return selectedIds().length;
    if (scope === 'preannotated') return state.pre || 0;
    return Number(state.issues[state.issue]) || 0;
  };
  const scopeAvailable = (scope) => (scope === 'issue' ? issueEntries().length > 0 : scopeCount(scope) > 0);

  /* the scope the person chose last in this project, else all the images (never a silent "only the unannotated") */
  const scopeKey = `al-scope-${projectId}`;
  const rememberScope = (scope) => { try { localStorage.setItem(scopeKey, scope); } catch (err) { /* not remembered */ } };
  function defaultScope() {
    let last = null;
    try { last = localStorage.getItem(scopeKey); } catch (err) { /* not remembered */ }
    return last && last !== 'issue' && scopeAvailable(last) ? last : 'all';
  }

  // ---- which images
  /* CVAT: what to synchronise (whole project, one task or one job of it) */
  const cvatTasks = () => ctx.tasksList || [];
  const cvatTask = () => cvatTasks().find((x) => x.id === state.cvat.task) || null;
  function normaliseCvat() {
    const c = state.cvat;
    const list = cvatTasks();
    if (!list.length) { c.scope = 'project'; c.task = null; c.job = null; return; }
    if (!list.some((x) => x.id === c.task)) c.task = list[0].id;
    const task = cvatTask();
    if (!task.jobs.some((j) => j.n === c.job)) c.job = task.jobs.length ? task.jobs[0].n : null;
    if (c.scope === 'job' && c.job == null) c.scope = 'task';
  }
  function drawCvatScope() {
    normaliseCvat();
    const c = state.cvat;
    const list = cvatTasks();
    const task = cvatTask();
    const focusedId = document.activeElement && document.activeElement.name === 'cvat-scope' ? document.activeElement.id : null;
    const row = (scope, label, extra, ok = true) => {
      const id = `cvat-scope-${scope}`;
      const input = el('input', { type: 'radio', name: 'cvat-scope', id, value: scope, disabled: !ok });
      input.checked = c.scope === scope;
      input.addEventListener('change', () => { c.scope = scope; drawScope(); drawSend(); });
      return el('div', { class: `scope-row${c.scope === scope ? ' selected' : ''}${ok ? '' : ' is-disabled'}` },
        el('div', { class: 'scope-line' }, input, el('label', { for: id }, label), extra));
    };
    const taskSelect = el('select', {
      id: 'cvat-task', 'aria-label': t('sync.scope_task'), disabled: !list.length,
      onchange: (e) => { c.task = Number(e.target.value); c.job = null; if (c.scope === 'project') c.scope = 'task'; drawScope(); drawSend(); },
    }, ...list.map((x) => el('option', { value: String(x.id), selected: x.id === c.task }, `${x.name} (${num(x.active)})`)));
    const jobSelect = el('select', {
      id: 'cvat-job', 'aria-label': t('sync.scope_job'), disabled: !task || !task.jobs.length,
      onchange: (e) => { c.job = Number(e.target.value); c.scope = 'job'; drawScope(); drawSend(); },
    }, ...(task ? task.jobs : []).map((j) => el('option', { value: String(j.n), selected: j.n === c.job }, t('tasks.job_option', { n: j.n, first: num(j.first), last: num(j.last) }))));
    fill(cvatScopeBox, el('fieldset', { class: 'field scope-box' },
      el('legend', null, t('sync.scope_title')),
      el('div', { class: 'scope-rows' },
        row('project', t('sync.scope_project'), el('span', { class: 'count' }, num(counts().active || 0))),
        row('task', t('sync.scope_task'), taskSelect, list.length > 0),
        row('job', t('sync.scope_job'), jobSelect,list.length > 0 && !!task && task.jobs.length > 0)),
      list.length ? null : el('p', { class: 'hint' }, t('sync.no_tasks'))));
    if (focusedId) { const f = document.getElementById(focusedId); if (f) f.focus(); }
    drawOpen();
  }

  /* the Open in CVAT / Open in FiftyOne rows (#602): what the scope chosen in CVAT opens, or why it cannot yet */
  function drawOpen() {
    const c = state.cvat;
    const task = c.scope === 'project' ? null : cvatTask();
    fill(cvatOpenBox, cvatOpenRow({ link: cvatLink(), configured: isReady('cvat'), task, job: c.scope === 'job' ? c.job : null, gone: !!cvatLink() && cvatProjectGone(page) }));
  }

  function drawSend() {
    drawCvatSend();
    drawSendDefault();
  }

  // ---- CVAT section: link info, direction, compare
  const cvatLink = () => ctx.links.find((l) => l.tool === 'cvat') || null;
  const cvatBusy = () => ctx.jobs.some((j) => j.kind === 'sync_plan' && isActive(j));
  const DIRECTIONS = ['both', 'pull', 'push'];

  function drawCvatInfo() {
    const link = cvatLink();
    const gone = !!link && cvatProjectGone(page);
    const ready = isReady('cvat');
    const name = toolName('cvat');
    cvatCtaBox.hidden = !state.loaded || ready || !!link;
    cvatCard.hidden = !state.loaded || !(ready || link);
    if (!state.loaded) return;
    fill(cvatCtaBox,
      icon('cvat', 24),
      el('p', null, t('sync.cta_text', { tool: name })),
      el('button', { type: 'button', class: 'btn primary', id: 'cvat-setup', onclick: () => openSetup('cvat') }, t('sync.cta', { tool: name })));
    const unlink = el('button', { type: 'button', class: 'btn small', onclick: async () => {
      const ok = await confirmDialog({ title: t('sync.unlink_title', { tool: name }), text: t('sync.unlink_text', { tool: name }), confirmLabel: t('sync.unlink') });
      if (!ok) return;
      try {
        await api('DELETE', `/projects/${projectId}/links/cvat`);
        toast(t('sync.unlinked', { tool: name }), 'success', 4000);
        await refreshProject();
        sync.draw();
        drawCvatInfo();
        drawCvatSend();
      } catch (err) { toastError(err); }
    } }, t('sync.unlink'));
    const head = el('div', { class: 'cvat-head' },
      el('span', { class: 'session-tool' }, icon('cvat', 22), el('strong', null, name)),
      el('div', { class: 'cvat-head-actions' },
        ready ? el('button', { type: 'button', class: 'btn small', id: 'cvat-audit', onclick: openCvatAudit }, t('audit.button')) : null,
        link ? unlink : null,
        el('button', { type: 'button', class: 'btn small', 'aria-label': t('annotate.setup_for', { tool: name }), onclick: () => openSetup('cvat') }, t('annotate.settings'))));
    const entry = state.tools.cvat || {};
    /* the task list asks CVAT: when it did not answer, the connection is set up but not "connected" now (#602) */
    const down = cvatTasks().some((x) => x.cvat === 'unreachable');
    const connected = ready && entry.url
      ? el('p', { class: 'muted small cvat-connected' }, t(down ? 'annotate.connected_down' : 'annotate.connected_as', { server: serverName(entry.url), user: entry.username || '?' }))
      : null;
    const facts = link
      ? el('dl', { class: 'facts link-facts' },
          el('div', null, el('dt', null, t('sync.remote_project', { tool: name })), el('dd', null,
            gone
              ? el('span', { class: 'muted' }, t('sync.remote_deleted', { name: link.remote_name || name })) /* no link to a deleted project (#695) */
              : isHttp(link.url)
              ? el('a', { href: link.url, target: '_blank', rel: 'noopener' }, link.remote_name || name, ' ', icon('external', 14))
              : el('span', null, link.remote_name || name))),
          el('div', null, el('dt', null, t('sync.images_linked')), el('dd', null, num(link.images || 0))),
          el('div', null, el('dt', null, t('sync.last')), el('dd', null, link.last_sync ? fmtDateTime(link.last_sync) : t('sync.never'))))
      : el('p', { class: 'muted' }, t('sync.creates', { tool: name }));
    fill(cvatInfoBox, head, connected, facts);
  }

  function drawCvatDirection() {
    const c = state.cvat;
    const name = toolName('cvat');
    const help = el('p', { class: 'muted dir-help', id: 'cvat-dir-help', 'aria-live': 'polite' }, t(`sync.dir_${c.direction}_help`, { tool: name }));
    const focusedId = document.activeElement && document.activeElement.name === 'cvat-direction' ? document.activeElement.id : null;
    const opts = DIRECTIONS.map((d) => {
      const id = `cvat-direction-${d}`;
      const input = el('input', { type: 'radio', name: 'cvat-direction', id, value: d, class: 'sr-only', 'aria-describedby': 'cvat-dir-help' });
      input.checked = c.direction === d;
      input.addEventListener('change', () => { c.direction = d; help.textContent = t(`sync.dir_${d}_help`, { tool: name }); });
      return el('label', { for: id, class: 'seg-opt' }, input, t(`sync.dir_${d}`, { tool: name }));
    });
    fill(cvatDirBox, el('fieldset', { class: 'field' },
      el('legend', null, t('sync.dir_title')),
      el('div', { class: 'seg', role: 'radiogroup', 'aria-label': t('sync.dir_title') }, ...opts),
      help));
    if (focusedId) { const f = document.getElementById(focusedId); if (f) f.focus(); }
  }

  function drawCvatSend() {
    if (!state.loaded) { fill(cvatSendBox); return; }
    fill(cvatSendBox,
      el('button', { type: 'button', class: 'btn primary big', id: 'cvat-compare', disabled: cvatBusy(), title: cvatBusy() ? t('sync.compare_busy') : null, onclick: (e) => syncCvat(e.currentTarget) },
        icon('cvat', 20), t('sync.compare', { tool: toolName('cvat') })));
  }

  /* redraws the CVAT card only when what it shows changed (keeps the focus) */
  let cvatSig = null;
  function cvatDraw(force) {
    if (!state.loaded) return;
    const sig = JSON.stringify([ctx.links, cvatBusy(), isReady('cvat'), cvatProjectGone(page), (ctx.tasksList || []).length, cvatTasks().some((x) => x.cvat === 'unreachable'), state.tools.cvat]);
    if (!force && sig === cvatSig) return;
    cvatSig = sig;
    drawCvatInfo();
    drawCvatSend();
    drawOpen();
  }
  ctx.cvatDraw = cvatDraw;

  async function syncCvat(button) {
    if (!isReady('cvat')) { openSetup('cvat'); return; }
    normaliseCvat();
    const c = state.cvat;
    const body = { scope: c.scope, direction: c.direction };
    if (c.scope !== 'project') body.task = c.task;
    if (c.scope === 'job') body.job = c.job;
    // no link yet: use the project of the same name in CVAT or create it
    if (!ctx.links.some((l) => l.tool === 'cvat')) body.create = true;
    button.disabled = true;
    try {
      await startSync('cvat', body);
    } catch (err) { toastError(err); } finally { drawSend(); }
  }

  function drawScope() {
    if (!state.loaded) return;
    drawCvatScope();
    if (!state.scope || !scopeAvailable(state.scope)) state.scope = defaultScope();
    if (!state.issue || !state.issues[state.issue]) state.issue = (issueEntries()[0] || [''])[0];
    const focusedId = document.activeElement && document.activeElement.name === 'annotate-scope' ? document.activeElement.id : null;
    const row = (scope, label, extra, hint) => {
      const id = `annotate-scope-${scope}`;
      const ok = scopeAvailable(scope);
      const input = el('input', { type: 'radio', name: 'annotate-scope', id, value: scope, disabled: !ok, 'aria-describedby': hint && !ok ? `${id}-hint` : null });
      input.checked = state.scope === scope;
      input.addEventListener('change', () => { state.scope = scope; rememberScope(scope); drawScope(); drawSend(); });
      return el('div', { class: `scope-row${state.scope === scope ? ' selected' : ''}${ok ? '' : ' is-disabled'}` },
        el('div', { class: 'scope-line' },
          input,
          el('label', { for: id }, label),
          extra),
        hint && !ok ? el('p', { id: `${id}-hint`, class: 'hint' }, hint) : null);
    };
    const issueSelect = el('select', {
      id: 'annotate-issue', 'aria-label': t('annotate.scope_issue_pick'), disabled: !issueEntries().length,
      onchange: (e) => { state.issue = e.target.value; state.scope = 'issue'; drawScope(); drawSend(); },
    }, ...issueEntries().map(([k, n]) => el('option', { value: k, selected: k === state.issue }, `${issueName(k)} (${num(n)})`)));
    fill(scopeBox, el('fieldset', { class: 'field scope-box' },
      el('legend', null, t('annotate.scope_title')),
      el('div', { class: 'scope-rows' },
        row('all', t('annotate.scope_all'), el('span', { class: 'count' }, num(counts().active || 0))),
        row('unlabeled', t('annotate.scope_unlabeled'), el('span', { class: 'count' }, num(unlabeledCount())), t('annotate.scope_unlabeled_none')),
        /* the other choices only appear when they apply (no greyed-out cards) */
        scopeAvailable('preannotated') ? row('preannotated', t('annotate.scope_preannotated'), el('span', { class: 'count' }, num(state.pre || 0))) : null,
        scopeAvailable('selected') ? row('selected', t('annotate.scope_selected'), el('span', { class: 'count' }, num(selectedIds().length))) : null,
        scopeAvailable('issue') ? row('issue', t('annotate.scope_issue'), issueSelect) : null),
      el('p', { class: 'hint' }, t('annotate.scope_more'))));
    if (focusedId) { const f = document.getElementById(focusedId); if (f) f.focus(); }
  }

  function drawSendDefault() {
    const n = state.loaded ? scopeCount(state.scope) : 0;
    fill(foOpenBox, fiftyoneOpenRow(page, { edit: send, count: n }));
  }

  /** "Edit in FiftyOne": sends the images chosen (all, unless chosen otherwise) and opens the App in a new tab. */
  async function send() {
    if (sayFiftyoneMissing()) return;   // #658
    const tool = 'fiftyone';
    const body = { tool, scope: state.scope };
    if (state.scope === 'selected') body.ids = selectedIds();
    if (state.scope === 'issue') body.issue = state.issue;
    const button = document.getElementById('fo-open');
    if (button) button.disabled = true;
    const tab = reserveTab(t('cur.fo_waiting'));
    try {
      const job = await api('POST', `/projects/${projectId}/annotate`, { json: body });
      if (job && job.id) ctx.openWhenDone.set(job.id, tab); else dropTab(tab);
      toast(t('annotate.send_started', { tool: toolName(tool) }), 'info', 4000);
      await jobStarted(job);
    } catch (err) {
      dropTab(tab);
      if (!fiftyoneError(err)) toastError(err);
    } finally {
      drawSend();
    }
  }

  const openSetup = (tool) => openToolSetup(tool, state.tools[tool], loadTools);

  // ---- sessions
  function sessionCard(s) {
    const tool = s.tool;
    const sid = encodeURIComponent(s.id);
    const links = (s.links || []).filter((l) => l && isHttp(l.url));
    const sessionTask = s.scope === 'task' ? (ctx.tasksList || []).find((x) => x.id === s.task) : null;
    const taskLabel = sessionTask ? sessionTask.name : `#${s.task}`;
    let scopeText = tr(`annotate.scope.${s.scope}`, s.scope || '');
    if (s.scope === 'issue' && s.issue) scopeText = t('annotate.scope_issue_named', { issue: issueName(s.issue) });
    else if (s.scope === 'selected' && imagesOf(s) === 1) scopeText = t('annotate.scope_one');  // "Edit in FiftyOne" of one image
    else if (s.scope === 'task' && s.task != null) scopeText = s.job ? t('annotate.scope_job_named', { task: taskLabel, n: s.job }) : t('annotate.scope_task_named', { task: taskLabel });
    const legacy = !SESSION_TOOLS.includes(tool);
    const legacyTool = !TOOLS.includes(tool);  // links to a tool this version does not have are not offered
    const live = !legacy && (s.status === 'open' || s.status === 'pulled');
    const act = (label, cls, fn) => {
      const b = el('button', { type: 'button', class: `btn ${cls}` }, label);
      b.addEventListener('click', async () => {
        b.disabled = true;
        try { await fn(); } catch (err) { b.disabled = false; toastError(err); await reloadIfChanged(err); }
      });
      return b;
    };
    const changes = s.diff ? (s.diff.images_changed || 0) : 0;
    const buttons = [];
    if (live) {
      buttons.push(act(s.status === 'pulled' ? t('annotate.pull_again') : t('annotate.pull'), s.status === 'open' ? 'primary' : '', async () => {
        const job = await api('POST', `/projects/${projectId}/annotate/${sid}/pull`);
        toast(t('annotate.pull_started'), 'info', 3000);
        await jobStarted(job);
      }));
    }
    if (!legacy && s.status === 'pulled' && s.diff) {
      const apply = act(t('annotate.apply'), 'primary', async () => {
        const r = await api('POST', `/projects/${projectId}/annotate/${sid}/apply`);
        const a = (r && r.applied) || {};
        toast(tn('annotate.applied_n', Number(a.images_changed ?? a.images ?? changes) || 0), 'success', 6000);
        await refreshProject();
        Object.values(ctx.panels).forEach((panel) => panel.mounted && panel.refresh && panel.refresh());
      });
      apply.disabled = !changes;
      buttons.unshift(apply);
      if (!changes) buttons.splice(1, 0, el('span', { class: 'muted small' }, t('annotate.apply_none')));
      buttons.push(act(t('annotate.discard'), '', async () => {
        await api('POST', `/projects/${projectId}/annotate/${sid}/discard`);
        toast(t('annotate.discarded'), 'success', 4000);
        await loadSessions();
      }));
    }
    if (!legacy && s.status === 'applied' && s.applied && 'backup' in s.applied) {
      buttons.push(act(t('annotate.undo'), '', async () => {
        await api('POST', `/projects/${projectId}/annotate/${sid}/undo`);
        toast(t('annotate.undone'), 'success', 6000);
        await refreshProject();
        await loadSessions();
      }));
    }
    const edited = sessionEdited(ctx.fiftyoneEdited, s.id);
    if (!legacy && s.status === 'applied' && edited) {   // edited again in the App after the apply
      buttons.unshift(act(t('annotate.edited_pull'), 'primary', async () => {
        const job = await api('POST', `/projects/${projectId}/annotate/${sid}/pull?apply=auto`);
        toast(t('annotate.pull_started'), 'info', 3000);
        await jobStarted(job);
      }));
    }
    buttons.push(el('button', { type: 'button', class: 'btn danger-outline', onclick: () => closeSession(s) }, t('annotate.close')));
    const diffShown = s.status === 'pulled' && s.diff;
    return el('li', { class: `session card status-${s.status}`, id: `session-${s.id}`, tabindex: '-1', 'data-session': s.id, 'aria-label': `${toolName(tool)} · ${scopeText}` },
      el('div', { class: 'session-head' },
        el('span', { class: 'session-tool' }, icon(TOOLS.includes(tool) ? tool : 'info', 22), el('strong', null, toolName(tool))),
        el('span', { class: `sbadge sbadge-${s.status}` }, statusLabel(s.status))),
      el('p', { class: 'session-meta' },
        el('span', null, scopeText),
        el('span', { class: 'sep', 'aria-hidden': 'true' }, '·'),
        el('span', null, tn('annotate.images_n', imagesOf(s))),
        el('span', { class: 'sep', 'aria-hidden': 'true' }, '·'),
        el('time', { datetime: s.created || null }, fmtDateTime(s.created))),
      links.length && !legacyTool
        ? el('p', { class: 'session-links' }, ...links.map((l) => el('a', { class: 'btn small', href: l.url, target: '_blank', rel: 'noopener' },
            links.length === 1 ? t('annotate.open_in', { tool: toolName(tool) }) : l.label || t('annotate.open_in', { tool: toolName(tool) }), icon('external', 14))))
        : null,
      legacy ? el('p', { class: 'muted' }, t('annotate.legacy_note')) : null,
      edited ? el('p', { class: 'fo-session-edited small' }, icon('fiftyone', 14), tn('annotate.edited_n', edited)) : null,
      s.status === 'open' && !legacy ? el('p', { class: 'muted' }, t('annotate.pull_hint')) : null,
      s.status === 'open' && tool === 'fiftyone' ? el('p', { class: 'muted small fo-mouse-hint' }, t('annotate.fo_mouse_hint')) : null,
      diffShown ? diffView(projectId, s.diff) : null,
      el('div', { class: 'session-actions' }, ...buttons));
  }

  async function closeSession(s) {
    const ok = await new Promise((resolve) => {
      const okBtn = el('button', { type: 'submit', value: 'ok', class: 'btn danger' }, t('annotate.close_confirm'));
      const cancelBtn = el('button', { type: 'button', class: 'btn', onclick: () => dlg.close('cancel') }, t('common.cancel'));
      const dlg = el('dialog', { class: 'dialog', 'aria-label': t('annotate.close_title') },
        el('form', { method: 'dialog', class: 'dialog-body' },
          el('h2', null, t('annotate.close_title')),
          el('p', null, t('annotate.close_text')),
          el('div', { class: 'form-actions' }, okBtn, cancelBtn)));
      dlg.addEventListener('close', () => { dlg.remove(); resolve(dlg.returnValue === 'ok'); });
      document.body.appendChild(dlg);
      openModal(dlg);
      cancelBtn.focus();
    });
    if (!ok) return;
    try {
      await api('DELETE', `/projects/${projectId}/annotate/${encodeURIComponent(s.id)}`);
      toast(t('annotate.closed'), 'success', 4000);
      await loadSessions();
    } catch (err) { toastError(err); }
  }

  /** A 404/409 means the session is not what this tab shows (another tab applied, discarded or closed it): show the real state. */
  async function reloadIfChanged(err) {
    if (err && (err.status === 404 || err.status === 409)) await loadSessions();
  }

  /** The line with the count of images edited in the App, and where it leads: the session that holds them. */
  function drawEdited() {
    const data = ctx.fiftyoneEdited;
    const session = editedSession(data);
    fill(editedBox, state.loaded ? editedNotice(data, { id: 'fo-edited', onReview: session ? () => reveal(session.id) : null }) : null);
  }

  function drawSessions() {
    if (!state.loaded) { fill(sessionsBox, el('p', { class: 'muted' }, t('common.loading'))); return; }
    fill(sessionsBox, state.sessions.length
      ? el('ul', { class: 'plain session-list' }, ...state.sessions.map(sessionCard))
      : el('div', { class: 'empty-note' }, icon('info'), el('p', null, t('annotate.sessions_empty'))));
  }

  // ---- data
  async function loadTools() {
    try {
      const list = await api('GET', '/annotation-tools');
      state.tools = {};
      (Array.isArray(list) ? list : []).forEach((x) => { state.tools[x.tool] = x; });
    } catch (err) { toastError(err); }
    drawSend();
    cvatDraw(true);
  }
  async function loadPrefs() {
    try {
      const prefs = await api('GET', `/projects/${projectId}/annotate/prefs`);
      reviewFirst.checked = !!(prefs && prefs.review_first);
    } catch (err) { if (err.status !== 404) toastError(err); }
  }
  async function saveReviewFirst(box) {
    try {
      const prefs = await api('PUT', `/projects/${projectId}/annotate/prefs`, { json: { review_first: box.checked } });
      box.checked = !!(prefs && prefs.review_first);
    } catch (err) { box.checked = !box.checked; toastError(err); }
  }
  async function loadSessions() {
    try {
      await refreshEdited(page);   // the sessions and the count of edited images, drawn by `onEdited`
    } catch (err) { toastError(err); }
    state.loaded = true;
    editedSig = JSON.stringify(ctx.fiftyoneEdited);
    drawSessions();
    drawEdited();
    drawScope();
    drawCvatDirection();
    drawSend();
    cvatDraw(true);
  }
  /* the sessions and the count came (here, or when the person came back to the window): draw them */
  let editedSig = null;
  function onEdited(data) {
    state.sessions = (data && data.sessions) || [];
    const sig = JSON.stringify(data);
    if (!state.loaded || sig === editedSig) return;   // nothing new: keep the cards (and the focus) as they are
    editedSig = sig;
    drawSessions();
    drawEdited();
  }
  ctx.editedWatchers = ctx.editedWatchers || [];
  ctx.editedWatchers.push(onEdited);

  async function loadTasks() {
    try {
      ctx.tasksList = (await api('GET', `/projects/${projectId}/tasks`)).tasks || [];
    } catch (err) { if (err.status !== 404) toastError(err); }
    sync.draw();
    drawScope();
  }
  async function loadPre() {
    try {
      const d = await api('GET', `/projects/${projectId}/preannotations`);
      state.pre = (d.tasks || []).reduce((a, x) => a + (x.unreviewed || 0), 0);
    } catch (err) { state.pre = 0; }
  }
  async function loadIssues() {
    try {
      const summary = await api('GET', `/projects/${projectId}/curation`);
      state.issues = (summary && summary.issues) || {};
    } catch (err) { state.issues = {}; }
  }
  const reveal = (sessionId) => {
    const card = sessionId ? document.getElementById(`session-${sessionId}`) : null;
    if (sessionId === 'sync') { sync.reveal(); return; }
    const target = card || document.getElementById('annotate-sessions');
    if (!target) return;
    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    if (card) { card.focus({ preventScroll: true }); card.classList.add('flash'); setTimeout(() => card.classList.remove('flash'), 2500); }
  };

  return {
    cvatSection,
    fiftyoneSection,
    reveal,
    mount: async () => { sync.draw(); await Promise.all([loadTools(), loadIssues(), loadTasks(), loadPre(), loadPrefs()]); await loadSessions(); },
    refresh: async () => { sync.draw(); await Promise.all([loadIssues(), loadTasks(), loadPre()]); await loadSessions(); },
    show: async () => { await Promise.all([loadIssues(), loadTasks(), loadPre()]); drawScope(); drawSend(); },
    /** the CVAT tab with one task chosen (review of its pre-annotations) */
    focusCvatTask: (taskId) => { state.cvat.scope = 'task'; state.cvat.task = taskId; state.cvat.job = null; drawScope(); drawSend(); },
    /** the FiftyOne tab with a scope chosen */
    focusScope: async (scope) => { await loadPre(); state.scope = scope; chooseBox.open = true; drawScope(); drawSend(); },
  };
}
