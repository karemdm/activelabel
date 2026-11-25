/* The annotation tools (CVAT, FiftyOne): names, statuses, job progress messages, badges, the login setup
 * dialog and the CVAT audit. */
import { api, errorParams } from '../shell/api.js';
import { confirmDialog, openModal } from '../shell/dialogs.js';
import { el, fill, icon } from '../shell/dom.js';
import { I18N, t, tn, tr } from '../shell/i18n.js';
import { toast, toastError } from '../shell/toast.js';

/* "Imported from CVAT"; the remote name is the tooltip. */
export const originBadge = (origin) => {
  if (!origin || !origin.tool) return null;
  const known = TOOLS.includes(origin.tool);
  const label = known ? t('project.imported_from', { tool: toolName(origin.tool) }) : t('project.origin_removed');
  return el('span', { class: 'badge origin-badge', title: origin.name ? String(origin.name) : label, tabindex: '0', 'aria-label': origin.name ? `${label}: ${origin.name}` : label },
    icon(known ? origin.tool : 'info', 14), label);
};

/* "Linked to CVAT"; the remote project name is the tooltip. */
export const linkBadge = (link) => {
  const known = TOOLS.includes(link.tool);
  const label = known ? t('project.linked_to', { tool: toolName(link.tool) }) : t('project.link_removed');
  return el('span', { class: 'badge origin-badge link-badge', title: link.remote_name ? String(link.remote_name) : label, tabindex: '0', 'aria-label': link.remote_name ? `${label}: ${link.remote_name}` : label },
    icon(known ? link.tool : 'info', 14), label);
};

export const TOOLS = ['fiftyone', 'cvat'];

/* tools that run edit sessions; a session of another one (an old workspace may hold one) can only be closed */
export const SESSION_TOOLS = ['fiftyone'];

export const DEFAULT_URLS = { cvat: 'https://app.cvat.ai' };

/* a tool this version no longer has (an old workspace may still mention one) is named generically */
export const toolName = (tool) => (TOOLS.includes(tool) ? tr(`annotate.tool.${tool}`, tool) : t('annotate.tool_removed'));

export const statusText = (status) => tr(`job.status.${status}`, status);

/* progress messages the server writes in English; the ones that carry a tool name are matched by shape */
export const JOB_MESSAGE_SHAPES = [
  [/^Listing (.+) data$/, 'jobmsg.listing'],
  [/^Reading annotations from (\w+)$/, 'jobmsg.reading_annotations_from'],
  [/^Reading (\w+)$/, 'jobmsg.reading'],
  [/^Writing to (\w+)$/, 'jobmsg.writing'],
  [/^Sending to (\w+)$/, 'jobmsg.sending'],
];

/** A job's progress message in the interface language (the text itself when it is not a known one). */
export const jobMessage = (job) => {
  const m = String((job && job.message) || '');
  for (const [shape, key] of JOB_MESSAGE_SHAPES) {
    const found = shape.exec(m);
    if (found) return t(key, { tool: toolName(found[1].toLowerCase()) });
  }
  return tr(`jobmsg.${m}`, m);
};

export const statusLabel = (st) => tr(`annotate.status.${st}`, st || '?');

export const isHttp = (u) => /^https?:\/\//i.test(String(u || ''));

/** The host of a server address (the address itself when it cannot be read). */
export const serverName = (url) => {
  try { return new URL(url).host; } catch (e) { return String(url); }
};

export const imagesOf = (s) => (Array.isArray(s.images) ? s.images.length : Number(s.images) || 0);

/** Read-only dialog with the tasks of the CVAT server, per project and the loose ones grouped by name. */
export function openCvatAudit() {
  const body = el('div', { class: 'audit-body', 'aria-live': 'polite' }, el('p', { class: 'muted' }, t('audit.loading')));
  const dlg = el('dialog', { class: 'dialog wide', 'aria-label': t('audit.title') },
    el('div', { class: 'dialog-body' },
      el('h2', null, t('audit.title')),
      el('p', { class: 'muted' }, t('audit.intro')),
      body,
      el('div', { class: 'form-actions' }, el('button', { type: 'button', class: 'btn', onclick: () => dlg.close() }, t('common.close')))));
  const taskList = (tasks) => el('ul', { class: 'plain audit-tasks' }, ...tasks.map((x) => el('li', null,
    isHttp(x.url) ? el('a', { href: x.url, target: '_blank', rel: 'noopener' }, x.name || `#${x.id}`) : el('span', null, x.name),
    el('span', { class: 'muted small' }, tn('annotate.images_n', x.size)))));
  const group = (summary, tasks) => el('details', { class: 'audit-group' },
    el('summary', null, ...summary, el('span', { class: 'count' }, tn('audit.tasks_n', tasks.length))),
    tasks.length ? taskList(tasks) : el('p', { class: 'muted small' }, t('audit.no_tasks')));
  function show(data) {
    const inProjects = data.projects.reduce((a, p) => a + p.tasks.length, 0);
    const loose = data.loose.reduce((a, g) => a + g.tasks.length, 0);
    fill(body,
      el('p', { class: 'audit-summary' }, [
        tn('audit.summary_projects', data.projects.length),
        tn('audit.summary_inprojects', inProjects),
        tn('audit.summary_loose', loose),
      ].join(' · ')),
      el('h3', null, t('audit.projects')),
      data.projects.length
        ? el('div', { class: 'audit-groups' }, ...data.projects.map((p) => group([
            el('span', { class: 'audit-name' }, p.name || `#${p.id}`),
            p.linked_to ? el('span', { class: 'badge' }, icon('check', 14), t('audit.linked', { name: p.linked_to.name })) : null,
          ], p.tasks)))
        : el('p', { class: 'muted' }, t('audit.no_projects')),
      el('h3', null, t('audit.loose')),
      el('p', { class: 'muted small' }, t('audit.loose_help')),
      data.loose.length
        ? el('div', { class: 'audit-groups' }, ...data.loose.map((g) => group([el('code', null, g.pattern)], g.tasks)))
        : el('p', { class: 'report-line ok' }, icon('check'), t('audit.no_loose')));
  }
  dlg.addEventListener('close', () => dlg.remove());
  document.body.appendChild(dlg);
  openModal(dlg);
  api('GET', '/cvat/audit').then(show, (err) => fill(body, el('p', { class: 'test-result bad' }, icon('error'), err.message)));
}

/** Dialog to set up, test or forget a tool login. `entry` is the tool entry (or {}); `onChange` runs after a save or forget. */
export function openToolSetup(tool, entry, onChange) {
  entry = entry || {};
  const secretLabel = tool === 'cvat' ? t('annotate.field_password') : t('annotate.field_token');
  const urlInput = el('input', { id: 'tool-url', type: 'text', inputmode: 'url', autocomplete: 'off', value: entry.url || '', placeholder: DEFAULT_URLS[tool] || '', 'aria-describedby': 'tool-url-help' });
  const userInput = tool === 'cvat' ? el('input', { id: 'tool-user', type: 'text', autocomplete: 'off', value: entry.username || '' }) : null;
  const secretInput = el('input', { id: 'tool-secret', type: 'password', autocomplete: 'off', placeholder: entry.configured ? '••••••••' : null, 'aria-describedby': 'tool-secret-help' });
  const message = el('p', { class: 'test-result', role: 'status', 'aria-live': 'polite' });
  const saveBtn = el('button', { type: 'submit', class: 'btn primary' }, t('common.save'));
  const testBtn = el('button', { type: 'button', class: 'btn' }, t('annotate.test'));
  const forgetBtn = el('button', { type: 'button', class: 'btn danger-outline' }, t('annotate.forget'));
  forgetBtn.hidden = !entry.configured;
  const dlg = el('dialog', { class: 'dialog wide', 'aria-label': t('annotate.setup_title', { tool: toolName(tool) }) },
    el('form', { class: 'dialog-body form', method: 'dialog' },
      el('h2', null, t('annotate.setup_title', { tool: toolName(tool) })),
      el('p', { class: 'muted' }, t(`annotate.setup_help.${tool}`)),
      el('div', { class: 'field' }, el('label', { for: 'tool-url' }, t('annotate.field_url')), urlInput,
        el('p', { id: 'tool-url-help', class: 'hint' }, t('annotate.url_help'))),
      userInput ? el('div', { class: 'field' }, el('label', { for: 'tool-user' }, t('annotate.field_username')), userInput) : null,
      el('div', { class: 'field' }, el('label', { for: 'tool-secret' }, secretLabel), secretInput,
        el('p', { id: 'tool-secret-help', class: 'hint' }, entry.configured ? t('annotate.secret_saved') : t('annotate.secret_help'))),
      message,
      el('div', { class: 'form-actions' }, saveBtn, testBtn, forgetBtn,
        el('button', { type: 'button', class: 'btn', onclick: () => dlg.close() }, t('common.cancel')))));

  const values = () => {
    const body = { url: urlInput.value.trim() };
    if (userInput) body.username = userInput.value.trim();
    if (secretInput.value) body.secret = secretInput.value;
    return body;
  };
  const store = async () => {
    const body = values();
    if (!body.url) { urlInput.focus(); throw new Error(t('annotate.url_required')); }
    const saved = await api('PUT', `/annotation-tools/${tool}`, { json: body });
    secretInput.value = '';
    return saved;
  };
  const clearMessage = () => { message.className = 'test-result'; message.textContent = ''; };
  for (const input of [urlInput, userInput, secretInput]) if (input) input.addEventListener('input', clearMessage);   // a message about the old value must not stay
  const busy = (on) => { [saveBtn, testBtn, forgetBtn].forEach((b) => { b.disabled = on; }); };
  dlg.querySelector('form').addEventListener('submit', async (e) => {
    e.preventDefault();
    busy(true);
    try {
      await store();
      if (onChange) await onChange();
      dlg.close();
      toast(t('annotate.saved', { tool: toolName(tool) }), 'success', 3000);
    } catch (err) { toastError(err); } finally { busy(false); }
  });
  /* the connection test's answer in the interface language (by its code), else the server's own words */
  const testText = (r) => {
    const key = r && r.code === 'test_connected' ? 'annotate.test_connected' : r && r.code ? `error.${r.code}` : '';
    if (key && key in I18N) return t(key, errorParams(r.params));
    return (r && r.message) || (r && r.ok ? t('annotate.test_ok') : t('annotate.test_failed'));
  };
  testBtn.addEventListener('click', async () => {
    busy(true);
    message.className = 'test-result';
    message.textContent = t('annotate.testing');
    try {
      await store();
      const r = await api('POST', `/annotation-tools/${tool}/test`);
      message.className = `test-result ${r && r.ok ? 'ok' : 'bad'}`;
      fill(message, icon(r && r.ok ? 'check' : 'error'), testText(r));
      if (onChange) await onChange();
      forgetBtn.hidden = false;
    } catch (err) {
      message.className = 'test-result bad';
      fill(message, icon('error'), err.message);
    } finally { busy(false); }
  });
  forgetBtn.addEventListener('click', async () => {
    const ok = await confirmDialog({ title: t('annotate.forget_title', { tool: toolName(tool) }), text: t('annotate.forget_text'), confirmLabel: t('annotate.forget') });
    if (!ok) return;
    busy(true);
    try {
      await api('DELETE', `/annotation-tools/${tool}`);
      if (onChange) await onChange();
      dlg.close();
      toast(t('annotate.forgotten', { tool: toolName(tool) }), 'success', 3000);
    } catch (err) { toastError(err); } finally { busy(false); }
  });
  dlg.addEventListener('close', () => dlg.remove());
  document.body.appendChild(dlg);
  openModal(dlg);
  urlInput.focus();
}
