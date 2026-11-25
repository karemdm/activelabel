/* The home page: the list of projects. */
import { annotationStatus, taskBadge } from '../cv/names.js';
import { api } from '../shell/api.js';
import { el, fill } from '../shell/dom.js';
import { fmtDate, num, t, tn } from '../shell/i18n.js';
import { toastError } from '../shell/toast.js';
import { progressBar } from '../shell/widgets.js';

export async function initProjects() {
  const root = document.getElementById('projects-root');
  let list;
  try {
    list = await api('GET', '/projects');
  } catch (err) {
    fill(root, el('p', { class: 'muted' }, t('common.load_failed')));
    toastError(err);
    return;
  }
  const projects = Array.isArray(list) ? list : (list && list.items) || [];
  if (!projects.length) {
    fill(root,
      el('div', { class: 'empty card' },
        el('h2', null, t('projects.empty_title')),
        el('p', null, t('projects.empty_text')),
        el('div', { class: 'empty-actions' },
          el('a', { class: 'btn primary big', href: '/projects/new' }, t('projects.create_first')),
          el('a', { class: 'btn big', href: '/projects/import' }, t('projects.import_first')))));
    return;
  }
  fill(root, el('ul', { class: 'project-grid' }, ...projects.map(projectCard)));
}

export function projectCard(p) {
  const c = p.counts || {};
  return el('li', null,
    el('a', { class: 'project-card card', href: `/projects/${p.id}` },
      el('div', { class: 'project-card-top' },
        el('h2', null, p.name),
        el('span', { class: 'project-id' }, `#${p.id}`)),
      taskBadge(p.task),
      p.description ? el('p', { class: 'project-desc' }, p.description) : null,
      el('dl', { class: 'facts' },
        el('div', null, el('dt', null, t('project.images')), el('dd', null, `${tn('status.images_active', c.active || 0)}${c.excluded ? ` · ${tn('status.excluded', c.excluded)}` : ''}`)),
        el('div', null, el('dt', null, t('project.annotations')), el('dd', null, annotationStatus(c)))),
      c.active > 0 ? progressBar(Math.min(c.annotated || 0, c.active), c.active, t('project.coverage', { done: num(c.annotated || 0), total: num(c.active) })) : null,
      el('p', { class: 'muted project-created' }, `${t('project.created')} ${fmtDate(p.created)}`)));
}
