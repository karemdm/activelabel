/* Step 1, Images: after an upload to a classification project, offers the folder names of the images as their
 * classes, with a preview of the classes found (#568). The check and the apply are the annotation routes. */
import { api } from '../shell/api.js';
import { el, fill, icon } from '../shell/dom.js';
import { num, t, tn } from '../shell/i18n.js';
import { toastError } from '../shell/toast.js';
import { problemRow } from '../shell/widgets.js';

/** Whether an upload can hold class folders: a name inside a folder, or a zip (its folders are read by the server).
 *  @param {{name: string}[]} entries The uploaded files, named as the browser sent them.
 *  @returns {boolean} True when it is worth asking the server. */
export function mayHaveClassFolders(entries) {
  return entries.some((e) => {
    const name = String(e.name || '').replaceAll('\\', '/');
    return name.split('/').filter(Boolean).length > 1 || /\.zip$/i.test(name);
  });
}

/** The classes of a check report, with how many images each one labels and whether the project lacks it.
 *  @param {object} report The answer of the check (`classes_found`, `classes_new`, `counts.labels_per_class`).
 *  @returns {{name: string, count: number, isNew: boolean}[]} The classes in use, in the order of the report. */
export function classRows(report) {
  const perClass = (report.counts || {}).labels_per_class || {};
  const fresh = new Set(report.classes_new || []);
  return (report.classes_found || [])
    .filter((c) => (perClass[c] || 0) > 0)
    .map((c) => ({ name: c, count: perClass[c], isNew: fresh.has(c) }));
}

/** Asks the server what the folder names of the project's images would give.
 *  @param {number|string} projectId The project.
 *  @returns {Promise<object|null>} The report to show, or null when the images are not in class folders. */
export async function checkClassFolders(projectId) {
  const report = await api('POST', `/projects/${projectId}/images/class-folders`);
  return report && report.found ? report : null;
}

/** The card that previews the classes found in the folders and offers to use them.
 *  @param {{projectId: number|string, report: object, onApplied: () => Promise<void>|void}} options The project, the
 *    report of the check, and what to do once the classes were imported (read the project again).
 *  @returns {HTMLElement} The card. */
export function classFoldersCard({ projectId, report, onApplied }) {
  const rows = classRows(report);
  const counts = report.counts || {};
  const left = report.split_folders || [];
  const card = el('div', { class: 'card report-card', id: 'class-folders' });
  const buttons = [];

  const finish = async (action) => {
    buttons.forEach((b) => { b.disabled = true; });
    try {
      if (action === 'discard') {
        await api('DELETE', `/projects/${projectId}/annotations/${report.token}`);
        card.remove();
        return;
      }
      const done = await api('POST', `/projects/${projectId}/annotations/${report.token}/apply`, { json: { action: 'import', mode: 'merge' } });
      fill(card,
        el('p', { class: 'report-line ok' }, icon('check'),
          el('strong', null, tn('images.folders_labelled', (done || {}).images_added || 0))),
        (done || {}).classes_added && done.classes_added.length
          ? el('p', { class: 'muted' }, tn('images.folders_new_classes', done.classes_added.length), ': ', done.classes_added.join(', '))
          : null);
      await onApplied();
    } catch (err) {
      buttons.forEach((b) => { b.disabled = false; });
      toastError(err);
    }
  };
  const use = el('button', { type: 'button', class: 'btn primary', id: 'class-folders-use', onclick: () => finish('import') }, t('images.folders_use'));
  const skip = el('button', { type: 'button', class: 'btn', id: 'class-folders-skip', onclick: () => finish('discard') }, t('images.folders_skip'));
  buttons.push(use, skip);

  fill(card,
    el('h3', null, t('images.folders_title')),
    el('p', { class: 'muted' }, t('images.folders_text')),
    el('dl', { class: 'facts' },
      el('div', null, el('dt', null, t('ann.classes_found')), el('dd', null, num(rows.length))),
      el('div', null, el('dt', null, t('ann.matched')), el('dd', null, num(counts.images_matched || 0))),
      el('div', null, el('dt', null, t('ann.unlabeled')), el('dd', null, num(counts.images_without_labels || 0)))),
    el('ul', { class: 'chips static' }, ...rows.map((r) => el('li', { class: `chip static${r.isNew ? ' is-new' : ''}` },
      el('span', null, r.name),
      el('span', { class: 'count' }, num(r.count)),
      r.isNew ? el('span', { class: 'new-mark' }, t('ann.new')) : null))),
    rows.some((r) => r.isNew) && rows.some((r) => !r.isNew) ? el('p', { class: 'hint' }, t('images.folders_existing')) : null,
    left.length ? el('p', { class: 'hint' }, t('images.folders_split', { names: left.join(', ') })) : null,
    (report.problems || []).length ? el('ul', { class: 'plain problems' }, ...report.problems.map(problemRow)) : null,
    el('div', { class: 'row-actions' }, use, skip));
  return card;
}
