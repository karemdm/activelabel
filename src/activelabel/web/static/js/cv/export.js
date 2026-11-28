/* Step 4, Data split: the buttons that download a dataset version as a YOLO or COCO zip (docs/DATASETS.md). */
import { api } from '../shell/api.js';
import { once } from '../shell/dialogs.js';
import { el, icon } from '../shell/dom.js';
import { t } from '../shell/i18n.js';
import { toast, toastError } from '../shell/toast.js';

/** The export formats a version of this task can have: neither format holds a number per image (regression),
 *  and COCO has no label for a whole image (classification). */
export function exportFormats(task) {
  if (task === 'regression') return [];
  return task === 'classification' ? ['yolo'] : ['yolo', 'coco'];
}

/** Where the zip of a version is served. */
export const exportUrl = (projectId, version, format) => `/api/projects/${projectId}/datasets/${version}/export?format=${format}`;

/** Asks the server whether the version can be exported (it explains why not), then starts the download. */
async function download(projectId, version, format) {
  await api('GET', `/projects/${projectId}/datasets/${version}/export/check?format=${format}`);
  toast(t('ds.exporting', { n: version, format: format.toUpperCase() }), 'info', 4000);
  const link = el('a', { href: exportUrl(projectId, version, format), download: '' });
  document.body.appendChild(link);
  link.click();
  link.remove();
}

/** The row "Export: YOLO, COCO" of a version card (nothing when the task has no format). */
export function exportRow(projectId, v) {
  const formats = exportFormats(v.type);
  if (!formats.length) return null;
  return el('div', { class: 'ds-view ds-export' },
    el('span', { class: 'muted', title: t('ds.export_help') }, t('ds.export')),
    ...formats.map((format) => el('button', {
      type: 'button', class: 'btn small', 'data-format': format, disabled: !!v.incomplete,
      'aria-label': t('ds.export_aria', { n: v.version, format: format.toUpperCase() }),
      onclick: once(async () => { try { await download(projectId, v.version, format); } catch (err) { toastError(err); } }),
    }, icon('download', 14), format.toUpperCase())));
}
