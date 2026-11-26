/* Computer-vision names: task types, curation issues, data-split parts, and the task picker and badge. */
import { el, fill, icon } from '../shell/dom.js';
import { t, tn, tr } from '../shell/i18n.js';

export const TASKS = ['classification', 'detection', 'segmentation', 'keypoints'];

export const ISSUES = ['blurry', 'dark', 'bright', 'low_contrast', 'small', 'odd_aspect', 'grayscale', 'near_duplicate'];

export const CHUNK = 25;

export const SKIP_FILES = new Set(['.ds_store', 'thumbs.db']);

export const taskName = (task) => tr(`task.${task}`, task || '?');

export const issueName = (issue) => tr(`issue.${issue}`, issue);

/* one image's tag (singular: "Blurry" / "Borrada") */
export const issueTag = (issue) => tr(`tag.${issue}`, issueName(issue));

/** The number in the automatic name of the next task, as the server gives it (`default_name` in tasks_store.py):
 *  one more than the highest "Task N" among the names of the tasks, not among their ids. */
export const nextTaskNumber = (tasks) =>
  tasks.reduce((highest, x) => {
    const match = /^task (\d+)$/i.exec(x.name || '');
    return match ? Math.max(highest, Number(match[1])) : highest;
  }, 0) + 1;

export const partName = (p) => (p === 'train' ? t('ds.part_train') : p === 'val' ? t('ds.part_val') : p === 'test' ? t('ds.part_test') : String(p || ''));

/** Four large selectable cards for the task. Returns {node, get, set}. */
export function taskPicker(name, value) {
  let current = value || 'detection';
  const node = el('div', { class: 'task-cards', role: 'radiogroup' });
  const draw = () => {
    fill(node,
      ...TASKS.map((task) => {
        const id = `${name}-${task}`;
        const input = el('input', { type: 'radio', name, id, value: task, class: 'sr-only' });
        input.checked = task === current;
        input.addEventListener('change', () => {
          current = task;
          draw();
          node.querySelector(`input[value="${task}"]`).focus();
        });
        return el(
          'label',
          { class: `task-card task-${task}${task === current ? ' selected' : ''}`, for: id },
          input,
          icon(task, 28),
          el('span', { class: 'task-card-title' }, taskName(task)),
          el('span', { class: 'task-card-help' }, t(`task.${task}.help`)),
        );
      }),
    );
  };
  draw();
  return { node, get: () => current, set: (v) => { current = v; draw(); } };
}

/* the type badge: the task colour (app.css .task-<type>) always with its icon and its name (#621) */
export const taskBadge = (task) => el('span', { class: `badge task-badge task-${task}` }, icon(task, 14), taskName(task));

/** How far the annotation of a project is (its counts, as the API gives them), in a few words. */
export function annotationStatus(c) {
  if (c.annotated > 0) return tn('status.annotated', c.annotated);
  return c.empty > 0 ? tn('status.empty', c.empty) : t('status.annotations_none');
}

/** A device (``auto``, ``cpu``, ``cuda:N``, ``mps``) as the training dialog and the Pre-annotate tab name it. */
export const deviceName = (d) => (d.startsWith('cuda:') ? t('tr.device_gpu', { n: d.slice(5) }) : tr(`tr.device_${d}`, d));
