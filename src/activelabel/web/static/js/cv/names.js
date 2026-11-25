/* Computer-vision names: task types, curation issues, data-split parts, and the task picker and badge. */
import { el, fill, icon } from '../shell/dom.js';
import { t, tr } from '../shell/i18n.js';

export const TASKS = ['classification', 'detection', 'segmentation', 'keypoints'];

export const ISSUES = ['blurry', 'dark', 'bright', 'low_contrast', 'small', 'odd_aspect', 'grayscale', 'near_duplicate'];

export const CHUNK = 25;

export const SKIP_FILES = new Set(['.ds_store', 'thumbs.db']);

export const taskName = (task) => tr(`task.${task}`, task || '?');

export const issueName = (issue) => tr(`issue.${issue}`, issue);

/* one image's tag (singular: "Blurry" / "Borrada") */
export const issueTag = (issue) => tr(`tag.${issue}`, issueName(issue));

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
          { class: `task-card${task === current ? ' selected' : ''}`, for: id },
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

export const taskBadge = (task) => el('span', { class: `badge task-${task}` }, icon(task, 14), taskName(task));
