/* The new-project form. */
import { taskPicker } from '../cv/names.js';
import { api } from '../shell/api.js';
import { t } from '../shell/i18n.js';
import { toastError } from '../shell/toast.js';
import { chipsInput } from '../shell/widgets.js';

const DRAFT_KEY = 'activelabel.new-project-draft';

/** The unsent form kept for this tab (switching the language reloads the page, #637); null when there is none. */
export function readDraft(storage) {
  try {
    const draft = JSON.parse((storage && storage.getItem(DRAFT_KEY)) || 'null');
    return draft && typeof draft === 'object' && !Array.isArray(draft) ? draft : null;
  } catch {
    return null;   // blocked storage or a damaged value: the form just starts empty
  }
}

/** Keep the form for this tab, or forget it (`draft` null or with nothing typed). */
export function writeDraft(storage, draft) {
  try {
    if (!storage) return;
    const typed = draft && (draft.name || draft.description || (draft.classes || []).length);
    if (typed) storage.setItem(DRAFT_KEY, JSON.stringify(draft));
    else storage.removeItem(DRAFT_KEY);
  } catch { /* private window or full storage: nothing is kept */ }
}

function tabStorage() {
  try { return window.sessionStorage; } catch { return null; }
}

export function initNew() {
  const form = document.getElementById('new-form');
  const storage = tabStorage();
  const draft = readDraft(storage) || {};
  const picker = taskPicker('task', 'detection');
  if ([...picker.node.querySelectorAll('input')].some((input) => input.value === draft.task)) picker.set(draft.task);
  document.getElementById('task-picker').replaceWith(picker.node);
  const chips = chipsInput('classes-input', Array.isArray(draft.classes) ? draft.classes.map(String) : []);
  document.getElementById('classes-field').replaceWith(chips.node);
  if (typeof draft.name === 'string') form.elements.name.value = draft.name;
  if (typeof draft.description === 'string') form.elements.description.value = draft.description;
  const btn = document.getElementById('create-btn');
  let created = false;
  /* the language links reload the page: keep what was typed until the project is created */
  window.addEventListener('pagehide', () => {
    if (created) return;
    writeDraft(storage, {
      name: form.elements.name.value, description: form.elements.description.value, task: picker.get(), classes: chips.get(),
    });
  });
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const name = form.elements.name.value.trim();
    if (!name) {
      toastError(new Error(t('new.name_required')));
      form.elements.name.focus();
      return;
    }
    btn.disabled = true;
    try {
      const project = await api('POST', '/projects', {
        json: { name, description: form.elements.description.value.trim(), task: picker.get(), classes: chips.get() },
      });
      created = true;
      writeDraft(storage, null);
      location.href = `/projects/${project.id}`;
    } catch (err) {
      btn.disabled = false;
      toastError(err);
    }
  });
  form.elements.name.focus();
}
