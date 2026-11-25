/* The new-project form. */
import { taskPicker } from '../cv/names.js';
import { api } from '../shell/api.js';
import { t } from '../shell/i18n.js';
import { toastError } from '../shell/toast.js';
import { chipsInput } from '../shell/widgets.js';

export function initNew() {
  const form = document.getElementById('new-form');
  const picker = taskPicker('task', 'detection');
  document.getElementById('task-picker').replaceWith(picker.node);
  const chips = chipsInput('classes-input', []);
  document.getElementById('classes-field').replaceWith(chips.node);
  const btn = document.getElementById('create-btn');
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
      location.href = `/projects/${project.id}`;
    } catch (err) {
      btn.disabled = false;
      toastError(err);
    }
  });
  form.elements.name.focus();
}
