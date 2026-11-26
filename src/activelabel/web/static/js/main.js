/* activelabel web interface: native ES modules, vanilla JS, no dependencies. Talks to the JSON API under /api.
 * This entry sets the texts, names the error params that need the task, tool and status names, and starts
 * the page the server rendered (body[data-page]). */
import { setMessages, lower } from './shell/i18n.js';
import { initTheme } from './shell/theme.js';
import { nameErrorParams } from './shell/api.js';
import { statusLabel, toolName } from './integrations/tools.js';
import { partName, taskName } from './cv/names.js';
import { initProject } from './cv/project.js';
import { initQueue } from './cv/queue.js';
import { initProjects } from './pages/projects.js';
import { initNew } from './pages/new.js';
import { initImport } from './pages/import.js';

const VL = window.VL || {};
setMessages(window.I18N || {}, VL.locale || 'en');
/* error params named with the task, tool and status names (the API module cannot import them) */
nameErrorParams({
  parts: (p) => lower(partName(p)),
  tool: (tool) => toolName(tool),
  task: (task) => lower(taskName(task)),
  type: (task) => lower(taskName(task)),
  kind: (kind) => lower(taskName(kind)),
  from_task: (task) => lower(taskName(task)),
  to_task: (task) => lower(taskName(task)),
  status: (status) => lower(statusLabel(status)),
});

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initQueue();
  const page = document.body.dataset.page;
  if (page === 'projects') initProjects();
  else if (page === 'new') initNew();
  else if (page === 'import') initImport();
  else if (page === 'project') initProject();
});
