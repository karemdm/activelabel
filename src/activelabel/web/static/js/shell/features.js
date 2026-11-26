/* The installed parts (#658, light install #218): FiftyOne is an extra of the Python package and the training
 * packages live in their own environment. GET /api/features says what is there; the dialog shows both, with the
 * command that adds FiftyOne and the removal of the training environment (DELETE /api/training/environment).
 * The buttons that need FiftyOne ask `fiftyoneMissing()` and, when it is missing, say why and how to add it
 * instead of failing when pressed. */
import { api } from './api.js';
import { confirmDialog, openModal } from './dialogs.js';
import { el, fill, icon } from './dom.js';
import { fmtBytes, num, t, tn, tr } from './i18n.js';
import { toast, toastError } from './toast.js';

let current = null;   // the last answer of GET /features (null until it came)
let pending = null;
const buttons = new Set();   // the FiftyOne buttons of the page (#675), marked when the answer comes

/** Reads what is installed (once; `force` reads again). Resolves with the answer, or null when it failed. */
export function loadFeatures(force = false) {
  if (!pending || force) pending = api('GET', '/features').then((f) => { current = f; markAll(); return f; }).catch(() => null);
  return pending;
}

/** The last known answer (null before it came). */
export const features = () => current;

/** Sets the known answer (tests, and after a removal that answered with the new state). */
export const setFeatures = (f) => { current = f; pending = Promise.resolve(f); markAll(); };

/** Whether FiftyOne is known not to be installed (unknown counts as installed: the server still says it). */
export const fiftyoneMissing = () => !!current && !!current.fiftyone && current.fiftyone.installed === false;

/** The command that installs FiftyOne. */
export const fiftyoneInstall = () => (current && current.fiftyone && current.fiftyone.install) || 'pip install "activelabel[fiftyone]"';

/** Copies a text to the clipboard (with the old fallback where the clipboard API is refused). Resolves true/false. */
export async function copyText(text) {
  try { await navigator.clipboard.writeText(text); return true; } catch (e) { /* falls back below */ }
  const area = el('textarea', { class: 'sr-only', 'aria-hidden': 'true', tabindex: '-1' }, text);
  document.body.appendChild(area);
  area.select();
  let ok = false;
  try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
  area.remove();
  return ok;
}

/** A command in a box with a Copy button. */
export const commandBox = (command) => el('div', { class: 'cmd-box' },
  el('code', null, command),
  el('button', { type: 'button', class: 'btn small', onclick: async () => toast(t((await copyText(command)) ? 'features.copied' : 'features.copy_failed'), 'info', 2500) },
    icon('copy', 14), t('features.copy')));

/** Why a FiftyOne button is not available, with the command and a way to the installed parts (#658). */
export const fiftyoneMissingNote = () => el('div', { class: 'fo-missing', id: 'fo-missing' },
  el('p', { class: 'report-line' }, icon('info', 16), t('features.fo_missing')),
  commandBox(fiftyoneInstall()),
  el('button', { type: 'button', class: 'btn small link-btn', onclick: (e) => openFeatures(e.currentTarget) }, t('features.see')));

/** The toast a FiftyOne button gives when pressed without FiftyOne: it stays until closed, so the command can be
 *  read and copied (#675). Returns true when FiftyOne was missing. */
export function sayFiftyoneMissing() {
  if (!fiftyoneMissing()) return false;
  toast(t('features.fo_missing_short', { install: fiftyoneInstall() }), 'info', 0);
  return true;
}

/** Marks one button: dimmed and aria-disabled (not disabled, so a tap still says why) while FiftyOne is missing. */
function mark(button) {
  const off = fiftyoneMissing();
  button.classList.toggle('is-unavailable', off);
  if (off) {
    button.setAttribute('aria-disabled', 'true');
    button.title = t('features.fo_missing_short', { install: fiftyoneInstall() });
  } else if (button.dataset.foMarked) {
    button.removeAttribute('aria-disabled');
    button.removeAttribute('title');
  }
  if (off) button.dataset.foMarked = '1';
}
/* a button made before its panel is in the page is kept; one that was in the page and left it (a redraw) is
   forgotten */
function markAll() {
  for (const b of [...buttons]) {
    if (b.isConnected) b.dataset.foAttached = '1';
    else if (b.dataset.foAttached) { buttons.delete(b); continue; }
    mark(b);
  }
}

/** Registers a button that needs FiftyOne (#675): it is marked now and whenever what is installed is known again.
 *  Returns the button. */
export function fiftyoneButton(button) {
  buttons.add(button);
  mark(button);
  return button;
}

/** A 501 fiftyone_not_installed from the server (the page did not know yet): remember it, mark the buttons and say
 *  how to install, in a toast that stays. Returns true when it was that error. */
export function fiftyoneError(err) {
  if (!err || err.code !== 'fiftyone_not_installed') return false;
  const install = (err.params && err.params.install) || fiftyoneInstall();
  current = { ...(current || {}), fiftyone: { ...((current && current.fiftyone) || {}), installed: false, install } };
  markAll();
  toast(t('features.fo_missing_short', { install }), 'info', 0);
  return true;
}

function fiftyoneCard(fo) {
  const installed = !!fo.installed;
  return el('section', { class: 'card feature-card', 'aria-labelledby': 'feat-fo' },
    el('div', { class: 'feature-head' },
      el('h3', { id: 'feat-fo' }, icon('fiftyone', 20), 'FiftyOne'),
      el('span', { class: `sbadge${installed ? ' sbadge-applied' : ''}` }, icon(installed ? 'check' : 'info', 14),
        installed ? t('features.installed_version', { version: fo.version || '?' }) : t('features.not_installed'))),
    el('p', { class: 'muted' }, t('features.fo_help')),
    el('h4', null, t('features.fo_provides')),
    el('ul', { class: 'plain feature-list' }, ...(fo.provides || []).map((p) => el('li', null, icon(installed ? 'check' : 'close', 14), tr(`features.provides.${p}`, p)))),
    installed ? null : el('h4', null, t('features.fo_how')),
    installed ? null : commandBox(fo.install || fiftyoneInstall()),
    installed ? null : el('p', { class: 'hint' }, t('features.fo_restart')));
}

function environmentCard(env, redraw) {
  const error = el('p', { class: 'tr-form-error', role: 'alert', hidden: true });
  const packages = Object.entries(env.packages || {});
  const remove = el('button', { type: 'button', class: 'btn danger-outline', id: 'feat-env-remove', onclick: async () => {
    const ok = await confirmDialog({
      title: t('features.env_remove_title'),
      text: t('features.env_remove_text', { size: fmtBytes(env.bytes || 0) }),
      confirmLabel: t('features.env_remove'),
    });
    if (!ok) return;
    remove.disabled = true;
    try {
      const after = await api('DELETE', '/training/environment', { json: { confirm: true } });
      setFeatures(after);
      toast(t('features.env_removed', { size: fmtBytes(env.bytes || 0) }), 'success', 5000);
      redraw();
    } catch (err) {
      remove.disabled = false;
      error.textContent = err.message || String(err);
      error.hidden = false;
    }
  } }, icon('trash', 16), t('features.env_remove'));
  return el('section', { class: 'card feature-card', 'aria-labelledby': 'feat-env' },
    el('div', { class: 'feature-head' },
      el('h3', { id: 'feat-env' }, icon('chart', 20), t('features.env_title')),
      el('span', { class: `sbadge${env.exists ? ' sbadge-applied' : ''}` }, icon(env.exists ? 'check' : 'info', 14),
        env.exists ? t('features.env_size', { size: fmtBytes(env.bytes || 0) }) : t('features.not_installed'))),
    el('p', { class: 'muted' }, t(env.exists ? 'features.env_help' : 'features.env_none')),
    env.exists ? el('dl', { class: 'facts' },
      el('div', null, el('dt', null, t('features.env_path')), el('dd', null, el('code', null, env.path || ''))),
      el('div', null, el('dt', null, t('features.env_packages')), el('dd', null, num(packages.length)))) : null,
    env.exists && packages.length ? el('details', { class: 'feature-packages' },
      el('summary', null, tn('features.env_packages_n', packages.length)),
      el('ul', { class: 'plain feature-list mono-list' }, ...packages.sort(([a], [b]) => a.localeCompare(b)).map(([name, version]) => el('li', null, `${name} ${version}`)))) : null,
    env.exists ? el('div', { class: 'form-actions' }, remove) : null,
    error);
}

/** The "Installed parts" dialog. `trigger` gets the focus back when it closes. */
export async function openFeatures(trigger = null) {
  const body = el('div', { class: 'feature-cards', 'aria-live': 'polite' }, el('p', { class: 'muted' }, t('common.loading')));
  const dlg = el('dialog', { class: 'dialog wide', 'aria-labelledby': 'feat-title' },
    el('div', { class: 'dialog-body' },
      el('h2', { id: 'feat-title' }, t('features.title')),
      el('p', { class: 'muted' }, t('features.intro')),
      body));
  const draw = () => {
    const f = features();
    if (!f) { fill(body, el('p', { class: 'tr-form-error', role: 'alert' }, t('common.load_failed'))); return; }
    fill(body, fiftyoneCard(f.fiftyone || {}), environmentCard(f.training_environment || {}, draw));
  };
  dlg.addEventListener('close', () => { dlg.remove(); if (trigger && trigger.isConnected) trigger.focus(); });
  document.body.appendChild(dlg);
  openModal(dlg);
  try { await loadFeatures(true); } catch (err) { toastError(err); }
  draw();
}
