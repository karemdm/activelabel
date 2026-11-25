/* Short messages at the bottom of the page (or inside the open modal dialog, which would hide them otherwise). */
import { el, icon } from './dom.js';
import { t } from './i18n.js';

/** Where toasts go: the page's own box, or one inside the open modal dialog (a modal sits in the top layer
 *  above everything else, and makes the rest of the page inert, so a toast outside it would be unseen). */
export function toastBox() {
  const modals = document.querySelectorAll('dialog:modal');
  const dialog = modals.length ? modals[modals.length - 1] : null;
  if (!dialog) return document.getElementById('toasts');
  let box = dialog.querySelector(':scope > .toasts');
  if (!box) {
    box = el('div', { class: 'toasts', 'aria-live': 'polite' });
    dialog.appendChild(box);
  }
  return box;
}

export function toast(message, kind = 'info', ms = 7000) {
  const box = toastBox();
  if (!box) return;
  const same = typeof message === 'string' ? [...box.children].find((n) => n.dataset.kind === kind && n.dataset.text === message) : null;
  if (same) { clearTimeout(same.timer); same.timer = ms ? setTimeout(() => same.remove(), ms) : 0; return; }   // the same message is shown once
  const node = el(
    'div',
    { class: `toast toast-${kind}`, 'data-kind': kind, 'data-text': typeof message === 'string' ? message : '', role: kind === 'error' ? 'alert' : 'status' },
    icon(kind === 'error' ? 'error' : kind === 'success' ? 'check' : 'info'),
    el('div', { class: 'toast-text' }, message),
    el('button', { type: 'button', class: 'icon-btn', 'aria-label': t('common.close'), onclick: () => node.remove() }, icon('close', 16)),
  );
  box.appendChild(node);
  if (ms) node.timer = setTimeout(() => node.remove(), ms);
}

/* the red toast with its icon already says it is an error: no "Error:" in front of the server's own words */
export const toastError = (err) => toast(String(err && err.message ? err.message : err), 'error', 10000);
