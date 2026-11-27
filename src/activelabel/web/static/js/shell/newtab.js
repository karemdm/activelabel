/* A tab for a tool that takes time to start (the FiftyOne App). Browsers only allow a new tab straight after a
 * click, so the tab is opened at the click, shows a waiting line, and goes to the address when the job ends:
 * the job's own end is too late (a popup blocker stops a window opened then). When there is no tab (blocked) the
 * address is offered as a link that stays until closed. */
import { el } from './dom.js';
import { t } from './i18n.js';
import { toast } from './toast.js';

/** Opens a blank tab that says `waiting` (call it from the click handler itself, before any `await`).
 *  Returns the tab, or null when the browser did not give one. */
export function reserveTab(waiting = '') {
  let tab = null;
  try { tab = window.open('', '_blank'); } catch (err) { tab = null; }
  if (!tab) return null;
  try {
    tab.opener = null;  // the tab must not reach back into ActiveLabel
    tab.document.title = waiting || 'FiftyOne';
    tab.document.body.textContent = waiting;
  } catch (err) { /* the tab stays blank */ }
  return tab;
}

/** Closes a reserved tab that will not be used (the job failed, was stopped or never started). */
export function dropTab(tab) {
  try { if (tab && !tab.closed) tab.close(); } catch (err) { /* already gone */ }
}

/** Shows `url` in the reserved tab, or in a new one, or, when the browser blocks both, in a toast with a link that
 *  stays until it is closed. Returns whether a tab now shows the address. */
export function showInTab(tab, url) {
  try {
    if (tab && !tab.closed) { tab.location.href = url; return true; }
  } catch (err) { /* the tab went away: try a new one */ }
  let fresh = null;
  try { fresh = window.open(url, '_blank'); } catch (err) { fresh = null; }
  if (fresh) {
    try { fresh.opener = null; } catch (err) { /* cross-origin already */ }
    return true;
  }
  toast(el('span', null, `${t('cur.fo_popup')} `,
    el('a', { href: url, target: '_blank', rel: 'noopener', 'data-open-link': 'true' }, url)), 'info', 0);
  return false;
}
