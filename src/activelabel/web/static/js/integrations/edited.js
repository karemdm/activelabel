/* Images edited in the FiftyOne App and not in the project yet (#700): the count the server gives with the
 * sessions (GET /annotate, `edited_count`), kept in `ctx.fiftyoneEdited`, and the line that shows it in the
 * project header and in the FiftyOne tab. The server answers from a cheap fingerprint of the session images, so
 * asking again whenever the person comes back to this window (they were editing in the other tab) is fine. */
import { api } from '../shell/api.js';
import { el, icon } from '../shell/dom.js';
import { t, tn } from '../shell/i18n.js';

/** Fewest milliseconds between two reads caused by coming back to the window. */
export const REFRESH_GAP = 3000;

/** How many images are edited in FiftyOne and not in the project yet, in what the server answered with. */
export const editedCount = (data) => Math.max(0, Number(data && data.edited_count) || 0);

/** The session to look at for the edits: the one that holds most of them (the newest of equals).
 *  `data` is the answer of GET /annotate (sessions newest first, `edited` the count by session id).
 *  Returns the session or null. */
export function editedSession(data) {
  const by = (data && data.edited) || {};
  let best = null;
  (data && data.sessions || []).forEach((s) => {
    if ((by[s.id] || 0) > (best ? by[best.id] : 0)) best = s;
  });
  return best;
}

/** How many images of one session are edited in FiftyOne and not in the project yet (0 when none). */
export const sessionEdited = (data, sessionId) => Number(((data && data.edited) || {})[sessionId]) || 0;

/** The line "N images edited in FiftyOne, not in the project yet" with the button that goes to the session, or
 *  null when nothing is edited. `onReview` opens the session (the button is left out without it). */
export function editedNotice(data, { id, onReview = null } = {}) {
  const n = editedCount(data);
  if (!n) return null;
  return el('p', { class: 'fo-edited', id: id || null, role: 'status' },
    icon('fiftyone', 16),
    el('span', null, tn('annotate.edited_n', n)),
    onReview ? el('button', { type: 'button', class: 'btn small', onclick: onReview }, t('annotate.edited_review')) : null);
}

/** Reads the sessions and the count, keeps them in `ctx.fiftyoneEdited` and tells whoever draws them
 *  (`ctx.editedWatchers`). `page` is the project page. Returns the data, or null when it could not be read. */
export async function refreshEdited(page) {
  const { ctx, projectId } = page;
  let data;
  try {
    data = await api('GET', `/projects/${projectId}/annotate`);
  } catch (err) {
    if (err.status !== 404) throw err;
    return null;
  }
  ctx.fiftyoneEdited = data || { sessions: [], edited_count: 0, edited: {} };
  (ctx.editedWatchers || []).forEach((draw) => draw(ctx.fiftyoneEdited));
  return ctx.fiftyoneEdited;
}

/** Reads the count now and again each time the person comes back to this window. Returns the stop function. */
export function watchEdited(page, { now = () => Date.now(), target = window } = {}) {
  let last = 0;
  const again = () => {
    if (target.document && target.document.visibilityState === 'hidden') return;
    const at = now();
    if (at - last < REFRESH_GAP) return;
    last = at;
    refreshEdited(page).catch(() => { /* the count is a hint: the next visit asks again */ });
  };
  target.addEventListener('focus', again);
  target.document.addEventListener('visibilitychange', again);
  return () => {
    target.removeEventListener('focus', again);
    target.document.removeEventListener('visibilitychange', again);
  };
}
