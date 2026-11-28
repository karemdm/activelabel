import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body></body></html>');
for (const [name, value] of Object.entries({ window, document: window.document, Node: window.Node })) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const edited = await import('../../../src/activelabel/web/static/js/integrations/edited.js');
const { REFRESH_GAP, editedCount, editedNotice, editedSession, refreshEdited, sessionEdited, watchEdited } = edited;

setMessages({
  'annotate.edited_n_one': '{n} image edited in FiftyOne, not in the project yet',
  'annotate.edited_n_other': '{n} images edited in FiftyOne, not in the project yet',
  'annotate.edited_review': 'Review and send',
}, 'en');

const DATA = { edited_count: 3, edited: { a: 1, b: 2 }, sessions: [{ id: 'a' }, { id: 'b' }, { id: 'c' }] };

test('the count is a whole number, 0 when the server gave none', () => {
  assert.equal(editedCount(DATA), 3);
  assert.equal(editedCount({}), 0);
  assert.equal(editedCount(null), 0);
  assert.equal(editedCount({ edited_count: -2 }), 0);
});

test('the session to look at holds most edits; the newest wins a tie; none when nothing is edited', () => {
  assert.equal(editedSession(DATA).id, 'b');
  assert.equal(editedSession({ edited: { a: 2, b: 2 }, sessions: [{ id: 'a' }, { id: 'b' }] }).id, 'a');
  assert.equal(editedSession({ edited: {}, sessions: [{ id: 'a' }] }), null);
  assert.equal(editedSession(null), null);
  assert.equal(sessionEdited(DATA, 'b'), 2);
  assert.equal(sessionEdited(DATA, 'c'), 0);
});

test('the notice says how many, in singular and plural, and is left out when there are none', () => {
  assert.equal(editedNotice({ edited_count: 0 }), null);
  assert.equal(editedNotice(null), null);
  assert.equal(editedNotice({ edited_count: 1 }).textContent.trim(), '1 image edited in FiftyOne, not in the project yet');
  const clicks = [];
  const notice = editedNotice(DATA, { id: 'x', onReview: () => clicks.push(1) });
  assert.equal(notice.id, 'x');
  assert.match(notice.textContent, /3 images edited in FiftyOne/);
  notice.querySelector('button').click();
  assert.deepEqual(clicks, [1]);
  assert.equal(editedNotice(DATA).querySelector('button'), null);
});

function fakeApi(answer) {
  const calls = [];
  globalThis.fetch = async (url) => {
    calls.push(url);
    return typeof answer === 'function' ? answer() : { ok: true, status: 200, headers: new Map(), json: async () => answer, text: async () => JSON.stringify(answer) };
  };
  return calls;
}

test('refreshEdited keeps the answer in ctx and tells the watchers', async () => {
  const calls = fakeApi(DATA);
  const seen = [];
  const ctx = { editedWatchers: [(d) => seen.push(d.edited_count)] };
  const data = await refreshEdited({ ctx, projectId: 7 });
  assert.equal(data.edited_count, 3);
  assert.equal(ctx.fiftyoneEdited.edited_count, 3);
  assert.deepEqual(seen, [3]);
  assert.match(String(calls[0]), /\/projects\/7\/annotate$/);
});

const settle = (ms = 40) => new Promise((resolve) => setTimeout(resolve, ms));

test('coming back to the window reads the count again, not more than once in a few seconds, and not while hidden', async () => {
  const calls = fakeApi(DATA);
  let clock = 100000;
  const dom = new JSDOM('<!doctype html><body></body>').window;
  const stop = watchEdited({ ctx: { editedWatchers: [] }, projectId: 1 }, { now: () => clock, target: dom });
  dom.dispatchEvent(new dom.Event('focus'));
  dom.dispatchEvent(new dom.Event('focus'));   // too soon
  await settle();
  assert.equal(calls.length, 1);
  clock += REFRESH_GAP + 1;
  dom.dispatchEvent(new dom.Event('focus'));
  await settle();
  assert.equal(calls.length, 2);
  stop();
  clock += REFRESH_GAP + 1;
  dom.dispatchEvent(new dom.Event('focus'));
  await settle();
  assert.equal(calls.length, 2);
});
