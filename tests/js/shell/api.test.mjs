import test from 'node:test';
import assert from 'node:assert/strict';
import { loadApp, messages } from '../helpers/load-app.mjs';
import { api, jobFailure, nameErrorParams, errorParams } from '../../../src/activelabel/web/static/js/shell/api.js';
import { setMessages } from '../../../src/activelabel/web/static/js/shell/i18n.js';

const en = await loadApp('en');
const pt = await loadApp('pt');

test('an error with a known code is translated by its error.<code> key, with the params filled in', () => {
  const data = { code: 'project_name_too_long', params: { max: 100 } };
  assert.equal(en.errorMessage(data, 400), en.t('error.project_name_too_long', { max: '100' }));
  assert.match(en.errorMessage(data, 400), /100/);
  assert.equal(pt.errorMessage(data, 400), messages('pt')['error.project_name_too_long'].replace('{max}', '100'));
});

test('the code and params may also come inside detail (older answers)', () => {
  const data = { detail: { code: 'project_not_found' } };
  assert.equal(en.errorMessage(data, 404), messages('en')['error.project_not_found']);
});

test('numbers in params follow the language number format', () => {
  const data = { code: 'too_many_classes', params: { count: 1500, max: 1000 } };
  assert.match(en.errorMessage(data, 400), /1,500/);
  assert.match(pt.errorMessage(data, 400), /1\.500/);
});

test('an unknown code falls back to the server text, never to the raw key', () => {
  const answer = en.errorMessage({ code: 'brand_new_code', detail: 'Server says no' }, 400);
  assert.equal(answer, 'Server says no');
  assert.ok(!answer.includes('error.'));
});

test('without a code: detail text, then a translated sentence with the status (never a bare "HTTP 500")', () => {
  assert.equal(en.errorMessage({ detail: 'plain text' }, 500), 'plain text');
  assert.equal(en.errorMessage(null, 502), en.t('error.http_status', { status: '502' }));
  assert.equal(pt.errorMessage({}, 500), messages('pt')['error.http_status'].replace('{status}', '500'));
  assert.ok(!/^HTTP \d+$/.test(en.errorMessage({}, 418)));
});

test('an unexpected server error and a FiftyOne database that did not answer are translated by their codes', () => {
  assert.equal(pt.errorMessage({ code: 'internal_error', detail: 'x', params: {} }, 500), messages('pt')['error.internal_error']);
  assert.equal(en.errorMessage({ code: 'fiftyone_unavailable', detail: 'x', params: {} }, 503), messages('en')['error.fiftyone_unavailable']);
});

test('a list in detail is a validation answer: "field: problem" in the interface language', () => {
  const data = { detail: [{ type: 'missing', loc: ['body', 'name'], msg: 'Field required' }] };
  assert.equal(en.errorMessage(data, 422), `name: ${messages('en')['error.v.missing']}`);
  assert.equal(pt.errorMessage(data, 422), `name: ${messages('pt')['error.v.missing']}`);
});

test('a validation problem with an unknown type keeps the server message', () => {
  const entry = { type: 'weird_type', loc: ['body', 'x'], msg: 'Weird problem' };
  assert.equal(en.validationText(entry), 'x: Weird problem');
});

test('errorParams joins lists, skips objects and formats numbers', () => {
  const out = en.errorParams({ a: [1, 2000], b: { nested: 1 }, c: 'text', d: 12345 });
  // compared as JSON: the object comes from the jsdom window, whose Object.prototype is another one
  assert.equal(JSON.stringify(out), JSON.stringify({ a: '1, 2,000', c: 'text', d: '12,345' }));
});

test('errorParams names known ids in the interface language', () => {
  assert.equal(en.errorParams({ task: 'classification' }).task, 'classification');
  assert.equal(pt.errorParams({ task: 'classification' }).task, 'classificação');
  assert.equal(en.errorParams({ runs: 3 }).runs, 'r3');
});

/** Replaces fetch for one call of fn; returns what fn returned (or threw). */
async function withFetch(answer, fn) {
  const real = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, init) => {
    calls.push([url, init]);
    if (answer instanceof Error) throw answer;
    return { ok: answer.status < 400, status: answer.status, text: async () => answer.body };
  };
  try {
    return { result: await fn(), calls };
  } catch (err) {
    return { error: err, calls };
  } finally {
    globalThis.fetch = real;
  }
}

test('api sends JSON under /api and returns the parsed answer', async () => {
  const { result, calls } = await withFetch({ status: 200, body: '{"id": 3}' }, () => api('POST', '/projects', { json: { name: 'x' } }));
  assert.deepEqual(result, { id: 3 });
  assert.equal(calls[0][0], '/api/projects');
  assert.equal(calls[0][1].method, 'POST');
  assert.equal(calls[0][1].headers['Content-Type'], 'application/json');
  assert.equal(calls[0][1].body, '{"name":"x"}');
});

test('api turns an error answer into an Error with status, code, params and the translated text', async () => {
  setMessages({ 'error.project_not_found': 'No project {id}' }, 'en');
  const body = JSON.stringify({ code: 'project_not_found', detail: 'Project 7 not found', params: { id: 7 } });
  const { error } = await withFetch({ status: 404, body }, () => api('GET', '/projects/7'));
  assert.equal(error.message, 'No project 7');
  assert.equal(error.status, 404);
  assert.equal(error.code, 'project_not_found');
  assert.deepEqual(error.params, { id: 7 });
});

test('api says "network error" when the server cannot be reached, and lets an abort through', async () => {
  setMessages({ 'common.network_error': 'No connection' }, 'en');
  const { error } = await withFetch(new TypeError('Failed to fetch'), () => api('GET', '/x'));
  assert.equal(error.message, 'No connection');
  const abort = new Error('aborted');
  abort.name = 'AbortError';
  const { error: aborted } = await withFetch(abort, () => api('GET', '/x'));
  assert.equal(aborted, abort);
});

test('jobFailure uses the error code when its text exists, else the server text', () => {
  setMessages({ 'error.disk_full': 'Disk full ({free})' }, 'en');
  assert.equal(jobFailure({ error_code: 'disk_full', error_params: { free: 2 }, error: 'raw' }), 'Disk full (2)');
  assert.equal(jobFailure({ error_code: 'other', error: 'raw' }), 'raw');
  assert.equal(jobFailure({ error: 'raw' }), 'raw');
});

test('nameErrorParams adds a way to name a param', () => {
  nameErrorParams({ shade: (s) => `<${s}>` });
  assert.deepEqual(errorParams({ shade: 'dark' }), { shade: '<dark>' });
});
