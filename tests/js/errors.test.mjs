import test from 'node:test';
import assert from 'node:assert/strict';
import { loadApp, messages } from './helpers/load-app.mjs';

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
