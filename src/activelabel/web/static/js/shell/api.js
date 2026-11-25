/* Calls to the JSON API under /api and the texts of its errors (by error code, in the interface language). */
import { I18N, nf, t, tn, tr } from './i18n.js';

/* how the interface names the values an error carries (the server sends raw ids, names and numbers) */
export const ERROR_PARAMS = {
  runs: (n) => `r${n}`,
  field: (field) => tr(`error.field.${field}`, field),
  threshold: (name) => tr(`threshold.${name}`, name),   // the label of the field, as in the Limits panel
  file: (file) => file || t('error.this_file'),
  what: (what) => tr(`error.what.${what}`, what),
};

/** Adds the ways to name error params that need names the API layer does not know (tasks, tools...). */
export function nameErrorParams(namers) {
  Object.assign(ERROR_PARAMS, namers);
}

/** An error's params as texts: numbers formatted, lists joined, known ids named in the interface language. */
export function errorParams(params) {
  const out = {};
  for (const [key, value] of Object.entries(params || {})) {
    const one = ERROR_PARAMS[key] || ((x) => (typeof x === 'number' ? nf.format(x) : String(x)));
    if (Array.isArray(value)) out[key] = value.filter((x) => typeof x !== 'object').map(one).join(', ');
    else if (value === null || typeof value !== 'object') out[key] = one(value);
  }
  return out;
}

/** One validation problem (HTTP 422): "field: problem", the problem written in the interface language. */
export function validationText(entry) {
  if (!entry || !entry.msg) return JSON.stringify(entry);
  const field = (Array.isArray(entry.loc) ? entry.loc : []).filter((x) => !['body', 'query', 'path'].includes(x) && typeof x === 'string').join(' › ');
  const problem = tr(`error.v.${entry.type}`, entry.msg, errorParams(entry.ctx));
  return field ? `${field}: ${problem}` : problem;
}

/** The text of an error answer: translated by its code (with the params filled in), or, for a code this
 *  interface does not know, the server's own text. */
export function errorMessage(data, status) {
  const detail = data && data.detail;
  if (Array.isArray(detail)) return detail.map(validationText).join('; ');  // a plain FastAPI 422
  if (data && data.code === 'validation_error' && data.params && Array.isArray(data.params.errors)) {
    return data.params.errors.map(validationText).join('; ');
  }
  const info = detail && typeof detail === 'object' ? detail : data;  // older answers keep a dict as detail
  const code = info && typeof info.code === 'string' ? info.code : null;
  const counted = info && info.params && typeof info.params.count === 'number';
  if (code && counted && `error.${code}_other` in I18N) return tn(`error.${code}`, info.params.count, errorParams(info.params));
  if (code && `error.${code}` in I18N) return t(`error.${code}`, errorParams(info.params || info));
  if (detail && typeof detail === 'object') return String(detail.message || JSON.stringify(detail));
  return detail ? String(detail) : t('error.http_status', { status: String(status) });  // never a bare "HTTP 500"
}

/** Why a job failed, in the interface language: by its error code (with the params filled in), else the server's own text. */
export function jobFailure(job) {
  const key = `error.${job.error_code}`;
  return job.error_code && key in I18N ? t(key, errorParams(job.error_params)) : job.error;
}

export async function api(method, path, { json, body, signal } = {}) {
  const init = { method, headers: { Accept: 'application/json' }, signal };
  if (json !== undefined) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(json);
  } else if (body) {
    init.body = body;
  }
  let response;
  try {
    response = await fetch(`/api${path}`, init);
  } catch (err) {
    if (err && err.name === 'AbortError') throw err;
    throw new Error(t('common.network_error'));
  }
  let data = null;
  const text = await response.text();
  if (text) {
    try {
      data = JSON.parse(text);
    } catch (e) {
      data = null;
    }
  }
  if (!response.ok) {
    const err = new Error(errorMessage(data, response.status));
    err.status = response.status;
    err.code = data && typeof data.code === 'string' ? data.code : null;
    err.params = data && data.params ? data.params : {};
    err.detail = data ? data.detail : null;  // structured details (e.g. a decision the user must make)
    throw err;
  }
  return data;
}
