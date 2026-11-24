'use strict';
/**
 * Loads the real app.js in jsdom, in one interface language, and returns the helpers it keeps private.
 *
 * app.js is one script wrapped in a function (no modules, no globals). The test does not change the file on
 * disk: it adds one line before the script's closing `})();` that copies a few functions to `window.__app`.
 * If app.js is renamed or restructured so that this closing line is missing, `loadApp` throws a clear error.
 */
const fs = require('node:fs');
const path = require('node:path');
const { JSDOM } = require('jsdom');

const ROOT = path.resolve(__dirname, '..', '..', '..');
const WEB = path.join(ROOT, 'src', 'activelabel', 'web');
const I18N_DIR = path.join(ROOT, 'src', 'activelabel', 'i18n');

const EXPORTED = [
  'fmt', 't', 'tr', 'tn', 'num', 'fmtBytes', 'esc',
  'errorMessage', 'errorParams', 'validationText', 'isHttp',
];

/** The texts of one language, as the server puts them in `window.I18N`. */
function messages(lang) {
  return JSON.parse(fs.readFileSync(path.join(I18N_DIR, `${lang}.json`), 'utf8'));
}

/** The source of app.js with the export line added. */
function appSource() {
  const source = fs.readFileSync(path.join(WEB, 'static', 'app.js'), 'utf8');
  const closing = /\}\)\(\);\s*$/;
  if (!closing.test(source)) {
    throw new Error('app.js does not end with "})();" any more: update tests/js/helpers/load-app.js');
  }
  return source.replace(closing, `window.__app = { ${EXPORTED.join(', ')} };\n})();\n`);
}

/**
 * @param {string} lang 'pt' or 'en'.
 * @param {object} [overrides] Replaces entries of the language's texts (to test a key that is missing).
 * @returns {object} The private helpers of app.js, bound to that language.
 */
function loadApp(lang, overrides = {}) {
  const dom = new JSDOM('<!doctype html><html><body data-page="none"></body></html>', {
    runScripts: 'outside-only',
    url: 'http://localhost/',
  });
  const { window } = dom;
  window.I18N = { ...messages(lang), ...overrides };
  window.VL = { locale: lang };
  window.eval(appSource());
  return window.__app;
}

module.exports = { loadApp, messages, ROOT, WEB };
