/**
 * Gives the tests the interface helpers bound to one language: each helper first calls `setMessages()` with
 * that language's texts (and overrides), so `en` and `pt` can be compared side by side.
 *
 * The helpers are imported from their own modules. main.js is imported once too, for what it sets up when it
 * is evaluated (the error params named with task, tool and status names); it expects a browser, so one jsdom
 * window gives it `window`, `document` and the DOM classes.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { JSDOM } from 'jsdom';

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
export const WEB = path.join(ROOT, 'src', 'activelabel', 'web');
const I18N_DIR = path.join(ROOT, 'src', 'activelabel', 'i18n');
const JS = path.join(WEB, 'static', 'js');

const dom = new JSDOM('<!doctype html><html><body data-page="none"></body></html>', { url: 'http://localhost/' });
const BROWSER = {
  window: dom.window,
  document: dom.window.document,
  localStorage: dom.window.localStorage,
  HTMLElement: dom.window.HTMLElement,
  Element: dom.window.Element,
  Node: dom.window.Node,
  matchMedia: () => ({ matches: false, addEventListener() {}, removeEventListener() {} }),
};
for (const [name, value] of Object.entries(BROWSER)) {
  // defineProperty, not assignment: some Node versions have a read-only global of the same name
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}

/** The texts of one language, as the server puts them in `window.I18N`. */
export function messages(lang) {
  return JSON.parse(fs.readFileSync(path.join(I18N_DIR, `${lang}.json`), 'utf8'));
}

/**
 * @param {string} lang 'pt' or 'en'.
 * @param {object} [overrides] Replaces entries of the language's texts (to test a key that is missing).
 * @returns {Promise<object>} The helpers exported by main.js; each one runs with the texts of that language.
 */
/** Where each helper the tests use lives (module under static/js, exported names). */
const HELPERS = {
  'shell/i18n.js': ['fmt', 't', 'tr', 'tn', 'num', 'fmtBytes'],
  'shell/dom.js': ['esc'],
  'shell/api.js': ['errorMessage', 'errorParams', 'validationText'],
  'integrations/tools.js': ['isHttp'],
};

/**
 * @param {string} lang 'pt' or 'en'.
 * @param {object} [overrides] Replaces entries of the language's texts (to test a key that is missing).
 * @returns {Promise<object>} The helpers; each one runs with the texts of that language.
 */
export async function loadApp(lang, overrides = {}) {
  dom.window.I18N = {};
  dom.window.VL = { locale: lang };
  await import(pathToFileURL(path.join(JS, 'main.js')).href);
  const { setMessages } = await import(pathToFileURL(path.join(JS, 'shell', 'i18n.js')).href);
  const texts = { ...messages(lang), ...overrides };
  const bound = {};
  for (const [file, names] of Object.entries(HELPERS)) {
    const module = await import(pathToFileURL(path.join(JS, ...file.split('/'))).href);
    for (const name of names) {
      bound[name] = (...args) => {
        setMessages(texts, lang);
        return module[name](...args);
      };
    }
  }
  return bound;
}
