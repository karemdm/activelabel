/**
 * Loads the real interface module (static/js/main.js) in jsdom, in one interface language, and returns the
 * helpers it exports for the tests.
 *
 * main.js is a native ES module. It reads `window.I18N` and `window.VL` when it is evaluated, so every call
 * imports a fresh instance of it: the import address carries the language and a counter in its query string.
 * The module expects a browser, so one jsdom window gives it `window`, `document` and the DOM classes.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { JSDOM } from 'jsdom';

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
export const WEB = path.join(ROOT, 'src', 'activelabel', 'web');
const I18N_DIR = path.join(ROOT, 'src', 'activelabel', 'i18n');
const MAIN = pathToFileURL(path.join(WEB, 'static', 'js', 'main.js')).href;

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

let instances = 0;

/** The texts of one language, as the server puts them in `window.I18N`. */
export function messages(lang) {
  return JSON.parse(fs.readFileSync(path.join(I18N_DIR, `${lang}.json`), 'utf8'));
}

/**
 * @param {string} lang 'pt' or 'en'.
 * @param {object} [overrides] Replaces entries of the language's texts (to test a key that is missing).
 * @returns {Promise<object>} The helpers exported by main.js, bound to that language.
 */
export async function loadApp(lang, overrides = {}) {
  dom.window.I18N = { ...messages(lang), ...overrides };
  dom.window.VL = { locale: lang };
  instances += 1;
  return import(`${MAIN}?lang=${lang}&instance=${instances}`);
}
