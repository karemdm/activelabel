/**
 * Loads the real interface modules (static/js/main.js) in jsdom and returns the helpers main.js exports for the
 * tests, bound to one interface language.
 *
 * The modules are imported once: the texts live in shell/i18n.js, so every helper returned here first calls
 * `setMessages()` with its own language (and overrides). Tests can then compare `en` and `pt` side by side.
 * The modules expect a browser, so one jsdom window gives them `window`, `document` and the DOM classes.
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
export async function loadApp(lang, overrides = {}) {
  dom.window.I18N = {};
  dom.window.VL = { locale: lang };
  const { setMessages } = await import(pathToFileURL(path.join(JS, 'shell', 'i18n.js')).href);
  const app = await import(pathToFileURL(path.join(JS, 'main.js')).href);
  const texts = { ...messages(lang), ...overrides };
  const bound = {};
  for (const [name, fn] of Object.entries(app)) {
    bound[name] = (...args) => {
      setMessages(texts, lang);
      return fn(...args);
    };
  }
  return bound;
}
