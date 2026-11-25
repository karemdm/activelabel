/* Texts and number/date formats in the interface language. setMessages() is called once by main.js before any
 * page starts; the exported LOCALE, I18N, nf and nf1 are live bindings, so importers always read the current
 * values. No module may call t() or read LOCALE when it is first imported (the texts are not set yet). */

export let LOCALE = 'en';
export let I18N = {};
export let nf = new Intl.NumberFormat(LOCALE);
export let nf1 = new Intl.NumberFormat(LOCALE, { maximumFractionDigits: 1 });

/** Sets the texts (as the server writes them in `window.I18N`) and the locale of the formats. */
export function setMessages(messages, locale) {
  I18N = messages || {};
  LOCALE = locale || 'en';
  nf = new Intl.NumberFormat(LOCALE);
  nf1 = new Intl.NumberFormat(LOCALE, { maximumFractionDigits: 1 });
}

export const fmt = (text, vars) =>
  String(text).replace(/\{(\w+)\}/g, (m, k) => (vars && k in vars ? vars[k] : m));
export const t = (key, vars) => fmt(key in I18N ? I18N[key] : key, vars);
/** Translation with a fallback text when the key is unknown (server-provided names). */
export const tr = (key, fallback, vars) => (key in I18N ? fmt(I18N[key], vars) : fallback);
/** Plural: key_one for exactly 1 and key_other for anything else, 0 included ("0 imagens", also in Portuguese,
 *  whose own plural rule would call 0 singular); {n} is the formatted number. */
export const tn = (key, n, vars) => {
  const suffix = n === 1 ? '_one' : '_other';
  return fmt(I18N[key + suffix] || key + suffix, { n: nf.format(n), ...vars });
};
export const num = (n) => nf.format(Number(n) || 0);
export const fmtDate = (iso) => {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleDateString(LOCALE, { dateStyle: 'medium' });
};
export const fmtDateTime = (iso) => {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleString(LOCALE, { dateStyle: 'short', timeStyle: 'short' });
};
export const fmtBytes = (n) => {
  if (n < 1024) return `${num(n)} B`;
  if (n < 1024 * 1024) return `${nf1.format(n / 1024)} KB`;
  return `${nf1.format(n / 1024 / 1024)} MB`;
};
/** Lower case in the interface language (for a name inside a sentence). */
export const lower = (text) => String(text).toLocaleLowerCase(LOCALE);
