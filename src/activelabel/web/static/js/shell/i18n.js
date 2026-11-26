/* Texts and number/date formats in the interface language. setMessages() is called once by main.js before any
 * page starts; the exported LOCALE, I18N, nf and nf1 are live bindings, so importers always read the current
 * values. No module may call t() or read LOCALE when it is first imported (the texts are not set yet). */

export let LOCALE = 'en';
export let I18N = {};
export let nf = new Intl.NumberFormat(LOCALE);
export let nf1 = new Intl.NumberFormat(LOCALE, { maximumFractionDigits: 1 });
export let nf3 = new Intl.NumberFormat(LOCALE, { maximumFractionDigits: 3 });
export let nf4 = new Intl.NumberFormat(LOCALE, { maximumFractionDigits: 4 });
/** A number exactly as it is (no rounding, no thousands separator): a learning rate of 0.0001 stays 0.0001 (0,0001 in pt). */
export let nfExact = new Intl.NumberFormat(LOCALE, { maximumFractionDigits: 20, useGrouping: false });

/** Sets the texts (as the server writes them in `window.I18N`) and the locale of the formats. */
export function setMessages(messages, locale) {
  I18N = messages || {};
  LOCALE = locale || 'en';
  nf = new Intl.NumberFormat(LOCALE);
  nf1 = new Intl.NumberFormat(LOCALE, { maximumFractionDigits: 1 });
  nf3 = new Intl.NumberFormat(LOCALE, { maximumFractionDigits: 3 });
  nf4 = new Intl.NumberFormat(LOCALE, { maximumFractionDigits: 4 });
  nfExact = new Intl.NumberFormat(LOCALE, { maximumFractionDigits: 20, useGrouping: false });
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
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleString(LOCALE, { dateStyle: 'medium', timeStyle: 'short' });
};
/** How a training run is named on every screen: `#3` (the same as the run's title). */
export const runName = (n) => `#${n}`;
/** The decimal sign of the interface language ("," in Portuguese, "." in English). */
export const decimalSign = () => nfExact.format(1.5).charAt(1);
/** A number the person typed in a field: the decimal comma counts as the decimal point where the language uses it.
 *  Returns NaN for anything that is not a plain number. */
export const parseNumber = (text) => {
  const token = String(text).trim();
  if (!token) return NaN;
  return Number(decimalSign() === ',' && /^[+-]?\d*,\d+(e[+-]?\d+)?$/i.test(token) ? token.replace(',', '.') : token);
};
/** The tokens of a list of numbers ("0,01; 0,001" in Portuguese, "0.01, 0.001" in English). Where the comma is the decimal sign
 *  the items are separated by ";" or by a comma followed by a space, so "0,01, 0,001" is two values; elsewhere by "," or ";".
 *  A list of whole numbers (`integers`) has no decimal sign, so every comma separates: "10,20" is two values in any language. */
export const splitNumbers = (text, integers = false) =>
  String(text).split(decimalSign() === ',' && !integers ? /;|,\s+/ : /[,;]/).map((s) => s.trim()).filter(Boolean);
/** The separator the interface language writes between the items of a list of numbers. */
export const listSeparator = () => (decimalSign() === ',' ? '; ' : ', ');
export const fmtBytes = (n) => {
  if (n < 1024) return `${num(n)} B`;
  if (n < 1024 * 1024) return `${nf1.format(n / 1024)} KB`;
  return `${nf1.format(n / 1024 / 1024)} MB`;
};
/** Lower case in the interface language (for a name inside a sentence). */
export const lower = (text) => String(text).toLocaleLowerCase(LOCALE);
