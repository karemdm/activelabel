/* Comparing only the chosen training runs or models (#214): the checkbox of a card, the compare bar (how many
 * are chosen, select all of this list, clear) and the table, one row per metric with the best value of each
 * row highlighted. The classes whose value is 0 (or missing) in every chosen column are folded away. */
import { el, icon } from '../shell/dom.js';
import { fmtBytes, nf3, num, t, tn, tr } from '../shell/i18n.js';

const PER_CLASS_KEYS = ['f1', 'ap50_95', 'iou'];
/** Rows whose best value is the lowest: the loss and the error metrics (catalog.LOWER_IS_BETTER, #619). */
export const LOWER_IS_BETTER = ['loss', 'mae', 'rmse'];

const metricName = (m) => tr(`tr.metric.${m}`, tr(`tr.col.${m}`, String(m).replace(/_/g, ' ')));

/** The rows of a comparison of `cols`, read through `resultsOf(col)` (an object with `val` and `test` parts).
 *  `sizeOf(col)`, when given, adds a size row. Returns `{ groups, zeroClasses }`: `groups` is a list of
 *  `{ title, rows }` (validation, test, per class, size) whose rows are `{ label, values, lower, size }`, and
 *  `zeroClasses` the names of the classes left out because every column has 0 or nothing for them. */
export function comparisonRows(cols, resultsOf, sizeOf = null) {
  const groups = [];
  for (const part of ['val', 'test']) {
    const keys = [];
    cols.forEach((c) => Object.entries((resultsOf(c) || {})[part] || {})
      .forEach(([k, v]) => { if (typeof v === 'number' && k !== 'images' && !keys.includes(k)) keys.push(k); }));
    if (keys.length) {
      groups.push({
        title: t(part === 'val' ? 'tr.val' : 'tr.test'),
        rows: keys.map((k) => ({ label: metricName(k), values: cols.map((c) => ((resultsOf(c) || {})[part] || {})[k]), lower: LOWER_IS_BETTER.includes(k) })),
      });
    }
  }
  const perClass = (c) => (((resultsOf(c) || {}).val || {}).per_class || {});
  const perKey = PER_CLASS_KEYS.find((k) => cols.some((c) => Object.values(perClass(c)).some((x) => x && k in x)));
  const zeroClasses = [];
  if (perKey) {
    const classes = [];
    cols.forEach((c) => Object.keys(perClass(c)).forEach((name) => { if (!classes.includes(name)) classes.push(name); }));
    const rows = [];
    classes.forEach((name) => {
      const values = cols.map((c) => (perClass(c)[name] || {})[perKey]);
      if (values.every((v) => !v)) zeroClasses.push(name); else rows.push({ label: name, values });
    });
    if (rows.length) groups.push({ title: t('cmp.per_class', { metric: metricName(perKey) }), rows });
  }
  if (sizeOf) groups.push({ title: t('md.size'), rows: [{ label: t('md.size'), values: cols.map(sizeOf), lower: true, size: true }] });
  return { groups, zeroClasses };
}

/** The best value of a row: the highest (the lowest for a loss or a size); null when there is nothing to choose
 *  (fewer than two numbers, or all equal). */
export function bestOf(row) {
  const nums = row.values.filter((v) => typeof v === 'number');
  if (nums.length < 2 || new Set(nums).size < 2) return null;
  return row.lower ? Math.min(...nums) : Math.max(...nums);
}

/** The comparison table of the chosen columns. `cols` are the runs or models, `head(col)` the nodes of a column
 *  header, `resultsOf` and `sizeOf` as in comparisonRows, `id` the id of the section, `title` its heading and
 *  `help` the line under it. */
export function compareTable({ id, title, help, cols, head, resultsOf, sizeOf = null }) {
  const { groups, zeroClasses } = comparisonRows(cols, resultsOf, sizeOf);
  const cell = (row, v, best) => el('td', { class: `num${best ? ' best' : ''}` }, v == null ? '—' : row.size ? fmtBytes(v) : nf3.format(v));
  const body = [];
  groups.forEach((g) => {
    if (!(g.rows.length === 1 && g.rows[0].size)) body.push(el('tr', { class: 'cmp-group' }, el('th', { scope: 'colgroup', colspan: String(cols.length + 1) }, g.title)));
    g.rows.forEach((row) => {
      const best = bestOf(row);
      body.push(el('tr', null, el('th', { scope: 'row' }, row.label), ...row.values.map((v) => cell(row, v, best != null && v === best))));
    });
  });
  return el('section', { class: 'tr-section md-compare', id, 'aria-labelledby': `${id}-t` },
    el('h3', { id: `${id}-t`, class: 'section-title' }, title),
    el('p', { class: 'muted tr-note' }, help),
    el('div', { class: 'tr-table-wrap' }, el('table', { class: 'tr-table md-table' },
      el('caption', { class: 'sr-only' }, title),
      el('thead', null, el('tr', null, el('th', { scope: 'col' }, t('md.metric')), ...cols.map((c) => el('th', { scope: 'col' }, ...head(c))))),
      el('tbody', null, ...body))),
    zeroClasses.length
      ? el('details', { class: 'cmp-zero' },
          el('summary', null, tn('cmp.zero_classes', zeroClasses.length)),
          el('p', { class: 'muted' }, zeroClasses.join(', ')))
      : null);
}

/** The checkbox of a card: chooses it for the comparison. `label` names the card ("Compare #3", read out and
 *  shown on hover; the bar above the list says what the boxes are for). */
export function compareCheck({ id, label, checked, onChange }) {
  const input = el('input', { type: 'checkbox', id, 'data-act': 'compare', 'aria-label': t('cmp.choose', { name: label }) });
  input.checked = checked;
  input.addEventListener('change', () => onChange(input.checked));
  return el('label', { class: 'cmp-check', for: id, title: t('cmp.choose', { name: label }) }, input);
}

/** The compare bar: select all of this list (checked, mixed or not), how many are chosen, and clear.
 *  `chosen` and `total` are counts; `onAll(on)` and `onClear()` change the choice; `prefix` makes the ids unique. */
export function compareBar({ prefix, chosen, total, onAll, onClear }) {
  const all = el('input', { type: 'checkbox', id: `${prefix}-all`, 'data-act': 'compare-all' });
  all.checked = total > 0 && chosen === total;
  all.indeterminate = chosen > 0 && chosen < total;
  all.addEventListener('change', () => onAll(all.checked));
  return el('div', { class: 'cmp-bar', role: 'group', 'aria-label': t('cmp.bar') },
    el('label', { class: 'cmp-check', for: `${prefix}-all` }, all, el('span', null, t('cmp.select_all', { n: num(total) }))),
    el('span', { class: 'cmp-count', 'aria-live': 'polite' }, icon('chart', 16),
      chosen >= 2 ? tn('cmp.chosen', chosen) : chosen === 1 ? t('cmp.need_one_more') : t('cmp.need_two')),
    chosen ? el('button', { type: 'button', class: 'btn small', 'data-act': 'compare-clear', onclick: onClear }, t('cmp.clear')) : null);
}
