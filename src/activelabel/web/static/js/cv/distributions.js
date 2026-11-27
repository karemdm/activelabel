/* The distribution of the annotations as horizontal bars (#623): the Curation block (classes, objects per
 * image, box sizes) and the class balance of the parts of a split. The numbers come from
 * GET /projects/{id}/curation/distributions and from the split preview. */
import { el, fill } from '../shell/dom.js';
import { num, t, tr } from '../shell/i18n.js';

export const SHOWN = 8;
const pct = (n, total) => (total > 0 ? Math.round((n / total) * 100) : 0);

/**
 * A list of bars on one common scale, each with its label and value.
 * @param {{label: string, value: number, text?: string, title?: string}[]} rows The bars.
 * @param {number} [max] The value of a full bar (the largest value when not given).
 * @returns {HTMLElement} The `ul.bars` list.
 */
export function barList(rows, max = Math.max(1, ...rows.map((r) => r.value))) {
  return el('ul', { class: 'bars' }, ...rows.map((r) =>
    el('li', null,
      el('span', { class: 'bar-label', title: r.title || r.label }, r.label),
      el('span', { class: 'bar-track' }, el('span', { class: 'bar-fill', style: `width:${Math.round((r.value / max) * 100)}%` })),
      el('span', { class: 'bar-value' }, r.text != null ? r.text : num(r.value)))));
}

/* a titled chart with its list of bars and a note under it */
function chart(title, rows, note) {
  return el('figure', { class: 'dist-chart' },
    el('figcaption', null, el('strong', null, title)),
    rows.length ? barList(rows) : el('p', { class: 'muted' }, t('dist.none')),
    note ? el('p', { class: 'muted dist-note' }, note) : null);
}

/**
 * The Curation block "Distribution of the annotations".
 * @param {object|null} data The answer of the distributions endpoint.
 * @returns {HTMLElement|null} The block, or null when there is nothing to show (no images or no annotations).
 */
export function distributionsBlock(data) {
  if (!data || !data.images || !(data.annotated > 0)) return null;
  const classification = data.task === 'classification';
  const classes = data.classes || [];
  let all = false;
  const classBox = el('div');
  const more = el('button', { type: 'button', class: 'btn small', 'aria-expanded': 'false', onclick: () => { all = !all; paint(); } });
  function paint() {
    const rows = classes.map((c) => ({
      label: c.name,
      value: classification ? c.images : c.objects,
      text: classification ? num(c.images) : `${num(c.objects)} · ${num(c.images)}`,
      title: t('dist.class_title', { name: c.name, objects: num(c.objects), images: num(c.images) }),
    }));
    fill(classBox, chart(classification ? t('dist.images_per_class') : t('dist.objects_per_class'), all ? rows : rows.slice(0, SHOWN),
      classification ? null : t('dist.classes_note')));
    more.hidden = rows.length <= SHOWN;
    more.textContent = all ? t('ds.show_less') : t('ds.show_all', { n: num(rows.length) });
    more.setAttribute('aria-expanded', String(all));
  }
  paint();
  const perImage = (data.objects_per_image || []).map((b) => ({ label: b.key, value: b.images }));
  const sizes = (data.box_area || []).map((b) => ({ label: tr(`dist.area.${b.key}`, b.key), value: b.objects }));
  return el('section', { class: 'dist', 'aria-labelledby': 'h-dist' },
    el('h3', { id: 'h-dist' }, t('dist.title')),
    el('p', { class: 'muted' }, t('dist.summary', { annotated: num(data.annotated), total: num(data.images), without: num(data.without_annotations) })),
    el('div', { class: 'dist-grid' },
      el('div', null, classBox, more),
      classification ? null : chart(t('dist.per_image'), perImage, t('dist.per_image_note')),
      data.box_area && data.boxes ? chart(t('dist.box_sizes'), sizes, t('dist.box_sizes_note', { boxes: num(data.boxes) })) : null));
}

/**
 * The class balance of the parts of a split: for every class, the share it has of each part (a bar per part).
 * Parts that are balanced have bars of the same length.
 * @param {{label: string, n: number, cls: string}[]} items The parts (as the preview draws them).
 * @param {{objects?: Object<string, number>}[]} per The statistics of each part, in the order of `items`.
 * @param {string[]} names The classes to show, in order.
 * @returns {HTMLElement|null} The chart, or null without classes.
 */
export function partBalance(items, per, names) {
  if (!names.length || !items.length) return null;
  const totals = per.map((s) => Object.values((s && s.objects) || {}).reduce((a, n) => a + n, 0));
  const shares = names.map((c) => per.map((s, i) => pct(((s && s.objects) || {})[c] || 0, totals[i])));
  const max = Math.max(1, ...shares.flat());
  return el('div', { class: 'dist-balance', style: `--parts:${items.length}` },
    el('div', { class: 'dist-balance-head' }, el('span', { class: 'sr-only' }, t('dist.class')),
      ...items.map((x) => el('span', null, el('span', { class: `ds-dot ds-${x.cls}`, 'aria-hidden': 'true' }), x.label))),
    ...names.map((c, k) => el('div', { class: 'dist-balance-row' },
      el('span', { class: 'bar-label', title: c }, c),
      ...items.map((x, i) => el('span', { class: 'dist-balance-cell', title: t('dist.share_title', { class: c, part: x.label, pct: shares[k][i] }) },
        el('span', { class: 'bar-track' }, el('span', { class: `bar-fill ds-${x.cls}`, style: `width:${Math.round((shares[k][i] / max) * 100)}%` })),
        el('span', { class: 'bar-value' }, `${shares[k][i]}%`))))));
}
