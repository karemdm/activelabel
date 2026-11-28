/* Step 5, Training, run detail: the precision × recall and F1 × confidence curves of a part of the results (#322),
   with a class picker and the same numbers as a table. The workers write `curves` to each part of results.json:
   {iou, match, confidence: [0 … 1], classes: {name: {precision, recall, f1, support}}, all: {precision, recall, f1}},
   every list sampled at the confidences of `confidence`. The validation part also has the confidence of the best
   F1 (#323: `suggested_threshold` for the mean of the classes, `suggested_thresholds` per class), marked on the
   F1 chart. No chart library: SVG in the style of the epoch charts. */
import { el, fill } from '../shell/dom.js';
import { lower, nf2, nf3, num, t } from '../shell/i18n.js';

const SVGNS = 'http://www.w3.org/2000/svg';
/** Confidences listed in the table: every tenth point of the grid (0, 0.1 … 1). */
const TABLE_STEP = 10;

/** Whether a part of the results has curves to draw (older runs have none). */
export function hasCurves(curves) {
  return !!curves && Array.isArray(curves.confidence) && curves.confidence.length > 1
    && !!curves.classes && Object.keys(curves.classes).length > 0;
}

/** The choices of the class picker: the mean of the classes first (value ''), then the classes with the most
 *  objects first (ties by name). */
export function curveChoices(curves) {
  const classes = Object.entries(curves.classes || {})
    .sort(([a, x], [b, y]) => (y.support || 0) - (x.support || 0) || a.localeCompare(b))
    .map(([name, c]) => ({ value: name, label: name, support: c.support || 0 }));
  return [{ value: '', label: t('tr.curves_all'), support: null }, ...classes];
}

/** The curve of one class, or the mean of the classes for '' (or a class the part does not have). */
export function curveOf(curves, name) {
  return (name && curves.classes[name]) || curves.all || { precision: [], recall: [], f1: [] };
}

/** Points of the precision × recall curve, [recall, precision], from the most confident threshold to the least
 *  (recall only grows that way). */
export function prPoints(curves, name) {
  const c = curveOf(curves, name);
  const n = Math.min(c.recall.length, c.precision.length);
  const pts = [];
  for (let i = n - 1; i >= 0; i -= 1) {
    const p = [c.recall[i], c.precision[i]];
    if (Number.isFinite(p[0]) && Number.isFinite(p[1])) pts.push(p);
  }
  return pts;
}

/** Points of the F1 × confidence curve, [confidence, F1]. */
export function f1Points(curves, name) {
  const c = curveOf(curves, name);
  return curves.confidence.map((x, i) => [x, c.f1[i]]).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
}

/** The highest F1 and the confidence where it is reached (the first one when several tie), or null. */
export function bestF1(curves, name) {
  let best = null;
  f1Points(curves, name).forEach(([x, y]) => { if (y > 0 && (!best || y > best.f1)) best = { confidence: x, f1: y }; });
  return best;
}

/** The threshold the validation suggested for a class ('' = the mean of the classes), or null. */
export function suggestionOf(part, name) {
  const value = name ? (part.suggested_thresholds || {})[name] : part.suggested_threshold;
  return Number.isFinite(value) ? value : null;
}

/** Rows of the accessible table: precision, recall and F1 at every `step`-th confidence of the grid. */
export function tableRows(curves, name, step = TABLE_STEP) {
  const c = curveOf(curves, name);
  return curves.confidence
    .map((confidence, i) => ({ confidence, precision: c.precision[i], recall: c.recall[i], f1: c.f1[i] }))
    .filter((_, i) => i % step === 0 || i === curves.confidence.length - 1);
}

function svg(tag, attrs, ...kids) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v != null) n.setAttribute(k, String(v));
  for (const k of kids) if (k) n.appendChild(typeof k === 'string' ? document.createTextNode(k) : k);
  return n;
}

/* A chart with both axes from 0 to 1 (the title names y first, then x). series: [{name, points: [[x, y], ...]}]; `mark` draws a dotted vertical
   rule at x = mark.x with its legend text. */
function unitChart(title, xName, series, mark) {
  const W = 440;
  const H = 230;
  const m = { l: 40, r: 14, t: 12, b: 40 };
  const X = (x) => m.l + x * (W - m.l - m.r);
  const Y = (y) => H - m.b - y * (H - m.t - m.b);
  const kids = [];
  for (let i = 0; i <= 4; i += 1) {
    const v = i / 4;
    kids.push(svg('line', { class: 'tr-grid', x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }));
    kids.push(svg('text', { class: 'tr-axis', x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end' }, nf2.format(v)));
    kids.push(svg('text', { class: 'tr-axis', x: X(v), y: H - m.b + 16, 'text-anchor': 'middle' }, nf2.format(v)));
  }
  kids.push(svg('text', { class: 'tr-axis', x: (m.l + W - m.r) / 2, y: H - 6, 'text-anchor': 'middle' }, xName));
  if (mark) kids.push(svg('line', { class: 'tr-best', x1: X(mark.x), x2: X(mark.x), y1: m.t, y2: H - m.b }));
  series.forEach((s, i) => {
    if (!s.points.length) return;
    kids.push(svg('polyline', { class: `tr-line s${i + 1}`, fill: 'none', points: s.points.map(([x, y]) => `${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join(' ') }));
  });
  const described = series.filter((s) => s.points.length).map((s) => s.name).join(', ');
  return el('figure', { class: 'tr-chart' },
    el('figcaption', null, el('strong', null, title)),
    svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'tr-svg', role: 'img', 'aria-label': `${title}: ${described}` }, ...kids),
    el('ul', { class: 'tr-legend' },
      ...series.map((s, i) => (s.points.length ? el('li', null, el('span', { class: `tr-swatch s${i + 1}`, 'aria-hidden': 'true' }), s.name) : null)),
      mark ? el('li', null, el('span', { class: 'tr-swatch best', 'aria-hidden': 'true' }), mark.label) : null));
}

function curveTable(curves, name, label) {
  const fmt = (v) => (Number.isFinite(v) ? nf3.format(v) : '–');
  return el('details', { class: 'tr-curve-table' },
    el('summary', null, t('tr.curves_table')),
    el('div', { class: 'tr-table-wrap' }, el('table', { class: 'tr-table' },
      el('caption', { class: 'sr-only' }, `${t('tr.curves_title')}: ${label}`),
      el('thead', null, el('tr', null, ...['tr.confidence', 'tr.metric.precision', 'tr.metric.recall', 'tr.metric.f1'].map((k) => el('th', { scope: 'col' }, t(k))))),
      el('tbody', null, ...tableRows(curves, name).map((r) => el('tr', null,
        el('th', { scope: 'row' }, nf2.format(r.confidence)), el('td', null, fmt(r.precision)), el('td', null, fmt(r.recall)), el('td', null, fmt(r.f1))))))));
}

function charts(part, name) {
  const { curves } = part;
  const choice = curveChoices(curves).find((c) => c.value === name) || curveChoices(curves)[0];
  const series = (points) => [
    { name: choice.label, points: points(curves, choice.value) },
    ...(choice.value ? [{ name: t('tr.curves_all'), points: points(curves, '') }] : []),
  ];
  const suggested = suggestionOf(part, choice.value);
  const best = bestF1(curves, choice.value);
  const mark = suggested != null ? { x: suggested, label: t('tr.curves_suggested', { c: nf2.format(suggested) }) }
    : best ? { x: best.confidence, label: t('tr.curves_best_f1', { f1: nf3.format(best.f1), c: nf2.format(best.confidence) }) } : null;
  return [
    el('div', { class: 'tr-charts' },
      unitChart(`${t('tr.metric.precision')} × ${lower(t('tr.metric.recall'))}`, t('tr.metric.recall'), series(prPoints), null),
      unitChart(`${t('tr.metric.f1')} × ${lower(t('tr.confidence'))}`, t('tr.confidence'), series(f1Points), mark)),
    curveTable(curves, choice.value, choice.label),
  ];
}

/** The block with the curves of a part of the results (`curves`, and the suggestions of the validation) and its
 *  class picker, or null without curves. */
export function curvesBlock(part) {
  const { curves } = part || {};
  if (!hasCurves(curves)) return null;
  const choices = curveChoices(curves);
  const box = el('div', { class: 'tr-curve-box' }, ...charts(part, ''));
  const picker = el('select', {
    onchange: (e) => { fill(box, ...charts(part, e.target.value)); },
  }, ...choices.map((c) => el('option', { value: c.value }, c.support == null ? c.label : `${c.label} (${num(c.support)})`)));
  const match = curves.match === 'mask' ? t('tr.curves_match_mask') : t('tr.curves_match_box');
  return el('div', { class: 'tr-block tr-curves' },
    el('h4', null, t('tr.curves_title')),
    el('p', { class: 'muted tr-note' }, t('tr.curves_help', { iou: nf2.format(curves.iou || 0.5), match })),
    suggestionOf(part, '') != null ? el('p', { class: 'tr-note' }, t('tr.suggested_note', { c: nf2.format(suggestionOf(part, '')) })) : null,
    el('label', { class: 'tr-curve-pick' }, el('span', null, t('tr.curves_class')), picker),
    box);
}
