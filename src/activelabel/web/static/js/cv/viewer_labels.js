/* The annotations drawn over the image of the lightbox (read-only, #339): boxes, polygons, keypoints and image
   classes with the colour of their class, a legend, and pre-annotations (dashed, with their score) told apart
   from ground truth. The geometry and colour logic is plain functions; the DOM parts only place what they return. */
import { api } from '../shell/api.js';
import { el, fill } from '../shell/dom.js';
import { t } from '../shell/i18n.js';

export const CLASS_COLOURS = 10;   // --cls-1..10 in app.css; a project with more classes reuses the colours
const SVG_NS = 'http://www.w3.org/2000/svg';
const PREF_KEY = 'al-lightbox-annotations';

/** The colour token of a class: its place in the project's class list picks one of the palette, so the same class
 *  has the same colour in every image; a class the project does not list gets the neutral colour. */
export function classColour(classes, name) {
  const at = (classes || []).indexOf(name);
  return at < 0 ? 'var(--cls-none)' : `var(--cls-${(at % CLASS_COLOURS) + 1})`;
}

/** Where an image of `imgW` x `imgH` is drawn inside a box of `boxW` x `boxH` with `object-fit: contain`
 *  (centred, scaled to fit): the rectangle, in the box's pixels, the annotation layer must cover. */
export function containedRect(boxW, boxH, imgW, imgH) {
  if (!(boxW > 0 && boxH > 0 && imgW > 0 && imgH > 0)) return { x: 0, y: 0, w: boxW || 0, h: boxH || 0 };
  const scale = Math.min(boxW / imgW, boxH / imgH);
  const w = imgW * scale;
  const h = imgH * scale;
  return { x: (boxW - w) / 2, y: (boxH - h) / 2, w, h };
}

const pct = (v) => Math.round(Math.min(1, Math.max(0, v)) * 1e4) / 100;   // relative 0..1 -> percent, two decimals

function boxOf(bbox) {
  if (!Array.isArray(bbox) || bbox.length !== 4) return null;
  const x = pct(bbox[0]);
  const y = pct(bbox[1]);
  return { x, y, w: Math.min(100 - x, pct(bbox[2])), h: Math.min(100 - y, pct(bbox[3])) };
}

/** The shapes to draw for the labels of one image (the API's `labels`), in percent of the image: `kind` is
 *  `box`, `polygon`, `keypoints` or `class` (a class of the whole image); regression values are not drawn.
 *  A label with a `score` is a pre-annotation (`pre`). `bbox` is `{x, y, w, h}`, `rings` lists of `[x, y]`, `points`
 *  `{x, y, v}` (v: 1 hidden, 2 visible; points not labeled, v 0, are left out). */
export function shapesOf(labels, classes) {
  const shapes = [];
  for (const lb of labels || []) {
    if (typeof lb.value === 'number') continue;
    const shape = { label: lb.label, colour: classColour(classes, lb.label), pre: typeof lb.score === 'number', score: lb.score ?? null };
    if (Array.isArray(lb.keypoints) && lb.keypoints.length) {
      shapes.push({ ...shape, kind: 'keypoints', points: lb.keypoints.filter((p) => p[2] > 0).map((p) => ({ x: pct(p[0]), y: pct(p[1]), v: p[2] })),
        bbox: boxOf(lb.bbox) });
    } else if (Array.isArray(lb.polygons) && lb.polygons.length) {
      shapes.push({ ...shape, kind: 'polygon', rings: lb.polygons.map((ring) => ring.map((p) => [pct(p[0]), pct(p[1])])) });
    } else if (Array.isArray(lb.bbox) && lb.bbox.length === 4) {
      shapes.push({ ...shape, kind: 'box', bbox: boxOf(lb.bbox) });
    } else {
      shapes.push({ ...shape, kind: 'class' });
    }
  }
  return shapes;
}

/** Where the name chip of a shape goes (percent of the image): the top-left of the box, of the polygon's
 *  bounds, or of the keypoints. Null for a class of the whole image (those stack in a corner). */
export function anchorOf(shape) {
  if (shape.kind === 'box') return { x: shape.bbox.x, y: shape.bbox.y };
  if (shape.kind === 'polygon') {
    const all = shape.rings.flat();
    return all.length ? { x: Math.min(...all.map((p) => p[0])), y: Math.min(...all.map((p) => p[1])) } : null;
  }
  if (shape.kind === 'keypoints') {
    if (shape.bbox) return { x: shape.bbox.x, y: shape.bbox.y };
    return shape.points.length ? { x: Math.min(...shape.points.map((p) => p.x)), y: Math.min(...shape.points.map((p) => p.y)) } : null;
  }
  return null;
}

/** One entry per class present: `{name, colour, count, pre}` (pre: how many are pre-annotations), most frequent first. */
export function legendOf(shapes) {
  const byName = new Map();
  for (const s of shapes) {
    const entry = byName.get(s.label) || { name: s.label, colour: s.colour, count: 0, pre: 0 };
    entry.count += 1;
    if (s.pre) entry.pre += 1;
    byName.set(s.label, entry);
  }
  return [...byName.values()].sort((a, b) => b.count - a.count || a.name.localeCompare(b.name));
}

/** A shape's text on the image: the class, with the confidence of a pre-annotation ("cat 87%"). */
export const chipText = (shape) => (shape.pre ? `${shape.label} ${Math.round(shape.score * 100)}%` : shape.label);

const svg = (tag, attrs) => {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
};

const ringPoints = (ring) => ring.map((p) => `${p[0]},${p[1]}`).join(' ');

/** The layer over the image: an SVG of boxes and polygons in a 0..100 box (strokes keep their width when it
 *  scales), HTML dots for keypoints and name chips. It fills its parent, so the caller sizes the parent to the
 *  drawn image (see containedRect). */
export function annotationLayer(shapes) {
  const drawing = svg('svg', { viewBox: '0 0 100 100', preserveAspectRatio: 'none', class: 'lb-svg', 'aria-hidden': 'true' });
  const layer = el('div', { class: 'lb-layer', 'aria-hidden': 'true' }, drawing);
  const corner = el('div', { class: 'lb-corner' });
  for (const s of shapes) {
    const style = `--c:${s.colour}`;
    const common = { class: `lb-shape${s.pre ? ' is-pre' : ''}`, style, 'vector-effect': 'non-scaling-stroke' };
    if (s.kind === 'box') {
      drawing.appendChild(svg('rect', { ...common, x: s.bbox.x, y: s.bbox.y, width: s.bbox.w, height: s.bbox.h }));
    } else if (s.kind === 'polygon') {
      for (const ring of s.rings) drawing.appendChild(svg('polygon', { ...common, points: ringPoints(ring) }));
    } else if (s.kind === 'keypoints') {
      if (s.bbox) drawing.appendChild(svg('rect', { ...common, x: s.bbox.x, y: s.bbox.y, width: s.bbox.w, height: s.bbox.h }));
      s.points.forEach((p) => layer.appendChild(el('span', { class: `lb-dot${p.v === 1 ? ' is-hidden' : ''}${s.pre ? ' is-pre' : ''}`, style: `${style};left:${p.x}%;top:${p.y}%` })));
    }
    const at = anchorOf(s);
    if (at) layer.appendChild(el('span', { class: `lb-chip${s.pre ? ' is-pre' : ''}${at.y < 8 ? ' is-low' : ''}${at.x > 70 ? ' is-right' : ''}`, style: `${style};left:${at.x}%;top:${at.y}%` }, chipText(s)));
    else if (s.kind === 'class') corner.appendChild(el('span', { class: `lb-chip static${s.pre ? ' is-pre' : ''}`, style }, chipText(s)));
  }
  if (corner.childNodes.length) layer.appendChild(corner);
  return layer;
}

/** The text list of the annotations (also for a screen reader): a colour swatch, the class, how many; a note
 *  for the pre-annotations (dashed on the image) among them. */
export function legendList(shapes) {
  const entries = legendOf(shapes);
  return el('ul', { class: 'lb-legend' }, ...entries.map((e) => el('li', null,
    el('span', { class: 'lb-swatch', style: `--c:${e.colour}`, 'aria-hidden': 'true' }),
    el('span', { class: 'lb-legend-name' }, e.name),
    el('span', { class: 'muted' }, `× ${e.count}`),
    e.pre ? el('span', { class: 'tag tag-pre' }, t('lightbox.annot_pre', { n: e.pre })) : null)));
}

/** Whether the person wants the annotations shown (remembered; on by default). */
export function showAnnotations() {
  try { return localStorage.getItem(PREF_KEY) !== 'off'; } catch (err) { return true; }
}

/** Remembers whether the annotations are shown. */
export function rememberShow(on) {
  try { localStorage.setItem(PREF_KEY, on ? 'on' : 'off'); } catch (err) { /* not remembered */ }
}

/** Puts the layer exactly over the drawn image: `object-fit: contain` leaves bars when the box and the image have
 *  other proportions, and the shapes are relative to the image, not to the box. */
export function placeLayer(photo, host, size) {
  const r = containedRect(photo.clientWidth, photo.clientHeight, photo.naturalWidth || size.w, photo.naturalHeight || size.h);
  Object.assign(host.style, { left: `${photo.offsetLeft + r.x}px`, top: `${photo.offsetTop + r.y}px`, width: `${r.w}px`, height: `${r.h}px` });
}

/** Keeps `host` over the image while it loads and whenever the window or the lightbox is resized. */
export function followImage(photo, host, size) {
  const place = () => placeLayer(photo, host, size);
  photo.addEventListener('load', place);
  if (typeof ResizeObserver === 'function') new ResizeObserver(place).observe(photo);
  place();
}

const toggleButton = (host) => {
  let on = showAnnotations();
  const apply = () => { host.classList.toggle('is-off', !on); button.setAttribute('aria-pressed', String(on)); button.classList.toggle('active', on); };
  const button = el('button', { type: 'button', class: 'chip-btn', 'data-action': 'toggle-annotations', onclick: () => { on = !on; rememberShow(on); apply(); } }, t('lightbox.annot_show'));
  apply();
  return button;
};

/** Reads the labels of the image (one request) and draws them: the layer goes into `host` (over the image), the
 *  toggle, the legend and the key of pre-annotations into `panel`. Returns the AbortController of the request, so
 *  moving to another image cancels it. A failure is said in the panel; the image stays as it was. */
export function mountAnnotations(projectId, imageId, host, panel) {
  const controller = new AbortController();
  api('GET', `/projects/${projectId}/images/${imageId}/labels`, { signal: controller.signal }).then((data) => {
    const shapes = shapesOf(data.labels, data.classes);
    if (!shapes.length) {
      fill(panel, (data.labels || []).length ? null : el('p', { class: 'muted' }, t('lightbox.annot_none')));
      return;
    }
    fill(host, annotationLayer(shapes));
    fill(panel,
      el('div', { class: 'lb-annots-head' }, el('h3', null, t('lightbox.annots')), toggleButton(host)),
      legendList(shapes),
      shapes.some((s) => s.pre) ? el('p', { class: 'muted small lb-pre-key' }, t('lightbox.annot_pre_key')) : null);
  }).catch((err) => {
    if (!err || err.name !== 'AbortError') fill(panel, el('p', { class: 'muted' }, t('lightbox.annot_error')));
  });
  return controller;
}
