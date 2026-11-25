/* Building and redrawing DOM nodes without HTML parsing of server text, and the icon set. */

export const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

export function append(node, kids) {
  for (const k of kids) {
    if (k == null || k === false) continue;
    if (Array.isArray(k)) append(node, k);
    else if (k instanceof Node) node.appendChild(k);
    else node.appendChild(document.createTextNode(String(k)));
  }
  return node;
}

/** Builds a DOM element. Strings become text nodes, so server text is never parsed as HTML. */
export function el(tag, props, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') node.className = v;
    else if (k === 'html') node.innerHTML = v; // only used with fixed, trusted markup (icons)
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else if (v === true) node.setAttribute(k, '');
    else node.setAttribute(k, v);
  }
  return append(node, kids);
}

/** Replaces a node's children. Optional parts are written `cond ? node : null`: like el(), this skips them
 *  (the DOM's own replaceChildren would print "null"). */
export const fill = (node, ...nodes) => {
  node.replaceChildren();
  return append(node, nodes);
};

/** Adds children to a node, skipping null/undefined/false like el() does. */
export const add = (node, ...nodes) => append(node, nodes);

export const ICONS = {
  download: '<path d="M12 4v11M7 10l5 5 5-5M5 20h14"/>',
  star: '<path d="m12 4 2.5 5 5.5.8-4 3.9.9 5.5L12 16.6l-4.9 2.6.9-5.5-4-3.9 5.5-.8z"/>',
  classification: '<path d="M4 4h8l8 8-8 8-8-8z"/><circle cx="8.5" cy="8.5" r="1.2"/>',
  detection: '<rect x="4" y="6" width="16" height="12" rx="1" stroke-dasharray="3 2"/><path d="M4 6h4M4 6v4M20 18h-4M20 18v-4"/>',
  segmentation: '<path d="M5 15 3 8l6-4 5 3 6 1-2 8-7 3z"/>',
  keypoints: '<circle cx="12" cy="5" r="1.8"/><circle cx="6" cy="12" r="1.8"/><circle cx="18" cy="12" r="1.8"/><circle cx="9" cy="20" r="1.8"/><circle cx="15" cy="20" r="1.8"/><path d="M12 7v5M7.6 12H16.4M12 12l-2.2 6.5M12 12l2.2 6.5"/>',
  warning: '<path d="M12 3 2 20h20z"/><path d="M12 10v5M12 17.5v.5"/>',
  error: '<circle cx="12" cy="12" r="9"/><path d="M9 9l6 6M15 9l-6 6"/>',
  check: '<circle cx="12" cy="12" r="9"/><path d="m8 12.5 3 3 5-6"/>',
  upload: '<path d="M12 16V4M7 9l5-5 5 5M4 16v4h16v-4"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/>',
  close: '<path d="M6 6l12 12M18 6 6 18"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5 19 19M5 19l1.5-1.5M17.5 6.5 19 5"/>',
  moon: '<path d="M20 14A8 8 0 0 1 10 4a8 8 0 1 0 10 10z"/>',
  fiftyone: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 9h18M9 9v11"/>',
  cvat: '<path d="M4 8V4h4M20 8V4h-4M4 16v4h4M20 16v4h-4"/><rect x="8" y="8" width="8" height="8" rx="1"/>',
  external: '<path d="M14 4h6v6M20 4l-9 9M18 14v5H5V6h5"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  refresh: '<path d="M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7"/>',
  chart: '<path d="M4 20V4M4 20h16M8 15l4-5 3 3 5-7"/>',
  copy: '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/>',
  chevron: '<path d="m6 9 6 6 6-6"/>',
  edit: '<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="m13.5 6.5 4 4"/>',
  trash: '<path d="M4 7h16M10 7V4h4v3M6 7l1 13h10l1-13M10 11v6M14 11v6"/>',
  images: '<rect x="3" y="5" width="18" height="14" rx="2"/><circle cx="9" cy="10" r="1.6"/><path d="m21 16-5-5-8 8"/>',
  video: '<rect x="3" y="6" width="13" height="12" rx="2"/><path d="m16 10 5-3v10l-5-3z"/>',
};

export const icon = (name, size = 18) =>
  el('span', {
    class: `icon icon-${name}`,
    'aria-hidden': 'true',
    html: `<svg viewBox="0 0 24 24" width="${size}" height="${size}" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${ICONS[name] || ''}</svg>`,
  });

/** Redraws a box whose controls carry `data-act` keys and keeps keyboard focus on the same control
 *  (or on the first one when it is gone), so a refresh does not send the person back to the page top. */
export function redraw(box, ...nodes) {
  const key = box.contains(document.activeElement) ? document.activeElement.dataset.act : null;
  fill(box, ...nodes);
  if (key == null) return;
  const keyed = [...box.querySelectorAll('[data-act]')];
  const again = keyed.find((x) => x.dataset.act === key && !x.disabled) || keyed.find((x) => !x.disabled);
  if (again) again.focus();
}
