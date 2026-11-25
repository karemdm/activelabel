/* The thumbnail grid of a project's images (paged, filterable, selectable) and the lightbox that opens one. */
import { issueTag } from './names.js';
import { api } from '../shell/api.js';
import { el, fill, icon } from '../shell/dom.js';
import { fmtBytes, nf1, num, t, tr } from '../shell/i18n.js';
import { toastError } from '../shell/toast.js';

/**
 * Paged grid of thumbnails. `query()` returns {status, issue}. With `selectable`, every card
 * has a checkbox and `selected` (a Set of ids) is kept up to date.
 */
export function imageGrid({ projectId, query, selectable = false, onSelect, onOpen }) {
  const selected = new Set();
  const items = [];
  let total = 0;
  let token = 0;  // only the latest request may draw: a filter change while one is in flight wins
  const list = el('ul', { class: 'thumb-grid' });
  const info = el('p', { class: 'muted grid-info', 'aria-live': 'polite' });
  const pager = el('nav', { class: 'pager', 'aria-label': t('grid.pages') });
  const node = el('div', { class: 'grid-wrap' }, info, list, pager);
  /* pages of a fixed size: the page never grows without limit, even with thousands of images */
  const SIZES = [60, 120, 240];
  let size = SIZES[0];
  try { size = SIZES.includes(Number(localStorage.getItem('al-page-size'))) ? Number(localStorage.getItem('al-page-size')) : size; } catch (e) { /* no storage */ }
  let page = 0;

  const fire = () => onSelect && onSelect(selected);
  const card = (img, index) => {
    const box = selectable
      ? el('label', { class: 'thumb-check' },
          el('input', {
            type: 'checkbox',
            'aria-label': t('grid.select', { name: img.name }),
            checked: selected.has(img.id),
            onchange: (e) => {
              if (e.target.checked) selected.add(img.id); else selected.delete(img.id);
              li.classList.toggle('is-selected', e.target.checked);
              fire();
            },
          }))
      : null;
    const li = el('li', { class: `thumb${selected.has(img.id) ? ' is-selected' : ''}`, 'data-id': img.id },
      el('button', { type: 'button', class: 'thumb-btn', title: img.name, onclick: () => onOpen && onOpen(items, index) },
        el('img', { src: `/api/projects/${projectId}/images/${img.id}/thumb`, alt: '', loading: 'lazy', width: '160', height: '160' })),
      box,
      el('div', { class: 'thumb-meta' },
        el('span', { class: 'thumb-name', title: img.name }, img.name),
        (img.issues || []).length
          ? el('span', { class: 'thumb-tags' }, ...img.issues.map((i) => el('span', { class: `tag tag-${i}`, title: tr(`issue.${i}.help`, '') }, issueTag(i))))
          : null,
        img.duplicate_of_name
          ? el('span', { class: 'thumb-dup', title: img.duplicate_of_name }, t('cur.duplicate_of', { name: img.duplicate_of_name }))
          : null));
    return li;
  };

  function renderPager() {
    const pages = Math.max(1, Math.ceil(total / size));
    const go = (p) => { if (p >= 0 && p < pages && p !== page) { page = p; load(false); node.scrollIntoView({ block: 'start' }); } };
    const btn = (label, p, opts = {}) => el('button', {
      type: 'button', class: `btn small${p === page && opts.current ? ' current' : ''}`,
      disabled: opts.disabled || false, 'aria-current': p === page && opts.current ? 'page' : null,
      'aria-label': opts.aria || null, onclick: () => go(p),
    }, label);
    const numbers = [];
    const shown = new Set([0, pages - 1, page - 1, page, page + 1].filter((p) => p >= 0 && p < pages));
    let last = -1;
    [...shown].sort((a, b) => a - b).forEach((p) => {
      if (last >= 0 && p - last > 1) numbers.push(el('span', { class: 'muted' }, '…'));
      numbers.push(btn(num(p + 1), p, { current: true }));
      last = p;
    });
    const sizeSelect = el('select', { 'aria-label': t('grid.per_page'), onchange: (e) => {
      size = Number(e.target.value);
      try { localStorage.setItem('al-page-size', String(size)); } catch (err) { /* no storage */ }
      page = 0; load(false);
    } }, ...SIZES.map((n) => el('option', { value: String(n), selected: n === size }, num(n))));
    fill(pager,
      total > size ? el('span', { class: 'pager-pages' },
        btn('‹', page - 1, { disabled: page === 0, aria: t('grid.prev') }), ...numbers,
        btn('›', page + 1, { disabled: page >= pages - 1, aria: t('grid.next') })) : el('span'),
      total > SIZES[0] ? el('label', { class: 'pager-size' }, t('grid.per_page'), sizeSelect) : el('span'));
  }

  async function load(reset) {
    const mine = ++token;
    if (reset) { page = 0; selected.clear(); fire(); }
    const q = query();
    const params = new URLSearchParams({ status: q.status || 'active', offset: String(page * size), limit: String(size) });
    if (q.issue) params.set('issue', q.issue);
    if (q.ids) params.set('ids', q.ids.join(','));
    if (q.task != null) params.set('task', String(q.task));
    if (q.job != null) params.set('job', String(q.job));
    if (q.dataset) {
      params.set('dataset', String(q.dataset.n));
      if (q.dataset.fold != null) params.set('fold', String(q.dataset.fold)); else if (q.dataset.part) params.set('part', q.dataset.part);
    }
    try {
      const data = await api('GET', `/projects/${projectId}/images?${params}`);
      if (mine !== token) return;
      total = data.total || 0;
      if (page > 0 && page * size >= total) { page = Math.max(0, Math.ceil(total / size) - 1); return await load(false); }
      items.length = 0;
      fill(list);
      (data.items || []).forEach((img, i) => { items.push(img); list.appendChild(card(img, i)); });
    } catch (err) {
      if (mine === token) toastError(err);
      return;
    }
    const from = page * size + 1;
    info.textContent = items.length
      ? t('grid.range', { from: num(from), to: num(from + items.length - 1), total: num(total) })
      : (query().emptyText || t('grid.empty'));
    renderPager();
  }

  /** Ids of every image of the current filter (all pages). */
  async function allIds() {
    const q = query();
    const ids = [];
    let offset = 0;
    for (;;) {
      const params = new URLSearchParams({ status: q.status || 'active', offset: String(offset), limit: '200' });
      if (q.issue) params.set('issue', q.issue);
      if (q.ids) params.set('ids', q.ids.join(','));
      if (q.task != null) params.set('task', String(q.task));
      if (q.job != null) params.set('job', String(q.job));
      if (q.dataset) {
        params.set('dataset', String(q.dataset.n));
        if (q.dataset.fold != null) params.set('fold', String(q.dataset.fold)); else if (q.dataset.part) params.set('part', q.dataset.part);
      }
      const data = await api('GET', `/projects/${projectId}/images?${params}`);
      const got = data.items || [];
      got.forEach((i) => ids.push(i.id));
      offset += got.length;
      if (!got.length || offset >= (data.total || 0)) return ids;
    }
  }

  const setSelected = (ids) => {
    selected.clear();
    ids.forEach((id) => selected.add(id));
    list.querySelectorAll('.thumb').forEach((li) => {
      const on = selected.has(li.dataset.id);
      li.classList.toggle('is-selected', on);
      const cb = li.querySelector('input[type=checkbox]');
      if (cb) cb.checked = on;
    });
    fire();
  };

  return { node, reload: () => load(true), selected, allIds, setSelected, items: () => items };
}

let lightbox = null;  // one lightbox dialog, reused

export function openLightbox(projectId, items, index) {
  if (!lightbox) {
    lightbox = el('dialog', { class: 'lightbox', 'aria-label': t('lightbox.title') });
    document.body.appendChild(lightbox);
    lightbox.addEventListener('click', (e) => { if (e.target === lightbox) lightbox.close(); });
  }
  let i = index;
  const go = (step) => { i += step; draw(step < 0 ? 'prev' : 'next'); };
  /* `focus` names the button that was used: the body is rebuilt, so focus is put back on it (or on the other
     one at the ends) and the arrow keys, which listen on the dialog, keep working */
  function draw(focus = null) {
    const img = items[i];
    const metrics = Object.entries(img.metrics || {}).filter(([, v]) => typeof v === 'number');
    fill(lightbox,
      el('div', { class: 'lightbox-body' },
        el('div', { class: 'lightbox-stage' },
          el('button', { type: 'button', class: 'btn lb-nav', 'aria-label': t('lightbox.prev'), 'data-nav': 'prev', disabled: i === 0, onclick: () => go(-1) }, '‹'),
          el('img', { src: `/api/projects/${projectId}/images/${img.id}/file`, alt: img.name }),
          el('button', { type: 'button', class: 'btn lb-nav', 'aria-label': t('lightbox.next'), 'data-nav': 'next', disabled: i >= items.length - 1, onclick: () => go(1) }, '›')),
        el('aside', { class: 'lightbox-side' },
          el('div', { class: 'lightbox-head' },
            el('h2', null, img.name),
            el('button', { type: 'button', class: 'icon-btn', 'aria-label': t('common.close'), onclick: () => lightbox.close() }, icon('close'))),
          img.width ? el('p', { class: 'muted' }, `${img.width} × ${img.height} px · ${fmtBytes(img.size_bytes || 0)}`) : null,
          img.status === 'excluded' ? el('p', null, el('span', { class: 'tag tag-excluded' }, t('grid.excluded')), img.excluded_reason ? ` ${img.excluded_reason}` : '') : null,
          (img.classes || []).length ? el('h3', null, t('lightbox.classes')) : null,
          (img.classes || []).length ? el('p', { class: 'thumb-tags' }, ...img.classes.map((c) => el('span', { class: 'chip static' }, c))) : null,
          el('h3', null, t('lightbox.issues')),
          (img.issues || []).length
            ? el('p', { class: 'thumb-tags' }, ...img.issues.map((x) => el('span', { class: `tag tag-${x}`, title: tr(`issue.${x}.help`, '') }, issueTag(x))))
            : el('p', { class: 'muted' }, t(metrics.length ? 'lightbox.no_issues' : 'lightbox.not_analyzed')),
          metrics.length ? el('h3', null, t('lightbox.metrics')) : null,
          metrics.length
            ? el('dl', { class: 'facts compact' }, ...metrics.map(([k, v]) => el('div', null, el('dt', null, tr(`metric.${k}`, k)), el('dd', null, nf1.format(v)))))
            : null)));
    if (focus) {
      const target = [focus, focus === 'prev' ? 'next' : 'prev'].map((k) => lightbox.querySelector(`[data-nav="${k}"]:not(:disabled)`)).find(Boolean);
      (target || lightbox.querySelector('.icon-btn')).focus();
    }
  }
  draw();
  if (!lightbox.open) lightbox.showModal();
  lightbox.onkeydown = (e) => {
    if (e.key === 'ArrowLeft' && i > 0) go(-1);
    if (e.key === 'ArrowRight' && i < items.length - 1) go(1);
  };
}
