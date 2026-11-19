/* activelabel web interface. Vanilla JS, no dependencies. Talks to the JSON API under /api. */
(() => {
  'use strict';

  const VL = window.VL || {};
  const I18N = window.I18N || {};
  const LOCALE = VL.locale || 'en';
  const nf = new Intl.NumberFormat(LOCALE);
  const nf1 = new Intl.NumberFormat(LOCALE, { maximumFractionDigits: 1 });

  const TASKS = ['classification', 'detection', 'segmentation', 'keypoints'];
  const ISSUES = ['blurry', 'dark', 'bright', 'low_contrast', 'small', 'odd_aspect', 'grayscale', 'near_duplicate'];
  const CHUNK = 25;
  const PAGE = 60;
  const SKIP_FILES = new Set(['.ds_store', 'thumbs.db']);

  // ---------------------------------------------------------------- helpers
  const fmt = (text, vars) =>
    String(text).replace(/\{(\w+)\}/g, (m, k) => (vars && k in vars ? vars[k] : m));
  const t = (key, vars) => fmt(key in I18N ? I18N[key] : key, vars);
  /** Translation with a fallback text when the key is unknown (server-provided names). */
  const tr = (key, fallback, vars) => (key in I18N ? fmt(I18N[key], vars) : fallback);
  /** Plural: uses key_one for 1 and key_other otherwise; {n} is the formatted number. */
  const tn = (key, n, vars) => {
    const suffix = n === 1 ? '_one' : '_other';
    return fmt(I18N[key + suffix] || key + suffix, { n: nf.format(n), ...vars });
  };
  const num = (n) => nf.format(Number(n) || 0);
  const fmtTime = (iso) => {
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? '' : d.toLocaleString(LOCALE, { dateStyle: 'short', timeStyle: 'short' });
  };
  const fmtDate = (iso) => {
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? '' : d.toLocaleDateString(LOCALE, { dateStyle: 'medium' });
  };
  const fmtDateTime = (iso) => {
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? '' : d.toLocaleString(LOCALE, { dateStyle: 'short', timeStyle: 'short' });
  };
  const fmtBytes = (n) => {
    if (n < 1024) return `${num(n)} B`;
    if (n < 1024 * 1024) return `${nf1.format(n / 1024)} KB`;
    return `${nf1.format(n / 1024 / 1024)} MB`;
  };
  const esc = (s) =>
    String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  function append(node, kids) {
    for (const k of kids) {
      if (k == null || k === false) continue;
      if (Array.isArray(k)) append(node, k);
      else if (k instanceof Node) node.appendChild(k);
      else node.appendChild(document.createTextNode(String(k)));
    }
    return node;
  }

  /** Builds a DOM element. Strings become text nodes, so server text is never parsed as HTML. */
  function el(tag, props, ...kids) {
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

  const ICONS = {
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
  };
  const icon = (name, size = 18) =>
    el('span', {
      class: `icon icon-${name}`,
      'aria-hidden': 'true',
      html: `<svg viewBox="0 0 24 24" width="${size}" height="${size}" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${ICONS[name] || ''}</svg>`,
    });

  const taskName = (task) => tr(`task.${task}`, task || '?');
  const issueName = (issue) => tr(`issue.${issue}`, issue);
  /* one image's tag (singular: "Blurry" / "Borrada") */
  const issueTag = (issue) => tr(`tag.${issue}`, issueName(issue));
  const sameTask = (a, b) => a === b;

  // ---------------------------------------------------------------- toasts
  function toast(message, kind = 'info', ms = 7000) {
    const box = document.getElementById('toasts');
    if (!box) return;
    const node = el(
      'div',
      { class: `toast toast-${kind}`, role: kind === 'error' ? 'alert' : 'status' },
      icon(kind === 'error' ? 'error' : kind === 'success' ? 'check' : 'info'),
      el('div', { class: 'toast-text' }, message),
      el('button', { type: 'button', class: 'icon-btn', 'aria-label': t('common.close'), onclick: () => node.remove() }, icon('close', 16)),
    );
    box.appendChild(node);
    if (ms) setTimeout(() => node.remove(), ms);
  }
  const toastError = (err) => toast(`${t('common.error')} ${err && err.message ? err.message : err}`, 'error', 10000);

  // ---------------------------------------------------------------- API
  function detailText(detail) {
    if (Array.isArray(detail)) return detail.map((d) => (d && d.msg) || JSON.stringify(d)).join('; ');
    if (detail && typeof detail === 'object') return JSON.stringify(detail);
    return String(detail);
  }

  async function api(method, path, { json, body, signal } = {}) {
    const init = { method, headers: { Accept: 'application/json' }, signal };
    if (json !== undefined) {
      init.headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(json);
    } else if (body) {
      init.body = body;
    }
    let response;
    try {
      response = await fetch(`/api${path}`, init);
    } catch (err) {
      if (err && err.name === 'AbortError') throw err;
      throw new Error(t('common.network_error'));
    }
    let data = null;
    const text = await response.text();
    if (text) {
      try {
        data = JSON.parse(text);
      } catch (e) {
        data = null;
      }
    }
    if (!response.ok) {
      const err = new Error(data && data.detail ? detailText(data.detail) : `HTTP ${response.status}`);
      err.status = response.status;
      err.detail = data ? data.detail : null;  // structured details (e.g. a decision the user must make)
      throw err;
    }
    return data;
  }

  // ---------------------------------------------------------------- theme
  function initTheme() {
    const root = document.documentElement;
    const button = document.getElementById('theme-toggle');
    const current = () => {
      if (root.dataset.theme) return root.dataset.theme;
      return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    };
    const paint = () => {
      if (!button) return;
      button.replaceChildren(icon(current() === 'dark' ? 'sun' : 'moon', 20));
      button.setAttribute('aria-pressed', current() === 'dark' ? 'true' : 'false');
    };
    if (button) {
      button.addEventListener('click', () => {
        const next = current() === 'dark' ? 'light' : 'dark';
        root.dataset.theme = next;
        try {
          localStorage.setItem('vl-theme', next);
        } catch (e) {
          /* storage blocked: the choice lasts for this page only */
        }
        paint();
      });
    }
    paint();
  }

  // ---------------------------------------------------------------- shared widgets
  /** Four large selectable cards for the task. Returns {node, get, set}. */
  function taskPicker(name, value) {
    let current = value || 'detection';
    const node = el('div', { class: 'task-cards', role: 'radiogroup' });
    const draw = () => {
      node.replaceChildren(
        ...TASKS.map((task) => {
          const id = `${name}-${task}`;
          const input = el('input', { type: 'radio', name, id, value: task, class: 'sr-only' });
          input.checked = task === current;
          input.addEventListener('change', () => {
            current = task;
            draw();
            node.querySelector(`input[value="${task}"]`).focus();
          });
          return el(
            'label',
            { class: `task-card${task === current ? ' selected' : ''}`, for: id },
            input,
            icon(task, 28),
            el('span', { class: 'task-card-title' }, taskName(task)),
            el('span', { class: 'task-card-help' }, t(`task.${task}.help`)),
          );
        }),
      );
    };
    draw();
    return { node, get: () => current, set: (v) => { current = v; draw(); } };
  }

  /** Chips input: type a name and press Enter (or comma); click x to remove. */
  function chipsInput(id, initial) {
    const items = [...(initial || [])];
    const list = el('ul', { class: 'chips' });
    const input = el('input', { type: 'text', id, autocomplete: 'off', placeholder: t('classes.placeholder'), 'aria-describedby': `${id}-help` });
    const wrap = el('div', { class: 'chips-box' }, list, input);
    const add = (raw) => {
      for (const part of String(raw).split(',')) {
        const name = part.trim();
        if (name && !items.some((i) => i.toLowerCase() === name.toLowerCase())) items.push(name);
      }
      draw();
    };
    const draw = () => {
      list.replaceChildren(
        ...items.map((name, i) =>
          el(
            'li',
            { class: 'chip' },
            el('span', null, name),
            el('button', {
              type: 'button',
              class: 'chip-x',
              'aria-label': t('classes.remove', { name }),
              onclick: () => { items.splice(i, 1); draw(); input.focus(); },
            }, icon('close', 12)),
          ),
        ),
      );
    };
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ',') {
        e.preventDefault();
        if (input.value.trim()) { add(input.value); input.value = ''; }
      } else if (e.key === 'Backspace' && !input.value && items.length) {
        items.pop();
        draw();
      }
    });
    input.addEventListener('blur', () => { if (input.value.trim()) { add(input.value); input.value = ''; } });
    wrap.addEventListener('click', () => input.focus());
    draw();
    const node = el('div', null, wrap, el('p', { id: `${id}-help`, class: 'sr-only' }, t('classes.help')));
    return {
      node,
      get: () => {
        if (input.value.trim()) { add(input.value); input.value = ''; }
        return [...items];
      },
    };
  }

  const taskBadge = (task) => el('span', { class: `badge task-${task}` }, icon(task, 14), taskName(task));

  /* Class chips; long lists show the first ones and a button for the rest. */
  const CLASSES_SHOWN = 12;
  function classList(classes) {
    const chip = (c) => el('span', { class: 'chip static' }, c);
    const box = el('p', { class: 'class-list' }, el('span', { class: 'muted' }, `${t('field.classes')} (${num(classes.length)}): `),
      ...classes.slice(0, CLASSES_SHOWN).map(chip));
    if (classes.length > CLASSES_SHOWN) {
      const rest = classes.slice(CLASSES_SHOWN).map(chip);
      const toggle = el('button', { type: 'button', class: 'btn small', 'aria-expanded': 'false' });
      const collapse = (open) => {
        rest.forEach((c) => { c.hidden = !open; });
        toggle.textContent = open ? t('classes.less') : `+${num(rest.length)}`;
        toggle.setAttribute('aria-expanded', String(open));
      };
      toggle.addEventListener('click', () => collapse(toggle.getAttribute('aria-expanded') !== 'true'));
      box.append(...rest, toggle);
      collapse(false);
    }
    return box;
  }

  function progressBar(done, total, label) {
    const pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;
    const bar = el('div', { class: 'progress', role: 'progressbar', 'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-valuenow': String(pct), 'aria-label': label || t('common.progress') },
      el('div', { class: 'progress-fill', style: `width:${pct}%` }));
    return bar;
  }

  /** A question with several answers; resolves with the chosen value, or null when cancelled. */
  function choiceDialog({ title, text, choices }) {
    return new Promise((resolve) => {
      const dlg = el('dialog', { class: 'dialog wide', 'aria-label': title });
      let value = null;
      dlg.append(el('div', { class: 'dialog-body' },
        el('h2', null, title),
        el('p', null, text),
        el('ul', { class: 'choices' }, ...choices.map((c) => el('li', null,
          el('button', { type: 'button', class: `btn ${c.danger ? 'danger-outline' : c.primary ? 'primary' : ''}`, onclick: () => { value = c.value; dlg.close(); } }, c.label),
          c.help ? el('span', { class: 'muted' }, c.help) : null))),
        el('div', { class: 'form-actions' }, el('button', { type: 'button', class: 'btn', onclick: () => dlg.close() }, t('common.cancel')))));
      dlg.addEventListener('close', () => { dlg.remove(); resolve(value); });
      document.body.appendChild(dlg);
      dlg.showModal();
    });
  }

  function confirmDialog({ title, text, confirmLabel, danger = true, requireText = null }) {
    return new Promise((resolve) => {
      const input = requireText
        ? el('input', { type: 'text', 'aria-label': t('delete.type_name'), autocomplete: 'off' })
        : null;
      const okBtn = el('button', { type: 'submit', value: 'ok', class: `btn ${danger ? 'danger' : 'primary'}` }, confirmLabel);
      if (input) {
        okBtn.disabled = true;
        input.addEventListener('input', () => { okBtn.disabled = input.value !== requireText; });
      }
      const dlg = el('dialog', { class: 'dialog', 'aria-label': title },
        el('form', { method: 'dialog', class: 'dialog-body' },
          el('h2', null, title),
          el('p', null, text),
          input,
          el('div', { class: 'form-actions' },
            okBtn,
            el('button', { type: 'button', class: 'btn', onclick: () => dlg.close('cancel') }, t('common.cancel')))));
      dlg.addEventListener('close', () => { dlg.remove(); resolve(dlg.returnValue === 'ok'); });
      document.body.appendChild(dlg);
      dlg.showModal();
      (input || okBtn).focus();
    });
  }

  // ---------------------------------------------------------------- page: projects list
  async function initProjects() {
    const root = document.getElementById('projects-root');
    let list;
    try {
      list = await api('GET', '/projects');
    } catch (err) {
      root.replaceChildren(el('p', { class: 'muted' }, t('common.load_failed')));
      toastError(err);
      return;
    }
    const projects = Array.isArray(list) ? list : (list && list.items) || [];
    if (!projects.length) {
      root.replaceChildren(
        el('div', { class: 'empty card' },
          el('h2', null, t('projects.empty_title')),
          el('p', null, t('projects.empty_text')),
          el('a', { class: 'btn primary big', href: '/projects/new' }, t('projects.create_first'))));
      return;
    }
    root.replaceChildren(el('ul', { class: 'project-grid' }, ...projects.map(projectCard)));
  }

  function annotationStatus(c) {
    return c.annotated > 0 ? tn('status.annotated', c.annotated) : t('status.annotations_none');
  }

  function projectCard(p) {
    const c = p.counts || {};
    return el('li', null,
      el('a', { class: 'project-card card', href: `/projects/${p.id}` },
        el('div', { class: 'project-card-top' },
          el('h2', null, p.name),
          el('span', { class: 'project-id' }, `#${p.id}`)),
        taskBadge(p.task),
        p.description ? el('p', { class: 'project-desc' }, p.description) : null,
        el('dl', { class: 'facts' },
          el('div', null, el('dt', null, t('project.images')), el('dd', null, `${tn('status.images_active', c.active || 0)}${c.excluded ? ` · ${tn('status.excluded', c.excluded)}` : ''}`)),
          el('div', null, el('dt', null, t('project.annotations')), el('dd', null, annotationStatus(c))),
          el('div', null, el('dt', null, t('project.created')), el('dd', null, fmtDate(p.created))))));
  }

  // ---------------------------------------------------------------- page: new project
  function initNew() {
    const form = document.getElementById('new-form');
    const picker = taskPicker('task', 'detection');
    document.getElementById('task-picker').replaceWith(picker.node);
    const chips = chipsInput('classes-input', []);
    document.getElementById('classes-field').replaceWith(chips.node);
    const btn = document.getElementById('create-btn');
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const name = form.elements.name.value.trim();
      if (!name) {
        toastError(new Error(t('new.name_required')));
        form.elements.name.focus();
        return;
      }
      btn.disabled = true;
      try {
        const project = await api('POST', '/projects', {
          json: { name, description: form.elements.description.value.trim(), task: picker.get(), classes: chips.get() },
        });
        location.href = `/projects/${project.id}`;
      } catch (err) {
        btn.disabled = false;
        toastError(err);
      }
    });
    form.elements.name.focus();
  }

  // ---------------------------------------------------------------- thumbnail grid
  /**
   * Paged grid of thumbnails. `query()` returns {status, issue}. With `selectable`, every card
   * has a checkbox and `selected` (a Set of ids) is kept up to date.
   */
  function imageGrid({ projectId, query, selectable = false, onSelect, onOpen }) {
    const selected = new Set();
    const items = [];
    let total = 0;
    let loading = false;
    let token = 0;
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
          el('img', { src: `/api/projects/${projectId}/images/${img.id}/thumb`, alt: img.name, loading: 'lazy', width: '160', height: '160' })),
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
      pager.replaceChildren(
        total > size ? el('span', { class: 'pager-pages' },
          btn('‹', page - 1, { disabled: page === 0, aria: t('grid.prev') }), ...numbers,
          btn('›', page + 1, { disabled: page >= pages - 1, aria: t('grid.next') })) : el('span'),
        total > SIZES[0] ? el('label', { class: 'pager-size' }, t('grid.per_page'), sizeSelect) : el('span'));
    }

    async function load(reset) {
      if (loading) return;
      loading = true;
      const mine = ++token;
      if (reset) { page = 0; selected.clear(); fire(); }
      const q = query();
      const params = new URLSearchParams({ status: q.status || 'active', offset: String(page * size), limit: String(size) });
      if (q.issue) params.set('issue', q.issue);
      try {
        const data = await api('GET', `/projects/${projectId}/images?${params}`);
        if (mine !== token) return;
        total = data.total || 0;
        if (page > 0 && page * size >= total) { page = Math.max(0, Math.ceil(total / size) - 1); loading = false; return load(false); }
        items.length = 0;
        list.replaceChildren();
        (data.items || []).forEach((img, i) => { items.push(img); list.appendChild(card(img, i)); });
      } catch (err) {
        toastError(err);
      } finally {
        loading = false;
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

  // ---------------------------------------------------------------- lightbox
  let lightbox = null;
  function openLightbox(projectId, items, index) {
    if (!lightbox) {
      lightbox = el('dialog', { class: 'lightbox', 'aria-label': t('lightbox.title') });
      document.body.appendChild(lightbox);
      lightbox.addEventListener('click', (e) => { if (e.target === lightbox) lightbox.close(); });
    }
    let i = index;
    const draw = () => {
      const img = items[i];
      const metrics = Object.entries(img.metrics || {}).filter(([, v]) => typeof v === 'number');
      lightbox.replaceChildren(
        el('div', { class: 'lightbox-body' },
          el('div', { class: 'lightbox-stage' },
            el('button', { type: 'button', class: 'btn lb-nav', 'aria-label': t('lightbox.prev'), disabled: i === 0, onclick: () => { i -= 1; draw(); } }, '‹'),
            el('img', { src: `/api/projects/${projectId}/images/${img.id}/file`, alt: img.name }),
            el('button', { type: 'button', class: 'btn lb-nav', 'aria-label': t('lightbox.next'), disabled: i >= items.length - 1, onclick: () => { i += 1; draw(); } }, '›')),
          el('aside', { class: 'lightbox-side' },
            el('div', { class: 'lightbox-head' },
              el('h2', null, img.name),
              el('button', { type: 'button', class: 'icon-btn', 'aria-label': t('common.close'), onclick: () => lightbox.close() }, icon('close'))),
            el('p', { class: 'muted' }, `${num(img.width)} × ${num(img.height)} px · ${fmtBytes(img.size_bytes || 0)}`),
            img.status === 'excluded' ? el('p', null, el('span', { class: 'tag tag-excluded' }, t('grid.excluded')), img.excluded_reason ? ` ${img.excluded_reason}` : '') : null,
            el('h3', null, t('lightbox.issues')),
            (img.issues || []).length
              ? el('p', { class: 'thumb-tags' }, ...img.issues.map((x) => el('span', { class: `tag tag-${x}`, title: tr(`issue.${x}.help`, '') }, issueTag(x))))
              : el('p', { class: 'muted' }, t('lightbox.no_issues')),
            metrics.length ? el('h3', null, t('lightbox.metrics')) : null,
            metrics.length
              ? el('dl', { class: 'facts compact' }, ...metrics.map(([k, v]) => el('div', null, el('dt', null, tr(`metric.${k}`, k)), el('dd', null, nf1.format(v)))))
              : null)));
    };
    draw();
    if (!lightbox.open) lightbox.showModal();
    lightbox.onkeydown = (e) => {
      if (e.key === 'ArrowLeft' && i > 0) { i -= 1; draw(); }
      if (e.key === 'ArrowRight' && i < items.length - 1) { i += 1; draw(); }
    };
  }

  // ---------------------------------------------------------------- page: project
  function initProject() {
    const root = document.getElementById('project-root');
    const projectId = VL.projectId;
    const ctx = {
      id: projectId,
      project: null,
      tab: 'images',
      jobs: [],
      jobStatus: new Map(),
      openWhenDone: new Set(),
      pollTimer: null,
      panels: {},
    };
    const STEPS = ['images', 'annotations', 'curation'];

    // ---- header, stepper, panels skeleton
    const headerBox = el('header', { class: 'project-head card' });
    const stepper = el('nav', { class: 'stepper', 'aria-label': t('project.steps') });
    const panelsBox = el('div', { class: 'panels' });
    const jobsBox = el('section', { class: 'jobs-bar', 'aria-label': t('jobs.title') });

    async function refreshProject() {
      try {
        ctx.project = await api('GET', `/projects/${projectId}`);
      } catch (err) {
        if (err.status === 404) { showMissing(); return false; }
        toastError(err);
        return false;
      }
      document.title = `${ctx.project.name} · ${t('app.name')}`;
      renderHeader();
      if (ctx.tab) showTab(ctx.tab); else renderStepper();  // opens a gated step once images arrive
      return true;
    }

    function showMissing() {
      root.replaceChildren(el('div', { class: 'empty card' },
        el('h1', null, t('notfound.title')),
        el('p', null, t('notfound.text')),
        el('a', { class: 'btn primary', href: '/' }, t('notfound.back'))));
    }

    function renderHeader() {
      const p = ctx.project;
      headerBox.replaceChildren(
        el('div', { class: 'project-head-main' },
          el('div', { class: 'project-title-row' },
            el('h1', null, p.name),
            el('span', { class: 'project-id' }, `#${p.id}`),
            taskBadge(p.task)),
          p.description ? el('p', { class: 'muted' }, p.description) : null,
          (p.classes || []).length ? classList(p.classes) : null),
        el('div', { class: 'project-head-actions' },
          el('button', { type: 'button', class: 'btn', onclick: openEdit }, t('project.edit')),
          el('button', { type: 'button', class: 'btn danger-outline', onclick: openDelete }, t('project.delete'))));
    }

    function stepStatus(step) {
      const c = (ctx.project && ctx.project.counts) || {};
      if (step === 'images') {
        if (!(c.active || c.excluded)) return t('status.images_none');
        return `${tn('status.images_active', c.active || 0)}${c.excluded ? ` · ${tn('status.excluded', c.excluded)}` : ''}`;
      }
      if (step === 'annotations') {
        return c.annotated > 0
          ? `${tn('status.annotated', c.annotated)} · ${tn('status.classes', (ctx.project.classes || []).length)}`
          : t('status.annotations_none');
      }
      return c.analyzed > 0 ? tn('status.issues', c.with_issues || 0) : t('status.not_analyzed');
    }

    function renderStepper() {
      stepper.replaceChildren(
        ...STEPS.map((step, i) =>
          el('button', {
            type: 'button',
            class: `step${ctx.tab === step ? ' current' : ''}`,
            'aria-current': ctx.tab === step ? 'step' : null,
            'data-step': step,
            onclick: () => showTab(step),
          },
            el('span', { class: 'step-num', 'aria-hidden': 'true' }, String(i + 1)),
            el('span', { class: 'step-text' },
              el('span', { class: 'step-title' }, t(`step.${step}`), step === 'annotations' ? el('span', { class: 'optional' }, ` · ${t('step.optional')}`) : null),
              el('span', { class: 'step-status' }, stepStatus(step))))));
    }

    /* steps 2 and 3 work on the project's images: without images they only say so */
    const hasImages = () => ((ctx.project && ctx.project.counts && ctx.project.counts.active) || 0) > 0;
    const gateNode = el('section', { class: 'card gate', hidden: true },
      el('p', { class: 'report-line' }, icon('info'), t('gate.need_images')),
      el('button', { type: 'button', class: 'btn primary', onclick: () => showTab('images') }, t('gate.go_images')));

    function showTab(step, scrollToPanel = false) {
      ctx.tab = step;
      history.replaceState(null, '', `#${step}`);
      const gated = step !== 'images' && !hasImages();
      for (const s of STEPS) ctx.panels[s].node.hidden = gated || s !== step;
      gateNode.hidden = !gated;
      renderStepper();
      if (gated) return;
      const panel = ctx.panels[step];
      if (!panel.mounted) { panel.mounted = true; panel.mount(); }
      if (scrollToPanel) panel.node.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    const afterChange = async () => {
      await refreshProject();
    };

    // ---- edit & delete
    function openEdit() {
      const p = ctx.project;
      const nameInput = el('input', { id: 'e-name', type: 'text', value: p.name, maxlength: '100', required: true });
      const descInput = el('textarea', { id: 'e-desc', rows: '3', maxlength: '2000' }, p.description || '');
      const picker = taskPicker('e-task', p.task);
      const chips = chipsInput('e-classes', p.classes || []);
      const save = el('button', { type: 'submit', class: 'btn primary' }, t('common.save'));
      const dlg = el('dialog', { class: 'dialog wide', 'aria-label': t('project.edit') },
        el('form', { class: 'dialog-body form', method: 'dialog' },
          el('h2', null, t('project.edit')),
          el('div', { class: 'field' }, el('label', { for: 'e-name' }, t('field.name')), nameInput),
          el('div', { class: 'field' }, el('label', { for: 'e-desc' }, t('field.description')), descInput),
          el('fieldset', { class: 'field' }, el('legend', null, t('field.task')), picker.node),
          el('div', { class: 'field' }, el('label', { for: 'e-classes' }, t('field.classes')), chips.node),
          el('div', { class: 'form-actions' }, save, el('button', { type: 'button', class: 'btn', onclick: () => dlg.close() }, t('common.cancel')))));
      dlg.querySelector('form').addEventListener('submit', async (e) => {
        e.preventDefault();
        save.disabled = true;
        const body = { name: nameInput.value.trim(), description: descInput.value.trim(), task: picker.get(), classes: chips.get() };
        try {
          /* the server may ask for decisions (task change, classes still in use); ask them one by one */
          let updated = null;
          const decisions = {};
          for (let round = 0; round < 3 && !updated; round += 1) {
            try {
              updated = await api('PATCH', `/projects/${projectId}`, { json: { ...body, ...decisions } });
            } catch (err) {
              const d = err.detail;
              if (err.status !== 409 || !d || !d.code) throw err;
              if (d.code === 'task_change_annotations') {
                const choice = await choiceDialog({
                  title: t('taskchange.title'),
                  text: t('taskchange.text', { from: taskName(d.from_task), to: taskName(d.to_task), images: num(d.images), labels: num(d.labels) }),
                  choices: [
                    ...(d.options.includes('convert') ? [{ value: 'convert', primary: true, label: t('taskchange.convert', { to: taskName(d.to_task) }), help: t(`taskchange.convert_help.${d.to_task}`) }] : []),
                    { value: 'discard', danger: true, label: t('taskchange.discard'), help: t('taskchange.discard_help') },
                  ],
                });
                if (!choice) { save.disabled = false; return; }
                decisions.annotations = choice;
              } else if (d.code === 'classes_in_use') {
                const list = (d.classes || []).map((c) => t('classes.in_use_item', { name: c.name, labels: num(c.labels), images: num(c.images) })).join('; ');
                const ok = await confirmDialog({
                  title: t('classes.in_use_title'),
                  text: t('classes.in_use_text', { list }),
                  confirmLabel: t('classes.in_use_remove'),
                });
                if (!ok) { save.disabled = false; return; }
                decisions.classes_action = 'remove_annotations';
              } else {
                throw err;
              }
            }
          }
          if (!updated) { save.disabled = false; return; }
          ctx.project = { ...ctx.project, ...updated };
          dlg.close();
          toast(t('project.saved'), 'success', 3000);
          await refreshProject();
          Object.values(ctx.panels).forEach((panel) => panel.refresh && panel.refresh());
        } catch (err) {
          save.disabled = false;
          toastError(err);
        }
      });
      dlg.addEventListener('close', () => dlg.remove());
      document.body.appendChild(dlg);
      dlg.showModal();
      nameInput.focus();
    }

    async function openDelete() {
      const p = ctx.project;
      const ok = await confirmDialog({
        title: t('delete.title'),
        text: t('delete.text', { name: p.name }),
        confirmLabel: t('delete.confirm'),
        requireText: p.name,
      });
      if (!ok) return;
      try {
        await api('DELETE', `/projects/${projectId}`, { json: { confirm_name: p.name } });
        location.href = '/';
      } catch (err) {
        toastError(err);
      }
    }

    // ================================================================ step 1: images
    function imagesPanel() {
      const node = el('section', { class: 'panel', id: 'panel-images', 'aria-labelledby': 'h-images' });
      const panel = { node, mounted: false };
      let uploading = false;
      let controller = null;

      const fileInput = el('input', { type: 'file', multiple: true, class: 'sr-only', id: 'file-input', accept: 'image/*,.zip,application/zip', tabindex: '-1' });
      const folderInput = el('input', { type: 'file', multiple: true, class: 'sr-only', id: 'folder-input', tabindex: '-1' });
      folderInput.setAttribute('webkitdirectory', '');
      folderInput.setAttribute('directory', '');
      const progressBox = el('div', { class: 'upload-progress', hidden: true });
      const reportBox = el('div', { class: 'report', 'aria-live': 'polite' });

      const drop = el('div', { class: 'dropzone', id: 'dropzone' },
        icon('upload', 36),
        el('p', { class: 'drop-title' }, t('images.drop_title')),
        el('p', { class: 'muted' }, t('images.drop_help')),
        el('div', { class: 'drop-actions' },
          el('button', { type: 'button', class: 'btn primary', onclick: () => fileInput.click() }, t('images.choose_files')),
          el('button', { type: 'button', class: 'btn', onclick: () => folderInput.click() }, t('images.choose_folder'))));
      ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
      ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
      drop.addEventListener('drop', async (e) => {
        const entries = await filesFromDrop(e.dataTransfer);
        startUpload(entries);
      });
      fileInput.addEventListener('change', () => { startUpload([...fileInput.files].map((f) => ({ file: f, name: f.webkitRelativePath || f.name }))); fileInput.value = ''; });
      folderInput.addEventListener('change', () => { startUpload([...folderInput.files].map((f) => ({ file: f, name: f.webkitRelativePath || f.name }))); folderInput.value = ''; });

      async function filesFromDrop(dt) {
        const out = [];
        const items = dt.items ? [...dt.items] : [];
        if (items.length && items[0].webkitGetAsEntry) {
          const roots = items.map((i) => i.webkitGetAsEntry()).filter(Boolean); // before any await
          const walk = async (entry, prefix) => {
            if (entry.isFile) {
              const file = await new Promise((res, rej) => entry.file(res, rej));
              out.push({ file, name: prefix + entry.name });
            } else if (entry.isDirectory) {
              const reader = entry.createReader();
              for (;;) {
                const batch = await new Promise((res, rej) => reader.readEntries(res, rej));
                if (!batch.length) break;
                for (const child of batch) await walk(child, `${prefix}${entry.name}/`);
              }
            }
          };
          for (const r of roots) await walk(r, '');
          return out;
        }
        return [...dt.files].map((f) => ({ file: f, name: f.name }));
      }

      const emptyReport = () => ({ added: 0, duplicates: [], duplicates_count: 0, rejected: [], rejected_count: 0, other_files: [], other_files_count: 0 });
      const mergeReport = (acc, r) => {
        acc.added += Number(r.added) || 0;
        for (const key of ['duplicates', 'rejected', 'other_files']) {
          const list = r[key] || [];
          acc[key].push(...list);
          acc[`${key}_count`] += Number(r[`${key}_count`]) || list.length;
        }
      };

      async function startUpload(entries) {
        const files = entries.filter((e) => !SKIP_FILES.has(e.name.split('/').pop().toLowerCase()));
        if (!files.length || uploading) return;
        uploading = true;
        controller = new AbortController();
        reportBox.replaceChildren();
        const merged = emptyReport();
        let sent = 0;
        let cancelled = false;
        const draw = () => {
          progressBox.hidden = false;
          progressBox.replaceChildren(
            el('div', { class: 'upload-line' },
              el('strong', null, t('images.uploading', { done: num(sent), total: num(files.length) })),
              el('button', { type: 'button', class: 'btn small', onclick: () => controller.abort() }, t('common.cancel'))),
            progressBar(sent, files.length, t('images.upload_progress')));
        };
        draw();
        for (let i = 0; i < files.length; i += CHUNK) {
          const chunk = files.slice(i, i + CHUNK);
          const form = new FormData();
          chunk.forEach((e) => form.append('files', e.file, e.name));
          try {
            mergeReport(merged, await api('POST', `/projects/${projectId}/images`, { body: form, signal: controller.signal }));
          } catch (err) {
            if (err.name === 'AbortError') cancelled = true; else toastError(err);
            break;
          }
          sent += chunk.length;
          draw();
        }
        uploading = false;
        progressBox.hidden = true;
        if (cancelled) toast(t('images.cancelled', { done: num(sent) }), 'info');
        renderReport(merged, sent < files.length);
        await afterChange();
        grid.reload();
        if (ctx.panels.curation.mounted) ctx.panels.curation.refresh();
      }

      const reasonText = (reason) => tr(`reason.${String(reason).toLowerCase().replace(/[^a-z]+/g, '_')}`, String(reason));
      const nameOf = (x) => (typeof x === 'string' ? x : (x && x.name) || '');

      function expandable(title, rows) {
        return el('details', { class: 'expand' }, el('summary', null, title), el('ul', { class: 'plain scroll-box' }, ...rows));
      }

      function renderReport(r, incomplete) {
        const rows = [];
        rows.push(el('p', { class: 'report-line ok' }, icon('check'), tn('images.added', r.added)));
        if (r.duplicates_count) {
          rows.push(el('div', { class: 'report-line' }, icon('info'),
            el('div', null, tn('images.duplicates', r.duplicates_count), ' ', el('span', { class: 'muted' }, t('images.duplicates_note')),
              expandable(t('images.show_list'), r.duplicates.map((d) => el('li', null, nameOf(d)))))));
        }
        if (r.rejected_count) {
          rows.push(el('div', { class: 'report-line bad' }, icon('warning'),
            el('div', null, tn('images.rejected', r.rejected_count),
              expandable(t('images.show_list'), r.rejected.map((x) => el('li', null, el('span', null, nameOf(x)), ' ', el('span', { class: 'muted' }, `— ${reasonText(x.reason || '')}`)))))));
        }
        if (incomplete) rows.push(el('p', { class: 'muted' }, t('images.incomplete')));
        const annotationLike = (r.other_files || []).some((f) => /\.(json|txt|xml|ya?ml|csv)$/i.test(nameOf(f)));
        if (r.other_files_count && annotationLike) {
          rows.push(el('div', { class: 'callout' }, icon('info'),
            el('div', null,
              el('strong', null, t('images.annotations_found')),
              el('p', { class: 'muted' }, tn('images.annotations_found_n', r.other_files_count)),
              el('button', { type: 'button', class: 'btn primary', onclick: () => showTab('annotations', true) }, t('images.go_annotations')))));
        }
        reportBox.replaceChildren(el('div', { class: 'card report-card' }, el('h3', null, t('images.result')), ...rows));
      }

      const grid = imageGrid({
        projectId,
        query: () => ({ status: 'active', emptyText: t('grid.no_images') }),
        onOpen: (items, i) => openLightbox(projectId, items, i),
      });

      node.append(
        el('h2', { id: 'h-images' }, t('step.images')),
        el('p', { class: 'muted' }, t('images.intro')),
        drop, fileInput, folderInput, progressBox, reportBox,
        el('h3', { class: 'section-title' }, t('images.grid_title')),
        grid.node);

      panel.mount = () => grid.reload();
      panel.refresh = () => grid.reload();
      return panel;
    }

    // ================================================================ step 2: annotations
    function annotationsPanel() {
      const node = el('section', { class: 'panel', id: 'panel-annotations', hidden: true, 'aria-labelledby': 'h-annotations' });
      const panel = { node, mounted: false };
      const fileInput = el('input', { type: 'file', multiple: true, class: 'sr-only', id: 'ann-input', tabindex: '-1' });
      const reportBox = el('div', { class: 'ann-report', 'aria-live': 'polite' });
      const gtBox = el('div', { class: 'gt-box' });
      let busy = false;

      const drop = el('div', { class: 'dropzone small', id: 'ann-drop' },
        icon('upload', 28),
        el('p', { class: 'drop-title' }, t('ann.drop_title')),
        el('p', { class: 'muted' }, t('ann.drop_help')),
        el('button', { type: 'button', class: 'btn primary', onclick: () => fileInput.click() }, t('ann.choose')));
      ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
      ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
      drop.addEventListener('drop', (e) => upload([...e.dataTransfer.files]));
      fileInput.addEventListener('change', () => { upload([...fileInput.files]); fileInput.value = ''; });

      async function upload(files) {
        if (!files.length || busy) return;
        busy = true;
        reportBox.replaceChildren(el('p', { class: 'muted' }, t('ann.checking')));
        const form = new FormData();
        files.forEach((f) => form.append('files', f, f.webkitRelativePath || f.name));
        try {
          const report = await api('POST', `/projects/${projectId}/annotations`, { body: form });
          renderValidation(report);
        } catch (err) {
          reportBox.replaceChildren();
          toastError(err);
        } finally {
          busy = false;
        }
      }

      function problemRow(p) {
        const message = tr(`problem.${p.code}`, p.message || p.code);
        const examples = (p.examples || []).slice(0, 5);
        const head = el('span', { class: `problem-head ${p.level === 'error' ? 'bad' : 'warn'}` },
          icon(p.level === 'error' ? 'error' : 'warning'),
          el('span', { class: 'problem-msg' }, message),
          p.count ? el('span', { class: 'count' }, `×${num(p.count)}`) : null);
        if (!examples.length) return el('li', { class: 'problem' }, head);
        return el('li', { class: 'problem' },
          el('details', null, el('summary', null, head),
            el('ul', { class: 'plain examples' }, ...examples.map((x) => el('li', null, el('code', null, x))))));
      }

      function renderValidation(report) {
        const found = report.inferred_task;
        const proj = ctx.project.task;
        const counts = report.counts || {};
        const hasExisting = ((ctx.project.counts || {}).annotated || 0) > 0;
        let mode = 'merge';

        let taskLine;
        if (!found) taskLine = t('ann.task_none');
        else if (sameTask(found, proj)) taskLine = t('ann.task_same', { task: taskName(found) });
        else taskLine = t('ann.task_diff', { found: taskName(found), project: taskName(proj) });

        const classes = report.classes_found || [];
        const fresh = new Set(report.classes_new || []);
        const perClass = counts.labels_per_class || {};

        const optionMeta = {
          import: { label: t('option.import'), help: t('option.import.help'), primary: true },
          change_task_and_import: { label: t('option.change_task_and_import', { task: taskName(found) }), help: t('option.change_task_and_import.help', { task: taskName(found) }) },
          convert_and_import: { label: t('option.convert_and_import', { task: taskName(proj) }), help: t('option.convert_and_import.help', { task: taskName(proj) }) },
          discard: { label: t('option.discard'), help: t('option.discard.help') },
        };
        const options = (report.options || []).filter((o) => optionMeta[o]);
        const needsMode = (o) => o !== 'discard' && hasExisting;

        const modeBox = el('fieldset', { class: 'mode-box' },
          el('legend', null, t('ann.mode_title')),
          ...['merge', 'replace'].map((m) =>
            el('label', { class: 'radio-row' },
              el('input', { type: 'radio', name: 'ann-mode', value: m, checked: m === mode, onchange: () => { mode = m; } }),
              el('span', null, el('strong', null, t(`ann.mode_${m}`)), el('span', { class: 'muted' }, ` — ${t(`ann.mode_${m}.help`)}`)))));

        const optionList = el('ul', { class: 'options' },
          ...options.map((o) => {
            const meta = optionMeta[o];
            return el('li', null,
              el('button', { type: 'button', class: `btn ${meta.primary ? 'primary' : o === 'discard' ? 'danger-outline' : ''}`, 'data-action': o, onclick: (e) => apply(report, o, needsMode(o) ? mode : 'replace', e.currentTarget) }, meta.label),
              el('span', { class: 'muted' }, meta.help));
          }));

        reportBox.replaceChildren(
          el('div', { class: 'card report-card' },
            el('h3', null, t('ann.report_title')),
            el('p', { class: 'task-line' }, el('span', { class: 'badge' }, tr(`format.${report.format}`, report.format || '?')), ' ', taskLine),
            el('dl', { class: 'facts' },
              el('div', null, el('dt', null, t('ann.matched')), el('dd', null, tn('ann.images_matched', counts.images_matched || 0))),
              el('div', null, el('dt', null, t('ann.unlabeled')), el('dd', null, tn('ann.images_without_labels', counts.images_without_labels || 0))),
              el('div', null, el('dt', null, t('ann.labels')), el('dd', null, num(counts.labels_total || 0)))),
            report.unmatched && report.unmatched.count
              ? el('p', { class: 'muted' }, tn('ann.unmatched', report.unmatched.count))
              : null,
            el('h4', null, t('ann.classes_found')),
            classes.length
              ? el('ul', { class: 'chips static' }, ...classes.map((c) => el('li', { class: `chip static${fresh.has(c) ? ' is-new' : ''}` },
                  el('span', null, c),
                  perClass[c] != null ? el('span', { class: 'count' }, num(perClass[c])) : null,
                  fresh.has(c) ? el('span', { class: 'new-mark' }, t('ann.new')) : null)))
              : el('p', { class: 'muted' }, t('ann.no_classes')),
            el('h4', null, t('ann.problems')),
            (report.problems || []).length
              ? el('ul', { class: 'plain problems' }, ...report.problems.map(problemRow))
              : el('p', { class: 'report-line ok' }, icon('check'), t('ann.no_problems')),
            el('h4', null, t('ann.what_now')),
            options.some(needsMode) ? modeBox : null,
            optionList));
      }

      async function apply(report, action, mode, button) {
        const all = reportBox.querySelectorAll('button');
        all.forEach((b) => { b.disabled = true; });
        try {
          if (action === 'discard') {
            await api('DELETE', `/projects/${projectId}/annotations/${report.token}`);
            reportBox.replaceChildren(el('p', { class: 'report-line ok' }, icon('check'), t('ann.discarded')));
            return;
          }
          const summary = await api('POST', `/projects/${projectId}/annotations/${report.token}/apply`, { json: { action, mode } });
          renderSummary(summary || {});
          await afterChange();
          loadGt();
        } catch (err) {
          all.forEach((b) => { b.disabled = false; });
          toastError(err);
        }
        if (button) button.blur();
      }

      function renderSummary(s) {
        /* what the person needs to know, in plain words; zero counts of optional facts are left out */
        const always = ['images_imported', 'labels_imported'];
        const optional = ['images_kept', 'converted', 'classes_added', 'problems'];
        const count = (v) => (Array.isArray(v) ? v.length : Number(v) || 0);
        const lines = [...always, ...optional]
          .filter((k) => k in s && (always.includes(k) || count(s[k]) > 0))
          .map((k) => el('div', null, el('dt', null, t(`summary.${k}`)), el('dd', null, num(count(s[k])))));
        reportBox.replaceChildren(el('div', { class: 'card report-card' },
          el('p', { class: 'report-line ok' }, icon('check'), el('strong', null, t('ann.applied'))),
          lines.length ? el('dl', { class: 'facts' }, ...lines) : null));
      }

      async function loadGt() {
        let data;
        try {
          data = await api('GET', `/projects/${projectId}/annotations`);
        } catch (err) {
          gtBox.replaceChildren();
          if (err.status !== 404) toastError(err);
          return;
        }
        data = data || {};
        let classes = data.classes || data.labels_per_class || data.per_class || [];
        if (!Array.isArray(classes)) classes = Object.entries(classes).map(([name, count]) => ({ name, count }));
        classes = classes.map((c) => (typeof c === 'string' ? { name: c, count: 0 } : c));
        const labels = data.labels_total ?? data.labels ?? classes.reduce((a, c) => a + (c.count || 0), 0);
        const images = data.images_annotated ?? data.images_with_labels ?? data.images ?? 0;
        if (!images && !labels) {
          gtBox.replaceChildren(el('h3', { class: 'section-title' }, t('ann.current')), el('p', { class: 'muted' }, t('ann.current_none')));
          return;
        }
        const max = Math.max(1, ...classes.map((c) => c.count || 0));
        gtBox.replaceChildren(
          el('h3', { class: 'section-title' }, t('ann.current')),
          el('p', null, `${tn('status.annotated', images)} · ${tn('ann.labels_n', labels)}`),
          classes.length
            ? el('ul', { class: 'bars' }, ...classes.map((c) =>
                el('li', null,
                  el('span', { class: 'bar-label', title: c.name }, c.name),
                  el('span', { class: 'bar-track' }, el('span', { class: 'bar-fill', style: `width:${Math.round(((c.count || 0) / max) * 100)}%` })),
                  el('span', { class: 'bar-value' }, num(c.count || 0)))))
            : el('span'));
      }

      node.append(
        el('h2', { id: 'h-annotations' }, t('step.annotations')),
        el('p', { class: 'muted' }, t('ann.intro')),
        drop, fileInput, reportBox, gtBox);
      panel.mount = () => loadGt();
      panel.refresh = () => loadGt();
      return panel;
    }

    // ================================================================ step 3: curation
    function curationPanel() {
      const node = el('section', { class: 'panel', id: 'panel-curation', hidden: true, 'aria-labelledby': 'h-curation' });
      const panel = { node, mounted: false };
      let summary = null;
      let issue = '';
      let view = 'active';
      const noticeBox = el('div', { class: 'cur-notice' });
      const chipsBox = el('div', { class: 'issue-chips', role: 'group', 'aria-label': t('cur.filter') });
      const histBox = el('div', { class: 'hists' });
      const advBox = el('details', { class: 'advanced' });
      const toolbar = el('div', { class: 'toolbar' });
      const tabs = el('div', { class: 'tabs', role: 'group', 'aria-label': t('cur.view') });
      const countText = el('span', { class: 'muted', 'aria-live': 'polite' });

      const grid = imageGrid({
        projectId,
        selectable: true,
        query: () => ({ status: view, issue: view === 'active' ? issue : '', emptyText: view === 'excluded' ? t('cur.no_excluded') : t('cur.no_match') }),
        onSelect: (sel) => { countText.textContent = sel.size ? tn('cur.selected', sel.size) : ''; drawToolbar(); },
        onOpen: (items, i) => openLightbox(projectId, items, i),
      });

      const analyzeBtn = (recompute) =>
        el('button', { type: 'button', class: `btn${recompute ? '' : ' primary'}`, onclick: () => analyze(recompute) }, recompute ? t('cur.reanalyze') : t('cur.analyze'));

      async function analyze(recompute) {
        try {
          const job = await api('POST', `/projects/${projectId}/curation/analyze`, { json: { recompute: !!recompute } });
          toast(t('cur.started'), 'info', 3000);
          jobStarted(job);
        } catch (err) {
          toastError(err);
        }
      }

      async function resetAnalysis() {
        const ok = await confirmDialog({ title: t('cur.reset_title'), text: t('cur.reset_text'), confirmLabel: t('cur.reset') });
        if (!ok) return;
        try {
          await api('DELETE', `/projects/${projectId}/curation`);
          toast(t('cur.reset_done'), 'success', 4000);
          await afterChange();
          load();
        } catch (err) { toastError(err); }
      }

      async function load() {
        try {
          summary = await api('GET', `/projects/${projectId}/curation`);
        } catch (err) {
          summary = null;
          if (err.status !== 404) toastError(err);
        }
        if (summary && ctx.project) {
          const ids = new Set();
          Object.values(summary.issue_ids || {}).forEach((l) => l.forEach((x) => ids.add(x)));
                    ctx.project.counts = { ...(ctx.project.counts || {}), analyzed: (summary.totals || {}).measured || 0, with_issues: ids.size };
          renderStepper();
        }
        draw();
        grid.reload();
      }

      const analyzed = () => summary && ((summary.totals || {}).measured || 0) > 0;

      function draw() {
        drawNotice();
        drawChips();
        drawHists();
        drawAdvanced();
        drawTabs();
        drawToolbar();
        histBox.hidden = !analyzed();
        chipsBox.hidden = !analyzed() || view !== 'active';
        advBox.hidden = !summary || !summary.thresholds;
      }

      function drawNotice() {
        if (analyzed()) {
          const totals = summary.totals || {};
          noticeBox.replaceChildren(el('div', { class: 'row-between' },
            el('p', { class: 'muted' }, t('cur.measured', { measured: num(totals.measured || 0), total: num(totals.active || 0) })),
            el('div', { class: 'row-actions' }, analyzeBtn(true),
              el('button', { type: 'button', class: 'btn danger-outline', onclick: resetAnalysis }, t('cur.reset')))));
        } else {
          noticeBox.replaceChildren(el('div', { class: 'callout' }, icon('info'),
            el('div', null, el('strong', null, t('cur.not_analyzed')), el('p', { class: 'muted' }, t('cur.intro')), analyzeBtn(false))));
        }
      }

      function drawChips() {
        const totals = (summary && summary.totals) || {};
        const counts = (summary && summary.issues) || {};
        const mk = (key, label, n, help) =>
          el('button', {
            type: 'button', class: `chip-btn${issue === key ? ' active' : ''}`, 'aria-pressed': issue === key ? 'true' : 'false',
            title: help || null, 'data-issue': key,
            onclick: () => { issue = key; drawChips(); grid.reload(); },
          }, el('span', null, label), el('span', { class: 'count' }, num(n)));
        chipsBox.replaceChildren(
          mk('', t('cur.all'), totals.active || 0, t('cur.all.help')),
          ...ISSUES.concat(Object.keys(counts).filter((k) => !ISSUES.includes(k))).map((k) => mk(k, issueName(k), counts[k] || 0, tr(`issue.${k}.help`, ''))));
      }

      function drawHists() {
        const h = (summary && summary.histograms) || {};
        histBox.replaceChildren(...['brightness', 'contrast', 'sharpness'].filter((m) => h[m] && (h[m].counts || []).length).map((m) => histogram(m, h[m])));
      }

      function histogram(metric, data) {
        const counts = data.counts;
        const edges = data.edges || [];
        const max = Math.max(1, ...counts);
        const w = 200;
        const hgt = 60;
        const bw = w / counts.length;
        const bars = counts.map((c, i) => {
          const bh = Math.round((c / max) * (hgt - 4));
          return `<rect x="${(i * bw + 0.5).toFixed(1)}" y="${hgt - bh}" width="${(bw - 1).toFixed(1)}" height="${bh}" rx="1"></rect>`;
        }).join('');
        const lo = edges.length ? nf1.format(edges[0]) : '';
        const hi = edges.length ? nf1.format(edges[edges.length - 1]) : '';
        const label = `${tr(`metric.${metric}`, metric)}: ${lo} – ${hi}`;
        return el('figure', { class: 'hist' },
          el('div', { html: `<svg viewBox="0 0 ${w} ${hgt}" width="100%" role="img" aria-label="${esc(label)}" preserveAspectRatio="none">${bars}</svg>` }),
          el('figcaption', null, el('strong', null, tr(`metric.${metric}`, metric)), el('span', { class: 'muted' }, ` ${lo} – ${hi}`)));
      }

      function drawAdvanced() {
        const th = summary && summary.thresholds;
        if (!th) { advBox.replaceChildren(); return; }
        const inputs = Object.entries(th).map(([k, v]) => {
          const id = `th-${k}`;
          return el('div', { class: 'field inline' }, el('label', { for: id }, tr(`threshold.${k}`, k)), el('input', { id, name: k, type: 'number', step: 'any', value: String(v) }));
        });
        const form = el('form', { class: 'th-form' }, el('p', { class: 'muted' }, t('cur.thresholds_help')), el('div', { class: 'th-grid' }, ...inputs),
          el('div', { class: 'form-actions' }, el('button', { type: 'submit', class: 'btn primary' }, t('cur.thresholds_apply'))));
        form.addEventListener('submit', async (e) => {
          e.preventDefault();
          const body = {};
          for (const input of form.querySelectorAll('input')) body[input.name] = Number(input.value);
          try {
            summary = await api('PUT', `/projects/${projectId}/curation/thresholds`, { json: body });
            toast(t('cur.thresholds_saved'), 'success', 3000);
            await load();
          } catch (err) {
            toastError(err);
          }
        });
        const open = advBox.open;
        advBox.replaceChildren(el('summary', null, t('cur.advanced')), form);
        advBox.open = open;
      }

      function drawTabs() {
        const excluded = ((ctx.project || {}).counts || {}).excluded || 0;
        tabs.replaceChildren(
          ...['active', 'excluded'].map((v) =>
            el('button', {
              type: 'button', class: `tab${view === v ? ' active' : ''}`, 'aria-pressed': view === v ? 'true' : 'false', 'data-view': v,
              onclick: () => { view = v; draw(); grid.reload(); },
            }, v === 'active' ? t('cur.tab_active') : `${t('cur.tab_excluded')} (${num(excluded)})`)));
      }

      function drawToolbar() {
        const n = grid.selected.size;
        const buttons = [];
        if (view === 'active') {
          buttons.push(
            el('button', { type: 'button', class: 'btn', onclick: selectAll }, t('cur.select_all')),
            el('button', { type: 'button', class: 'btn', onclick: selectSuggested }, t('cur.select_suggested')),
            el('button', { type: 'button', class: 'btn', disabled: !n, onclick: () => grid.setSelected([]) }, t('cur.clear')),
            el('button', { type: 'button', class: 'btn primary', disabled: !n, onclick: exclude }, t('cur.exclude')));
        } else {
          buttons.push(
            el('button', { type: 'button', class: 'btn', onclick: selectAll }, t('cur.select_all')),
            el('button', { type: 'button', class: 'btn', disabled: !n, onclick: () => grid.setSelected([]) }, t('cur.clear')),
            el('button', { type: 'button', class: 'btn primary', disabled: !n, onclick: restore }, t('cur.restore')),
            el('button', { type: 'button', class: 'btn danger-outline', disabled: !n, onclick: remove }, t('cur.delete')));
        }
        toolbar.replaceChildren(...buttons, countText);
      }

      async function selectAll() {
        try { grid.setSelected(await grid.allIds()); } catch (err) { toastError(err); }
      }
      async function selectSuggested() {
        try {
          const data = await api('GET', `/projects/${projectId}/curation/suggested`);
          const ids = Array.isArray(data) ? data : (data && data.ids) || [];
          grid.setSelected(ids);
          toast(ids.length ? tn('cur.suggested_n', ids.length) : t('cur.suggested_none'), 'info', 4000);
        } catch (err) { toastError(err); }
      }
      const after = async () => { await afterChange(); await load(); };
      async function exclude() {
        const ids = [...grid.selected];
        try {
          const r = await api('POST', `/projects/${projectId}/images/exclude`, { json: { ids, reason: issue || 'manual' } });
          toast(tn('cur.excluded_n', (r && r.changed) ?? ids.length), 'success', 5000);
          await after();
        } catch (err) { toastError(err); }
      }
      async function restore() {
        const ids = [...grid.selected];
        try {
          const r = await api('POST', `/projects/${projectId}/images/restore`, { json: { ids } });
          toast(tn('cur.restored_n', (r && r.changed) ?? ids.length), 'success', 5000);
          await after();
        } catch (err) { toastError(err); }
      }
      async function remove() {
        const ids = [...grid.selected];
        const ok = await confirmDialog({
          title: t('cur.delete_title'), text: tn('cur.delete_text', ids.length), confirmLabel: t('cur.delete_confirm'),
        });
        if (!ok) return;
        try {
          const r = await api('POST', `/projects/${projectId}/images/delete`, { json: { ids, confirm: true } });
          toast(tn('cur.deleted_n', (r && r.deleted) ?? ids.length), 'success', 5000);
          await after();
        } catch (err) { toastError(err); }
      }

      async function openFiftyOne() {
        const view_ = issue ? `curation - ${issue}` : 'curation - quality issues';
        try {
          const job = await api('POST', `/projects/${projectId}/fiftyone`, { json: { view: view_ } });
          if (job && job.id) ctx.openWhenDone.add(job.id);
          toast(t('cur.fo_starting'), 'info', 4000);
          jobStarted(job);
        } catch (err) { toastError(err); }
      }
      async function pullExclusions() {
        try {
          const r = await api('POST', `/projects/${projectId}/fiftyone/pull-exclusions`);
          toast(tn('cur.pulled_n', (r && r.excluded) || 0), 'success', 5000);
          await after();
        } catch (err) { toastError(err); }
      }

      const foBox = el('div', { class: 'fo-box card' },
        el('div', null,
          el('button', { type: 'button', class: 'btn', onclick: openFiftyOne }, t('cur.open_fo')),
          el('button', { type: 'button', class: 'btn', onclick: pullExclusions }, t('cur.pull_fo'))),
        el('p', { class: 'muted' }, t('cur.fo_hint')));

      node.append(
        el('h2', { id: 'h-curation' }, t('step.curation')),
        noticeBox,
        histBox,
        el('div', { class: 'cur-main' },
          tabs, chipsBox, toolbar, grid.node),
        advBox,
        foBox);
      panel.mount = () => load();
      panel.refresh = () => load();
      return panel;
    }

    // ================================================================ jobs bar
    const kindText = (kind) => tr(`job.kind.${kind}`, kind);
    const statusText = (status) => tr(`job.status.${status}`, status);
    const isActive = (job) => job.status === 'queued' || job.status === 'running';

    function jobStep(job) {
      const kind = String(job.kind || '');
      if (/fiftyone/.test(kind)) return 'fiftyone';
      if (/analy|curat/.test(kind)) return 'curation';
      if (/annot/.test(kind)) return 'annotations';
      return 'images';
    }

    /* one job row (used by the bar and by the history dialog) */
    function jobRow(job) {
      const active = isActive(job);
      const step = jobStep(job);
      const url = job.result && job.result.url;
      return el('li', { class: `job job-${job.status}`, 'data-job': job.id },
        el('div', { class: 'job-top' },
          el('strong', null, kindText(job.kind)),
          el('span', { class: `status status-${job.status}` }, statusText(job.status)),
          el('span', { class: 'muted small' }, fmtTime(job.finished || job.started || job.created)),
          active ? el('button', { type: 'button', class: 'btn small', onclick: () => cancelJob(job) }, t('common.cancel')) : null,
          !active && job.status === 'done' && step === 'fiftyone' && url
            ? el('a', { class: 'btn small', href: url, target: '_blank', rel: 'noopener' }, t('jobs.open'))
            : null,
          !active && job.status === 'done' && step !== 'fiftyone'
            ? el('button', { type: 'button', class: 'btn small', onclick: () => { closeHistory(); showTab(step, true); } }, t('jobs.see_result'))
            : null),
        active ? el('div', { class: 'job-progress' }, progressBar(job.done, job.total, kindText(job.kind)),
          el('span', { class: 'muted' }, job.total ? `${num(job.done)} / ${num(job.total)}` : '')) : null,
        active && job.message ? el('div', { class: 'muted job-msg' }, tr(`jobmsg.${job.message}`, job.message)) : null,
        !active && job.status === 'done' && resultText(job) ? el('div', { class: 'muted job-msg' }, resultText(job)) : null,
        job.error ? el('div', { class: 'job-error' }, `${t('common.error')} ${job.error}`) : null);
    }

    /* The bar shows what is running and the latest finished task; the rest is in the history. */
    function renderJobs() {
      const all = [...ctx.jobs].sort((a, b) => String(b.created).localeCompare(String(a.created)));
      const running = all.filter(isActive);
      const finished = all.filter((j) => !isActive(j));
      const shown = [...running, ...finished.slice(0, 1)];
      jobsBox.hidden = !shown.length;
      document.getElementById('main').classList.toggle('no-jobs', !shown.length);
      if (!shown.length) { jobsBox.replaceChildren(); return; }
      const head = el('div', { class: 'jobs-head' }, el('h2', null, t('jobs.title')),
        finished.length > 1
          ? el('button', { type: 'button', class: 'btn small', onclick: openHistory }, t('jobs.history', { n: num(finished.length) }))
          : null);
      jobsBox.replaceChildren(head, el('ul', { class: 'jobs' }, ...shown.map(jobRow)));
      if (historyDlg && historyDlg.open) fillHistory();
    }

    let historyDlg = null;
    function fillHistory() {
      const finished = [...ctx.jobs].filter((j) => !isActive(j))
        .sort((a, b) => String(b.created).localeCompare(String(a.created)));
      const clear = el('button', { type: 'button', class: 'btn danger-outline', onclick: async () => {
        try {
          await api('DELETE', `/projects/${projectId}/jobs`);
          await pollJobs();
          closeHistory();
        } catch (err) { toastError(err); }
      } }, t('jobs.clear'));
      historyDlg.replaceChildren(el('div', { class: 'dialog-body' },
        el('h2', null, t('jobs.history_title')),
        finished.length ? el('ul', { class: 'jobs scroll-box' }, ...finished.map(jobRow)) : el('p', { class: 'muted' }, t('jobs.history_empty')),
        el('div', { class: 'form-actions' },
          finished.length ? clear : null,
          el('button', { type: 'button', class: 'btn', onclick: closeHistory }, t('common.close')))));
    }
    function openHistory() {
      if (!historyDlg) { historyDlg = el('dialog', { class: 'dialog wide', 'aria-label': t('jobs.history_title') }); document.body.appendChild(historyDlg); }
      fillHistory();
      historyDlg.showModal();
    }
    function closeHistory() { if (historyDlg && historyDlg.open) historyDlg.close(); }

    /* one line saying what a finished job changed */
    function resultText(job) {
      const r = job.result || {};
      if (job.kind === 'analyze' && 'measured' in r) {
        const flagged = Object.values(r.issues || {}).reduce((a, n) => a + (Number(n) || 0), 0);
        return t(r.measured ? 'jobs.result_analyze' : 'jobs.result_analyze_none', { measured: num(r.measured), flagged: num(flagged) });
      }
      if (job.kind === 'import_folder' && 'added' in r) return tn('images.added', r.added);
      return '';
    }

    async function cancelJob(job) {
      try {
        await api('POST', `/jobs/${job.id}/cancel`);
        await pollJobs();
      } catch (err) { toastError(err); }
    }

    async function pollJobs() {
      clearTimeout(ctx.pollTimer);
      ctx.pollTimer = null;
      let data;
      try {
        data = await api('GET', `/projects/${projectId}/jobs`);
      } catch (err) {
        if (err.status === 404) return;
        toastError(err);
        ctx.pollTimer = setTimeout(pollJobs, 5000);
        return;
      }
      const jobs = Array.isArray(data) ? data : (data && data.jobs) || [];
      ctx.jobs = jobs;
      let finished = false;
      for (const job of jobs) {
        const before = ctx.jobStatus.get(job.id);
        ctx.jobStatus.set(job.id, job.status);
        if (before && (before === 'queued' || before === 'running') && !isActive(job)) {
          finished = true;
          if (job.status === 'failed') toastError(new Error(job.error || statusText('failed')));
          if (job.status === 'done' && ctx.openWhenDone.has(job.id)) {
            ctx.openWhenDone.delete(job.id);
            const url = job.result && job.result.url;
            if (url && !window.open(url, '_blank', 'noopener')) toast(t('cur.fo_popup'), 'info', 8000);
          }
        }
      }
      renderJobs();
      if (finished) {
        await refreshProject();
        Object.values(ctx.panels).forEach((panel) => panel.mounted && panel.refresh && panel.refresh());
      }
      if (jobs.some(isActive)) ctx.pollTimer = setTimeout(pollJobs, 2000);
    }

    function jobStarted(job) {
      if (job && job.id) ctx.jobStatus.set(job.id, job.status || 'queued');
      pollJobs();
    }

    // ---- assemble
    ctx.panels = { images: imagesPanel(), annotations: annotationsPanel(), curation: curationPanel() };
    panelsBox.append(gateNode, ...STEPS.map((s) => ctx.panels[s].node));
    root.replaceChildren(headerBox, stepper, panelsBox, jobsBox);
    renderJobs();
    const startHash = location.hash.replace('#', '');  // read before anything rewrites the address
    ctx.tab = STEPS.includes(startHash) ? startHash : 'images';
    refreshProject().then((ok) => {
      if (!ok) return;
      showTab(ctx.tab);
      pollJobs();
    });
  }

  // ---------------------------------------------------------------- start
  document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    const page = document.body.dataset.page;
    if (page === 'projects') initProjects();
    else if (page === 'new') initNew();
    else if (page === 'project') initProject();
  });
})();
