/* Small shared widgets: chips input, class chips, progress bar and empty state. */
import { openModal } from './dialogs.js';
import { add, el, fill, icon } from './dom.js';
import { num, t, tr } from './i18n.js';

/** Chips input: type a name and press Enter (or comma); click x to remove. */
export function chipsInput(id, initial) {
  const items = [...(initial || [])];
  const list = el('ul', { class: 'chips' });
  const input = el('input', { type: 'text', id, autocomplete: 'off', placeholder: t('classes.placeholder'), 'aria-describedby': `${id}-help` });
  const wrap = el('div', { class: 'chips-box' }, list, input);
  const note = el('p', { class: 'hint', role: 'status' });
  let hint = input.placeholder;   // the example text, shown only while there is no chip
  /* a name that is already in the list (any case) is not added twice: the note says which ones were left out (#637) */
  const add = (raw) => {
    const repeated = [];
    for (const part of String(raw).split(',')) {
      const name = part.trim();
      if (!name) continue;
      if (items.some((i) => i.toLowerCase() === name.toLowerCase())) {
        if (!repeated.some((r) => r.toLowerCase() === name.toLowerCase())) repeated.push(name);
      } else items.push(name);
    }
    note.textContent = repeated.length ? t('classes.repeated', { names: repeated.join(', ') }) : '';
    draw();
  };
  const draw = () => {
    input.placeholder = items.length ? '' : hint;
    fill(list,
      ...items.map((name, i) =>
        el(
          'li',
          { class: 'chip' },
          el('span', null, name),
          el('button', {
            type: 'button',
            class: 'chip-x',
            'aria-label': t('classes.remove', { name }),
            onclick: () => { items.splice(i, 1); note.textContent = ''; draw(); input.focus(); },
          }, icon('close', 12)),
        ),
      ),
    );
  };
  input.addEventListener('keydown', (e) => {
    if (e.isComposing) return;   // an IME is choosing characters: its Enter and comma are not ours (#637)
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      if (input.value.trim()) { add(input.value); input.value = ''; }
    } else if (e.key === 'Backspace' && !input.value && items.length) {
      items.pop();
      draw();
    }
  });
  input.addEventListener('blur', () => { if (input.value.trim()) { add(input.value); input.value = ''; } });
  /* a pasted list (one per line, or separated by commas, semicolons or tabs) becomes one chip per name (#586) */
  input.addEventListener('paste', (e) => {
    const text = (e.clipboardData && e.clipboardData.getData('text')) || '';
    if (!/[,;\t\r\n]/.test(text)) return;
    e.preventDefault();
    add(`${input.value},${text.replace(/[;\t\r\n]+/g, ',')}`);
    input.value = '';
  });
  wrap.addEventListener('click', () => input.focus());
  draw();
  const node = el('div', null, wrap, note, el('p', { id: `${id}-help`, class: 'sr-only' }, t('classes.help')));
  return {
    node,
    /** The example text of the empty field (it hides while there are chips). */
    setPlaceholder: (text) => { hint = text; draw(); },
    get: () => {
      if (input.value.trim()) { add(input.value); input.value = ''; }
      return [...items];
    },
  };
}

/* Class chips; long lists show the first ones and a button for the rest. */
const CLASSES_SHOWN = 12;

export function classList(classes) {
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
    add(box, ...rest, toggle);
    collapse(false);
  }
  return box;
}

/* The classes of the project on one line of the page header: the first ones that fit, and "see all" for the rest. */
/* a regression project calls its classes targets (#619) */
const classesTitle = (task) => t(task === 'regression' ? 'field.targets' : 'field.classes');

export function classLine(classes, task) {
  const title = classesTitle(task);
  const chip = (c) => el('span', { class: 'chip static' }, c);
  const open = (e) => {
    const list = el('ul', { class: 'chips' }, ...classes.map((c) => el('li', null, chip(c))));
    const search = el('input', { type: 'search', placeholder: t('ann.classes_search'), 'aria-label': t('ann.classes_search'),
      oninput: (ev) => {
        const q = ev.target.value.trim().toLowerCase();
        fill(list, ...classes.filter((c) => c.toLowerCase().includes(q)).map((c) => el('li', null, chip(c))));
      } });
    const dlg = el('dialog', { class: 'dialog', 'aria-label': `${title} (${num(classes.length)})` },
      el('div', { class: 'dialog-body' }, el('h2', null, `${title} (${num(classes.length)})`),
        classes.length > 12 ? el('div', { class: 'field' }, search) : null, list,
        el('div', { class: 'form-actions' }, el('button', { type: 'button', class: 'btn', onclick: () => dlg.close() }, t('common.close')))));
    dlg.addEventListener('close', () => { dlg.remove(); e.target.focus(); });
    document.body.appendChild(dlg);
    openModal(dlg);
  };
  const chips = el('span', { class: 'class-line-chips' }, ...classes.map(chip));
  const btn = el('button', { type: 'button', class: 'btn small', onclick: open, hidden: true }, t('classes.see_all'));
  /* the button and the fade appear only when a class does not fit on the line */
  const measure = () => {
    const over = chips.scrollWidth > chips.clientWidth + 1;
    btn.hidden = !over;
    chips.classList.toggle('overflowing', over);
  };
  new ResizeObserver(measure).observe(chips);
  return el('div', { class: 'class-line' },
    el('span', { class: 'muted class-line-label' }, `${title} (${num(classes.length)})`),
    chips, btn);
}

export function progressBar(done, total, label) {
  const pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;
  const bar = el('div', { class: 'progress', role: 'progressbar', 'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-valuenow': String(pct), 'aria-label': label || t('common.progress') },
    el('div', { class: 'progress-fill', style: `width:${pct}%` }));
  return bar;
}

/* An empty list says what is missing and puts the action in the same place; the header button is hidden then. */
export function emptyState(iconName, text, ...actions) {
  return el('div', { class: 'ds-empty' }, icon(iconName, 28), el('p', { class: 'muted' }, text), actions.length ? el('div', { class: 'ds-empty-actions' }, ...actions) : null);
}

/** One problem of an import or check report: its message, how many times, and up to five examples. */
export function problemRow(p) {
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
