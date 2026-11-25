/* Modal dialogs (choice, confirm), the menu button and the once() click guard. */
import { add, el, icon } from './dom.js';
import { t } from './i18n.js';

/** Closes every open dialog, except one that refuses to be cancelled (an upload in progress). */
export function closeOpenDialogs() {
  document.querySelectorAll('dialog[open]').forEach((dlg) => {
    if (dlg.dispatchEvent(new Event('cancel', { cancelable: true }))) dlg.close();
  });
}

/** Click handler for a button that starts something: the button stays disabled until the request settles,
 *  so a double click cannot start it twice. `stillBusy` says when it must stay disabled after that
 *  (a job of its kind is running). */
export const once = (fn, stillBusy = () => false) => async (e) => {
  const button = e.currentTarget;
  if (button.disabled) return;
  button.disabled = true;
  try {
    await fn(button);
  } finally {
    button.disabled = stillBusy();
  }
};

let closeOpenMenu = null;  // the menu that is open, closed when another one opens

/** An icon button that opens a small menu. `getItems()` is read each time it opens and gives
 *  `{label, reason?, disabled?, href?, onclick?}`: a disabled item stays in the list (and in the keyboard
 *  order) with its `reason` written under it. Enter, Space or ArrowDown open it on the first item; the arrows,
 *  Home and End move; Escape closes it and gives the focus back to the button; Tab or a click outside close it. */
export function menuButton(iconName, label, getItems, extraClass = '') {
  const trigger = el('button', { type: 'button', class: `icon-btn small${extraClass ? ` ${extraClass}` : ''}`, 'aria-label': label, title: label, 'aria-haspopup': 'menu', 'aria-expanded': 'false' }, icon(iconName, 16));
  const wrap = el('span', { class: 'menu-wrap' }, trigger);
  let menu = null;
  const entries = () => (menu ? [...menu.querySelectorAll('[role="menuitem"]')] : []);
  function onOutside(e) { if (menu && !wrap.contains(e.target)) close(false); }
  function close(refocus) {
    if (!menu) return;
    menu.remove();
    menu = null;
    trigger.setAttribute('aria-expanded', 'false');
    document.removeEventListener('pointerdown', onOutside, true);
    if (closeOpenMenu === close) closeOpenMenu = null;
    if (refocus) trigger.focus();
  }
  function onKey(e) {
    const all = entries();
    const at = all.indexOf(document.activeElement);
    const go = (i) => { e.preventDefault(); all[(i + all.length) % all.length].focus(); };
    if (e.key === 'Escape') { e.preventDefault(); close(true); }
    else if (e.key === 'Tab') close(false);
    else if (e.key === 'ArrowDown') go(at + 1);
    else if (e.key === 'ArrowUp') go(at < 0 ? -1 : at - 1);
    else if (e.key === 'Home') go(0);
    else if (e.key === 'End') go(-1);
    else if (e.key === ' ' && document.activeElement.tagName === 'A') { e.preventDefault(); document.activeElement.click(); }
  }
  function item(it) {
    const body = [el('span', { class: 'menu-label' }, it.label), it.reason ? el('span', { class: 'menu-reason' }, it.reason) : null];
    if (it.disabled) return el('span', { class: 'menu-item', role: 'menuitem', 'aria-disabled': 'true', tabindex: '-1' }, ...body);
    if (it.href) return el('a', { class: 'menu-item', role: 'menuitem', tabindex: '-1', href: it.href, target: '_blank', rel: 'noopener', onclick: () => close(false) }, ...body);
    return el('button', { type: 'button', class: 'menu-item', role: 'menuitem', tabindex: '-1', onclick: () => { close(true); it.onclick(); } }, ...body);
  }
  function open() {
    if (closeOpenMenu) closeOpenMenu(false);
    menu = el('div', { class: 'menu', role: 'menu', 'aria-label': label }, ...getItems().map(item));
    menu.addEventListener('keydown', onKey);
    wrap.appendChild(menu);
    trigger.setAttribute('aria-expanded', 'true');
    document.addEventListener('pointerdown', onOutside, true);
    closeOpenMenu = close;
    const box = menu.getBoundingClientRect();  // on a narrow screen the button may sit near the left edge: keep the menu in view
    const shift = Math.max(8 - box.left, 0) - Math.max(box.right - (window.innerWidth - 8), 0);
    if (shift) menu.style.transform = `translateX(${shift}px)`;
    const first = entries().find((x) => !x.hasAttribute('aria-disabled')) || entries()[0];
    if (first) first.focus();
  }
  trigger.addEventListener('click', () => (menu ? close(true) : open()));
  trigger.addEventListener('keydown', (e) => { if (e.key === 'ArrowDown' && !menu) { e.preventDefault(); open(); } });
  return wrap;
}

export function openModal(dlg) {
  const title = dlg.querySelector('h2');
  if (title && !dlg.querySelector('.dlg-head')) {
    const head = el('div', { class: 'dlg-head' });
    title.replaceWith(head);
    head.append(title, el('button', { type: 'button', class: 'icon-btn', 'aria-label': t('common.close'), title: t('common.close'), onclick: () => dlg.close() }, icon('close', 18)));
  }
  dlg.showModal();
}

/** A question with several answers; resolves with the chosen value, or null when cancelled. */
export function choiceDialog({ title, text, choices }) {
  return new Promise((resolve) => {
    const dlg = el('dialog', { class: 'dialog wide', 'aria-label': title });
    const cancelBtn = el('button', { type: 'button', class: 'btn', onclick: () => dlg.close() }, t('common.cancel'));
    let value = null;
    add(dlg, el('div', { class: 'dialog-body' },
      el('h2', null, title),
      el('p', null, text),
      el('ul', { class: 'choices' }, ...choices.map((c) => el('li', null,
        el('button', { type: 'button', class: `btn ${c.danger ? 'danger-outline' : c.primary ? 'primary' : ''}`, onclick: () => { value = c.value; dlg.close(); } }, c.label),
        c.help ? el('span', { class: 'muted' }, c.help) : null))),
      el('div', { class: 'form-actions' }, cancelBtn)));
    dlg.addEventListener('close', () => { dlg.remove(); resolve(value); });
    document.body.appendChild(dlg);
    openModal(dlg);
    cancelBtn.focus();  // Enter must never pick an answer (some are destructive)
  });
}

export function confirmDialog({ title, text, confirmLabel, danger = true, requireText = null }) {
  return new Promise((resolve) => {
    const input = requireText
      ? el('input', { type: 'text', 'aria-label': t('delete.type_name'), autocomplete: 'off' })
      : null;
    const okBtn = el('button', { type: 'submit', value: 'ok', class: `btn ${danger ? 'danger' : 'primary'}` }, confirmLabel);
    const mismatch = el('p', { class: 'hint', role: 'status', hidden: true }, t('delete.mismatch'));
    if (input) {
      okBtn.disabled = true;
      input.addEventListener('input', () => {
        okBtn.disabled = input.value !== requireText;
        mismatch.hidden = !input.value || !okBtn.disabled;  // say why the button is off
        if (mismatch.hidden) input.removeAttribute('aria-invalid'); else input.setAttribute('aria-invalid', 'true');
      });
    }
    const cancelBtn = el('button', { type: 'button', class: 'btn', onclick: () => dlg.close('cancel') }, t('common.cancel'));
    const dlg = el('dialog', { class: 'dialog', 'aria-label': title },
      el('form', { method: 'dialog', class: 'dialog-body' },
        el('h2', null, title),
        el('p', null, text),
        input,
        input ? mismatch : null,
        el('div', { class: 'form-actions' },
          okBtn,
          cancelBtn)));
    dlg.addEventListener('close', () => { dlg.remove(); resolve(dlg.returnValue === 'ok'); });
    document.body.appendChild(dlg);
    openModal(dlg);
    (input || (danger ? cancelBtn : okBtn)).focus();  // a destructive answer is never the one Enter gives
  });
}
