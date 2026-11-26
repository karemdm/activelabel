/* The step bar of the project page (six steps in two groups, each with its state), the tabs inside a step,
 * and showing a step (or the card that says what it still needs). */
import { annotationStatus } from './names.js';
import { add, el, icon, redraw } from '../shell/dom.js';
import { num, t, tn } from '../shell/i18n.js';

/** Builds the step bar in `page.stepper` and returns what the page and its steps use of it. `page` gives
 *  ctx, projectId, STEPS and the stepper element (the page object itself is made after it). */
export function createStepper(page) {
  const { STEPS, ctx, projectId, stepper } = page;
  function stepStatus(step) {
    const c = (ctx.project && ctx.project.counts) || {};
    if (step === 'images') {
      if (!(c.active || c.excluded)) return t('status.images_none');
      return `${tn('status.images_active', c.active || 0)}${c.excluded ? ` · ${tn('status.excluded', c.excluded)}` : ''}`;
    }
    if (step === 'annotations') {
      const waiting = c.preannotated ? ` · ${tn('status.to_review', c.preannotated)}` : '';
      const none = c.empty ? ` · ${tn('status.empty', c.empty)}` : '';
      return c.annotated > 0 || c.empty > 0
        ? `${annotationStatus(c)}${c.annotated > 0 ? none : ''}${waiting} · ${tn('status.classes', (ctx.project.classes || []).length)}`
        : t('status.annotations_none');
    }
    if (step === 'training') {
      if (!ctx.runCount) return t('status.runs_none');
      return `${tn('status.runs', ctx.runCount)}${ctx.runActive ? ` · ${tn('status.runs_active', ctx.runActive)}` : ''}`;
    }
    if (step === 'datasets') return ctx.datasetCount ? tn('status.versions', ctx.datasetCount) : t('status.versions_none');
    if (step === 'models') {
      if (!ctx.modelCount) return t('status.models_none');
      return `${tn('status.models', ctx.modelCount)}${ctx.currentModel ? ` · ${t('status.current_model', { n: ctx.currentModel })}` : ''}`;
    }
    return c.analyzed > 0 ? tn('status.issues', c.with_issues || 0) : t('status.not_analyzed');
  }

  /* where a step stands, for the marker on the bar: done, needs attention, busy or still to do */
  function stepState(step) {
    const c = (ctx.project && ctx.project.counts) || {};
    if (step === 'images') return c.active > 0 ? 'done' : 'todo';
    if (step === 'annotations') return c.preannotated > 0 ? 'attention' : (c.annotated > 0 || c.empty > 0) ? 'done' : 'todo';
    if (step === 'curation') return c.analyzed > 0 ? (c.with_issues > 0 ? 'attention' : 'done') : 'todo';
    if (step === 'datasets') return ctx.datasetCount ? 'done' : 'todo';
    if (step === 'training') return ctx.runActive ? 'busy' : ctx.runCount ? 'done' : 'todo';
    return ctx.modelCount ? 'done' : 'todo';
  }

  /* the step bar groups the steps: the data (1-3) and the model made from it (4-6) */
  const STEP_GROUPS = [['data', ['images', 'annotations', 'curation']], ['model', ['datasets', 'training', 'models']]];
  function renderStepper() {
    const item = (step) => {
      const i = STEPS.indexOf(step);
      const state = stepState(step);
      return el('button', {
        type: 'button',
        class: `step is-${state}${ctx.tab === step ? ' current' : ''}`,
        'aria-current': ctx.tab === step ? 'step' : null,
        'data-step': step,
        'data-act': step,
        onclick: () => showTab(step),
      },
        el('span', { class: 'step-num', 'aria-hidden': 'true' }, state === 'done' ? icon('check', 16) : state === 'attention' ? '!' : String(i + 1)),
        el('span', { class: 'step-text' },
          el('span', { class: 'step-title' }, t(`step.${step}`), state === 'todo' ? null : el('span', { class: 'sr-only' }, ` (${t(`step.state_${state}`)})`), step === 'annotations' ? el('span', { class: 'optional step-tag' }, t('step.optional')) : null),
          el('span', { class: 'step-status', title: stepStatus(step) }, stepStatus(step))));
    };
    redraw(stepper, ...STEP_GROUPS.map(([group, steps]) => el('div', { class: 'step-group', role: 'group', 'aria-labelledby': `sg-${group}`, 'data-group': group },
      el('p', { class: 'step-group-title', id: `sg-${group}` }, t(`project.group_${group}`)),
      ...steps.map(item))));
    /* on a narrow screen the bar scrolls sideways: keep the current step in view */
    requestAnimationFrame(() => {  // after the bar is in the page and has its width
      const here = stepper.querySelector('.step.current');
      if (here && stepper.scrollWidth > stepper.clientWidth) {
        const box = stepper.getBoundingClientRect();
        const at = here.getBoundingClientRect();
        stepper.scrollLeft += at.left - box.left - (box.width - at.width) / 2;
      }
    });
  }

  /* Tabs inside a step: one subject per tab. The last tab is remembered per project and step. */
  function tabsView(key, label, tabs, initial) {
    const store = `al.tab.${projectId}.${key}`;
    let saved = null;
    try { saved = localStorage.getItem(store); } catch (err) { /* not remembered */ }
    let current = null;
    const parts = new Map();
    const list = el('div', { class: 'step-tabs', role: 'tablist', 'aria-label': label });
    function select(id, focus = false) {
      if (!parts.has(id)) return;
      current = id;
      for (const [tid, part] of parts) {
        const on = tid === id;
        part.button.setAttribute('aria-selected', String(on));
        part.button.tabIndex = on ? 0 : -1;
        part.pane.hidden = !on;
      }
      try { localStorage.setItem(store, id); } catch (err) { /* not remembered */ }
      if (focus) parts.get(id).button.focus();
      const tab = tabs.find((x) => x.id === id);
      if (tab && tab.onShow) tab.onShow();
    }
    function keys(e) {
      const ids = [...parts.keys()];
      const at = ids.indexOf(current);
      const next = { ArrowRight: at + 1, ArrowLeft: at - 1, Home: 0, End: ids.length - 1 }[e.key];
      if (next == null) return;
      e.preventDefault();
      select(ids[(next + ids.length) % ids.length], true);
    }
    const panes = tabs.map((tab) => {
      const id = `tab-${key}-${tab.id}`;
      const count = el('span', { class: 'step-tab-count', hidden: true });
      const button = el('button', { type: 'button', role: 'tab', id, class: 'step-tab', 'aria-controls': `${id}-pane`, onclick: () => select(tab.id), onkeydown: keys }, tab.label, count);
      const pane = el('div', { role: 'tabpanel', id: `${id}-pane`, class: 'step-tab-pane', 'aria-labelledby': id }, ...tab.nodes);
      parts.set(tab.id, { button, pane, count });
      add(list, button);
      return pane;
    });
    select(parts.has(saved) ? saved : (typeof initial === 'function' ? initial() : initial) || tabs[0].id);
    return {
      node: el('div', { class: 'tabs-view' }, list, ...panes),
      select,
      current: () => current,
      /** true when the tab came from the person's last visit */
      remembered: parts.has(saved),
      /** a number next to a tab's name (hidden for null) */
      setCount: (id, n) => { const part = parts.get(id); if (!part) return; part.count.hidden = n == null; part.count.textContent = n == null ? '' : num(n); },
    };
  }

  /* steps 2 and 3 work on the project's images: without images they only say so */
  const hasImages = () => {
    const c = (ctx.project && ctx.project.counts) || {};
    return (c.active || 0) + (c.excluded || 0) > 0;  // excluded images can be restored in step 3
  };
  const gateNode = el('section', { class: 'card gate', hidden: true },
    el('p', { class: 'report-line' }, icon('info'), t('gate.need_images')),
    el('button', { type: 'button', class: 'btn primary', onclick: () => showTab('images') }, t('gate.go_images')));

  /* step 5 needs a dataset version: without one it sends the person to step 4 */
  const gateDatasetNode = el('section', { class: 'card gate', hidden: true },
    el('p', { class: 'report-line' }, icon('info'), t('gate.need_dataset')),
    el('button', { type: 'button', class: 'btn primary', onclick: () => showTab('datasets') }, t('gate.go_datasets')));

  function showTab(step, scrollToPanel = false) {
    ctx.tab = step;
    /* a step the person opens is a history entry (Back goes to the previous step); the first one and redraws are not */
    if (location.hash !== `#${step}`) history[ctx.shown ? 'pushState' : 'replaceState'](null, '', `#${step}`);
    ctx.shown = true;
    const noImages = step !== 'images' && step !== 'models' && !hasImages();
    const noDataset = step === 'training' && !noImages && !ctx.datasetCount;
    const gated = noImages || noDataset;
    for (const s of STEPS) ctx.panels[s].node.hidden = gated || s !== step;
    gateNode.hidden = !noImages;
    gateDatasetNode.hidden = !noDataset;
    renderStepper();
    if (gated) return;
    const panel = ctx.panels[step];
    if (!panel.mounted) { panel.mounted = true; panel.mount(); } else if (panel.show) panel.show();
    if (scrollToPanel) panel.node.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  return { gateDatasetNode, gateNode, hasImages, renderStepper, showTab, tabsView };
}
