/* Step 4, Data split: the frozen dataset versions (train / validation / test or k folds), making a new one
 * with its checks, and what changed in the labels since a version was made. */
import { partBalance } from './distributions.js';
import { partName, taskBadge } from './names.js';
import { api, errorParams } from '../shell/api.js';
import { confirmDialog, once, openModal, returnFocus } from '../shell/dialogs.js';
import { add, el, fill, icon } from '../shell/dom.js';
import { fmtDateTime, lower, num, t, tn, tr } from '../shell/i18n.js';
import { toast, toastError } from '../shell/toast.js';
import { emptyState } from '../shell/widgets.js';

/* images whose labels changed after the version was made (0 when none); the version keeps the old ones */
/* images whose labels changed, or that were excluded or put back, since the version was made (#636) */
/** Whole percentages of some counts that add up to 100 (largest remainder); equal counts always show the same
    percentage (rounded as usual when the leftover points cannot go to all of them, so ties can leave the sum a
    little off); all zeros when the total is 0. */
export function sharesOf(counts) {
  const total = counts.reduce((a, n) => a + n, 0);
  if (!(total > 0)) return counts.map(() => 0);
  const exact = counts.map((n) => (n / total) * 100);
  const out = exact.map(Math.floor);
  let left = 100 - out.reduce((a, n) => a + n, 0);
  /* the remainders, largest first; equal counts form one group that gets the extra point together or not at all */
  const groups = [...new Set(counts)].map((n) => ({ n, rest: (n / total) * 100 - Math.floor((n / total) * 100), at: counts.map((c, i) => (c === n ? i : -1)).filter((i) => i >= 0) }))
    .sort((a, b) => b.rest - a.rest || b.n - a.n);
  for (const g of groups) {
    const up = g.rest > 0 && (g.at.length <= left || g.rest >= 0.5);   // a group too big for what is left rounds as usual
    if (up) { g.at.forEach((i) => { out[i] += 1; }); left -= g.at.length; }
  }
  return out;
}
export const staleCount = (v) => { const s = v.stale || {}; return ['changed_images', 'added', 'removed', 'excluded', 'restored'].reduce((n, k) => n + (s[k] || 0), 0); };

export function datasetsPanel(page) {
  const { ctx, projectId, renderStepper, showTab } = page;
  const node = el('section', { class: 'panel', id: 'panel-datasets', 'aria-labelledby': 'h-datasets' });
  const panel = { node, mounted: false };
  const listBox = el('div', { class: 'ds-list' });
  let versions = [];
  const isClassification = () => ctx.project && ctx.project.task === 'classification';

  /* sizes: {train, val, test} or {test, folds: [n, ...]}; the preview may send counts or id lists */
  const count = (x) => (Array.isArray(x) ? x.length : Number(x) || 0);
  function normSizes(d) {
    if (d.sizes) return d.sizes;
    const src = d.parts || d;
    const out = {};
    ['train', 'val', 'test'].forEach((k) => { if (src[k] != null) out[k] = count(src[k]); });
    if (Array.isArray(d.folds)) out.folds = d.folds.map(count);
    return out;
  }
  /* one entry per part to draw: {key, label, n, cls, part, fold} */
  /* parts to show: the ones with images, and the ones that were asked for even if they ended up empty */
  function partItems(sizes, options) {
    const asked = ((options || {}).split) || {};
    const wanted = (p) => count(sizes[p]) > 0 || p === 'train' || Number(asked[p]) > 0;
    const items = [];
    if (Array.isArray(sizes.folds)) {
      sizes.folds.forEach((n, i) => items.push({ key: `fold${i}`, label: t('ds.fold', { n: i + 1 }), n: count(n), cls: `fold-${i % 2}`, fold: i }));
    } else {
      ['train', 'val'].forEach((p) => { if (sizes[p] != null && wanted(p)) items.push({ key: p, label: partName(p), n: count(sizes[p]), cls: p, part: p }); });
    }
    if (sizes.test != null && wanted('test')) items.push({ key: 'test', label: partName('test'), n: count(sizes.test), cls: 'test', part: 'test' });
    return items;
  }
  const totalOf = (items) => items.reduce((a, x) => a + x.n, 0);
  const pctOf = (n, total) => (total > 0 ? Math.round((n / total) * 100) : 0);
  /* the shares of the parts, rounded so that they add up to 100 (largest remainder, #636) */
  const pctsOf = (items) => sharesOf(items.map((x) => x.n));

  function partsBar(items, label) {
    return el('div', { class: 'ds-bar', role: 'img', 'aria-label': label },
      ...items.filter((x) => x.n > 0).map((x) => el('span', { class: `ds-seg ds-${x.cls}`, style: `flex-grow:${x.n}`, title: `${x.label}: ${num(x.n)}` })));
  }
  const legend = (items) => el('ul', { class: 'ds-legend' }, ...items.map((x) => el('li', null,
    el('span', { class: `ds-dot ds-${x.cls}`, 'aria-hidden': 'true' }), `${x.label}: `, el('strong', null, num(x.n)),
    el('span', { class: 'muted' }, ` · ${pctsOf(items)[items.indexOf(x)]}%`))));
  const barLabel = (items) => items.map((x) => `${x.label} ${num(x.n)}`).join(', ');

  /* background images of a stats object {part: {images, background, objects}} */
  const statsList = (stats) => (stats ? (Array.isArray(stats.folds) ? [...stats.folds, stats.test] : Object.values(stats)).filter((s) => s && typeof s === 'object') : []);
  const backgroundOf = (stats) => statsList(stats).reduce((a, s) => a + (Number(s.background) || 0), 0);
  function statFor(stats, item) {
    if (!stats) return {};
    if (item.fold != null) return (Array.isArray(stats.folds) ? stats.folds[item.fold] : stats[`fold${item.fold + 1}`] || stats[`fold_${item.fold + 1}`] || stats[`fold${item.fold}`]) || {};
    return stats[item.key] || {};
  }

  /* one warning per fact: an empty part says so once (not "class missing" for every class), and the
     proportion drift is said once */
  function cleanWarnings(warnings, items) {
    const empty = new Set(items.filter((x) => x.n === 0).map((x) => (x.fold != null ? `fold${x.fold}` : x.part)));
    const seen = new Set();
    const out = [];
    for (const w of warnings || []) {
      if (w.code === 'video_split') continue;
      const where = w.fold != null ? `fold${w.fold}` : w.part;
      let x = w;
      if (empty.has(where) && /class_missing|few_images/.test(String(w.code))) x = { code: 'part_empty', part: w.part, fold: w.fold };
      else if (/drift|group/.test(String(w.code))) x = { code: 'drift' };
      const key = JSON.stringify([x.code, x.part, x.fold, x.class]);
      if (!seen.has(key)) { seen.add(key); out.push(x); }
    }
    /* several classes missing from the same part: one warning listing them */
    const grouped = [];
    const byPart = new Map();
    for (const w of out) {
      if (w.code !== 'class_missing') { grouped.push(w); continue; }
      const where = w.fold != null ? `fold${w.fold}` : w.part;
      if (!byPart.has(where)) {
        byPart.set(where, { code: 'classes_missing', part: w.part, fold: w.fold, classes: [], available: [], avoidable: [] });
        grouped.push(byPart.get(where));
      }
      const group = byPart.get(where);
      group.classes.push(w.class);
      group.available.push(w.available);
      group.avoidable.push(w.avoidable);
    }
    const fewFolds = grouped.filter((w) => w.code === 'few_images' && w.fold != null);
    if (fewFolds.length > 1) {
      const at = grouped.indexOf(fewFolds[0]);
      const rest = grouped.filter((w) => !fewFolds.includes(w));
      rest.splice(at, 0, { code: 'few_images_folds', folds: fewFolds.map((w) => w.fold) });
      grouped.splice(0, grouped.length, ...rest);
    }
    return grouped.map((w) => (w.code === 'classes_missing' && w.classes.length === 1
      ? { code: 'class_missing', part: w.part, fold: w.fold, class: w.classes[0], available: w.available[0], avoidable: w.avoidable[0] } : w));
  }

  /* why a hold-out version cannot train a model (an empty training or validation part), or null */
  function trainBlockReason(sizes) {
    if (Array.isArray(sizes.folds)) return null;
    if (!count(sizes.train)) return t('ds.cannot_train_train');
    if (!count(sizes.val)) return t('ds.cannot_train_val');
    return null;
  }

  /* information about the videos that were cut into segments (not a warning) */
  const videoInfos = (warnings) => (warnings || []).filter((w) => w.code === 'video_split');
  /* parts of one video sit in training and in validation/test: the scores may be optimistic (shown prominently) */
  const sharedVideos = (warnings) => (warnings || []).find((w) => w.code === 'video_parts_shared') || null;
  const sharedAlert = (w, tag = 'div') => (w ? el(tag, { class: 'ds-shared', role: 'alert' }, icon('warning', 16), el('strong', null, t('ds.w_video_parts_shared'))) : null);
  /* the gap actually used for a video, from the version's or the preview's video list */
  const gapSecondsOf = (w, videos) => { const v = (videos || []).find((x) => x.video === w.video); return v && Number(v.gap_seconds) > 0 ? Number(v.gap_seconds) : 0; };
  function videoText(w, videos = null) {
    const parts = (w.parts || []).map((p) => (p === 'fold' ? t('ds.folds_word') : partName(p))).join(t('ds.and'));
    const vars = { name: w.name || w.video, n: num(w.segments), parts, gap: num(w.gap || 0) };
    const secs = gapSecondsOf(w, videos);
    const gapNote = secs ? ` ${t('ds.video_gap_s', { s: num(secs) })}` : '';
    return t(w.gap > 0 ? 'ds.w_video_split' : 'ds.w_video_split_nogap', vars) + gapNote;
  }

  /* whether a missing class could have been in every part (the API says; older versions do not) */
  function avoidText(w) {
    if (w.code === 'class_missing') {
      if (w.avoidable == null) return '';
      return w.avoidable ? tn('ds.w_avoidable', w.available || 0) : tn('ds.w_unavoidable', w.available || 0);
    }
    const flags = (w.avoidable || []).filter((x) => x != null);
    if (!flags.length) return '';
    const fixed = flags.filter((x) => x === false).length;
    if (fixed === 0) return t('ds.w_avoidable_all');
    return fixed === flags.length ? t('ds.w_unavoidable_all') : tn('ds.w_unavoidable_some', fixed, { total: num(flags.length) });
  }

  /* warnings in plain words; short = for the badges on a card */
  function warnText(w, items, short = false) {
    const part = w.fold != null ? t('ds.fold', { n: w.fold + 1 }) : partName(w.part);
    const code = String(w.code || '');
    if (code === 'class_missing') return short ? t('ds.b_class_missing', { class: w.class, part }) : `${t('ds.w_class_missing', { class: w.class, part })} ${avoidText(w)}`.trim();
    if (code === 'classes_missing') {
      const list = w.classes.slice(0, 6).join(', ') + (w.classes.length > 6 ? ` (+${w.classes.length - 6})` : '');
      return short ? tn('ds.b_classes_missing', w.classes.length, { part }) : `${tn('ds.w_classes_missing', w.classes.length, { part, list })} ${avoidText(w)}`.trim();
    }
    if (code === 'part_empty') return short ? t('ds.b_part_empty', { part }) : t('ds.w_part_empty', { part });
    if (code === 'few_images') return short ? t('ds.b_few_images', { part }) : t('ds.w_few_images', { part });
    if (code === 'few_images_folds') {   // the same fact for several folds: said once (#636)
      const folds = w.folds.map((f) => num(f + 1)).join(', ');
      return short ? t('ds.b_few_images_folds', { n: num(w.folds.length) }) : t('ds.w_few_images_folds', { folds });
    }
    if (code === 'classes_unused') {
      const list = (w.classes || []).slice(0, 6).join(', ') + ((w.classes || []).length > 6 ? ` (+${w.classes.length - 6})` : '');
      return short ? tn('ds.b_classes_unused', (w.classes || []).length) : tn('ds.w_classes_unused', (w.classes || []).length, { list });
    }
    if (code === 'target_constant') {   // regression (#619): every value of a target is the same
      const vars = { target: w.target, value: num(w.value) };
      return short ? t('ds.b_target_constant', vars) : t('ds.w_target_constant', vars);
    }
    if (code === 'labels_without_geometry' || code === 'multiple_classes' || code === 'keypoints_count_mismatch') {
      const vars = { n: num(w.count || 0), expected: num(w.expected || 0) };
      return short ? tn(`ds.b_${code}`, w.count || 0, vars) : tn(`ds.w_${code}`, w.count || 0, vars);
    }
    if (/drift|group/.test(code)) {
      const total = totalOf(items);
      const real = total > 0 ? pctsOf(items).join('/') : items.map(() => 0).join('/');
      const why = w.cause === 'video' ? '' : w.cause === 'duplicates' ? '_duplicates' : '_generic';  // only frames of a video talk about videos
      return short ? t('ds.b_group_drift', { real }) : t(`ds.w_group_drift${why}`, { real });
    }
    if (/dup|curation/.test(code)) return short ? t('ds.b_duplicates') : t('ds.w_duplicates');
    return tr(`ds.w_${code}`, code);
  }

  // ---------------------------------------------------------------- list
  function splitText(v) {
    const o = v.options || {};
    if (o.kfold) return t(o.split && Number(o.split.test) > 0 ? 'ds.kfold_short' : 'ds.kfold_only', { k: o.kfold });
    const s = o.split || {};
    /* what was asked for; the real sizes are in the bar and the legend below (groups can make them differ) */
    return t('ds.split_asked', { split: [s.train, s.val, s.test].filter((x) => x != null).map((x) => `${x}%`).join(' / ') });
  }

  /* warnings go into a few groups: one line says how many, the list opens on demand */
  const WARN_GROUP = {
    class_missing: 'classes', classes_missing: 'classes', part_empty: 'size', few_images: 'size',
    labels_without_geometry: 'labels', multiple_classes: 'labels', keypoints_count_mismatch: 'labels',
    classes_unused: 'classes', few_images_folds: 'size', no_curation: 'curation', target_constant: 'labels',
  };
  const warnGroup = (w) => WARN_GROUP[w.code] || (/drift|group/.test(String(w.code)) ? 'balance' : 'other');
  function warnSummary(list) {
    const count = new Map();
    list.forEach((w) => count.set(warnGroup(w), (count.get(warnGroup(w)) || 0) + 1));
    const groups = [...count].map(([g, n]) => t(`ds.warn_g_${g}`, { n: num(n) })).join(' · ');
    return `${tn('ds.warn_count', list.length)}: ${groups}`;
  }

  function versionCard(v) {
    const sizes = normSizes(v);
    const items = partItems(sizes, v.options);
    const bg = backgroundOf(v.stats);
    const cleaned = cleanWarnings(v.warnings, items).filter((w) => w.code !== 'video_parts_shared');
    const warns = cleaned.map((w) => el('li', { class: 'ds-badge warn', title: warnText(w, items) }, icon('warning', 14), warnText(w, items, true)));
    const infoBadges = videoInfos(v.warnings).map((w) => el('li', { class: 'ds-badge', title: videoText(w, v.videos) }, icon('video', 14), t('ds.b_video_split', { n: num(w.segments) })));
    const id = `ds-v${v.version}`;
    const noTrain = trainBlockReason(sizes);
    const stale = staleCount(v);
    return el('li', { class: 'ds-card card', id, 'aria-labelledby': `${id}-title` },
      el('div', { class: 'ds-card-head' },
        el('h3', { id: `${id}-title` }, `v${v.version}`),
        el('span', { class: 'muted' }, fmtDateTime(v.created)),
        v.type ? taskBadge(v.type) : null,
        v.incomplete ? el('span', { class: 'ds-badge bad', title: t('ds.incomplete_help') }, icon('error', 14), t('ds.incomplete')) : null,
        el('button', { type: 'button', class: 'icon-btn danger ds-del', 'aria-label': t('ds.delete_aria', { n: v.version }), title: t('ds.delete'), onclick: () => remove(v) }, icon('trash', 18))),
      v.note ? el('p', { class: 'ds-note' }, v.note) : null,
      el('p', { class: 'muted ds-meta' }, splitText(v), bg > 0 ? ` · ${t('ds.background_count', { n: num(bg) })}` : ''),
      partsBar(items, barLabel(items)),
      legend(items),
      warns.length ? el('details', { class: 'ds-warn-group' },
        el('summary', null, icon('warning', 14), warnSummary(cleaned)),
        el('ul', { class: 'ds-badges', 'aria-label': t('ds.warnings') }, ...warns)) : null,
      sharedAlert(sharedVideos(v.warnings)),
      infoBadges.length ? el('ul', { class: 'ds-badges' }, ...infoBadges) : null,
      noTrain ? el('p', { class: 'ds-block', id: `${id}-block` }, icon('warning', 14), noTrain) : null,
      stale ? el('div', { class: 'ds-stale', role: 'status' },
        el('p', null, icon('warning', 14), el('strong', null, tn('ds.stale', stale, { n: num(stale) }))),
        el('button', { type: 'button', class: 'btn small ds-new-from', onclick: once(() => openNew(null, v)) }, icon('plus', 14), t('ds.new_from'))) : null,
      el('div', { class: 'ds-view' },
        el('span', { class: 'muted' }, t('ds.see_images')),
        ...items.filter((x) => x.n > 0).map((x) => el('button', { type: 'button', class: 'btn small', 'aria-label': t('ds.see_part', { n: v.version, part: x.label }),
          onclick: () => { ctx.panels.images.showDataset(v.version, x.part || null, x.fold != null ? x.fold : null); showTab('images', true); } }, x.label)),
        el('button', { type: 'button', class: 'btn small primary ds-train-with', disabled: !!v.incomplete || !!noTrain, 'aria-describedby': noTrain ? `${id}-block` : null,
          onclick: () => { showTab('training'); ctx.panels.training.startWith(v.version); } }, icon('chart', 14), t('ds.train_with'))));
  }

  function draw() {
    newBtn.hidden = !versions.length;
    fill(listBox, versions.length
      ? el('ul', { class: 'ds-cards' }, ...versions.map(versionCard))
      : emptyState('images', t('ds.empty'),
        el('button', { type: 'button', class: 'btn primary', onclick: once(() => openNew(newBtn)) }, icon('plus', 16), t('ds.new'))));
  }

  async function load() {
    try {
      const d = await api('GET', `/projects/${projectId}/datasets`);
      versions = (d.versions || []).slice().sort((a, b) => b.version - a.version);
    } catch (err) {
      toastError(err);
      return;
    }
    ctx.datasetCount = versions.length;
    draw();
    renderStepper();
  }

  async function remove(v) {
    const ok = await confirmDialog({ title: t('ds.delete_title', { n: v.version }), text: t('ds.delete_text'), confirmLabel: t('ds.delete') });
    if (!ok) return;
    const send = (extra) => api('DELETE', `/projects/${projectId}/datasets/${v.version}`, { json: { confirm: true, ...extra } });
    try {
      try {
        await send({});
      } catch (err) {
        /* finished runs used this version: say which, and delete only when the person accepts they cannot be repeated */
        if (err.code !== 'dataset_used_by_runs') throw err;
        const again = await confirmDialog({
          title: t('ds.delete_used_title', { n: v.version }),
          text: t('ds.delete_used_text', { runs: errorParams(err.params).runs }),
          confirmLabel: t('ds.delete_used_confirm'),
        });
        if (!again) return;
        await send({ confirm_runs: true });
      }
      toast(t('ds.deleted', { n: v.version }), 'success', 3000);
      ctx.panels.images.forgetDataset(v.version);
    } catch (err) {
      toastError(err);
    }
    await load();
  }

  // ---------------------------------------------------------------- new version form
  /* `from`: a version whose options fill the form ("new version from this one") */
  async function openNew(trigger, from = null) {
    const opener = document.activeElement;   // the empty state's own button when the header one is hidden
    let tasks;
    try {
      tasks = ((await api('GET', `/projects/${projectId}/tasks`)).tasks || []).filter((x) => x.active > 0);
    } catch (err) {
      toastError(err);
      return;
    }
    const cls = isClassification();
    const st = { tasks: new Set(tasks.map((x) => x.id)), mode: 'split', bg: false, maxK: null, blocked: false };
    let timer = null;
    let token = 0;

    const numInput = (id, value, min, max) => el('input', { type: 'number', id, inputmode: 'numeric', min: String(min), max: String(max), step: '1', value: String(value) });
    const train = numInput('ds-train', 70, 1, 100);
    const val = numInput('ds-val', 20, 0, 99);
    const test = numInput('ds-test', 10, 0, 99);
    const kInput = numInput('ds-k', 5, 2, 10);
    const kHelp = el('p', { class: 'muted ds-help', id: 'ds-k-help' });
    kInput.setAttribute('aria-describedby', 'ds-k-help');
    const allowBox = el('input', { type: 'checkbox', id: 'ds-allow-empty', onchange: () => syncCreate() });
    const ktestInput = numInput('ds-ktest', 10, 0, 90);
    const seed = el('input', { type: 'number', id: 'ds-seed', inputmode: 'numeric', step: '1', value: '42' });
    const note = el('input', { type: 'text', id: 'ds-note', maxlength: '500', autocomplete: 'off' });
    const bgBox = el('input', { type: 'checkbox', id: 'ds-bg', role: 'switch' });
    const videosSel = el('select', { id: 'ds-videos' }, el('option', { value: 'auto' }, t('ds.videos_auto')), el('option', { value: 'whole' }, t('ds.videos_whole')));
    const gapInput = el('input', { type: 'text', id: 'ds-gap', inputmode: 'decimal', autocomplete: 'off', value: num(0.5) });   // text: "0,5" in pt
    /* the safety gap is a choice: on by default, off sends 0 seconds (frames of one video may then sit on both sides) */
    const gapBox = el('input', { type: 'checkbox', id: 'ds-gap-on', role: 'switch', checked: true });
    const gapField = el('div', { class: 'ds-gap-field' }, el('div', { class: 'field ds-num' }, el('label', { for: 'ds-gap' }, t('ds.gap')), gapInput),
      el('p', { class: 'muted ds-help' }, t('ds.gap_help')));
    const gapOff = el('p', { class: 'ds-gap-off', role: 'status', hidden: true }, icon('warning', 14), t('ds.gap_off_warn'));
    const syncGap = () => { gapField.hidden = !gapBox.checked; gapOff.hidden = gapBox.checked; };
    gapBox.addEventListener('change', () => { syncGap(); changed(); });
    const create = el('button', { type: 'submit', class: 'btn primary' }, t('ds.create'));
    if (from) {
      const o = from.options || {};
      const asked = new Set(o.tasks || []);
      if (asked.size) st.tasks = new Set(tasks.filter((x) => asked.has(x.id)).map((x) => x.id));
      st.bg = !!o.background && !cls;
      bgBox.checked = st.bg;
      seed.value = String(o.seed != null ? o.seed : 42);
      videosSel.value = o.videos || 'auto';
      const gapSeconds = o.gap_seconds != null ? Number(o.gap_seconds) : 0.5;
      gapBox.checked = gapSeconds > 0;
      gapInput.value = num(gapSeconds > 0 ? gapSeconds : 0.5);
      const sp = o.split || {};
      if (o.kfold) {
        st.mode = 'kfold';
        kInput.value = String(o.kfold);
        ktestInput.value = String(sp.test != null ? sp.test : 10);
      } else {
        [train.value, val.value, test.value] = [sp.train, sp.val, sp.test].map((x, i) => String(x != null ? x : [70, 20, 10][i]));
      }
    }
    /* a version whose training or validation part is empty is created only on an explicit "create anyway" */
    const syncCreate = () => { create.disabled = st.blocked && !allowBox.checked; };

    // tasks
    const taskBoxes = tasks.map((x) => {
      const box = el('input', { type: 'checkbox', id: `ds-task-${x.id}`, checked: st.tasks.has(x.id), onchange: (e) => { if (e.target.checked) st.tasks.add(x.id); else st.tasks.delete(x.id); changed(); } });
      return el('li', null, el('label', { for: box.id, class: 'ds-check' }, box, el('span', { class: 'ds-check-name' }, x.name), el('span', { class: 'muted' }, tn('tasks.meta_images', x.active))));
    });

    // ratios
    const ratioBar = el('div', { class: 'ds-bar', 'aria-hidden': 'true' });
    const sumText = el('p', { class: 'ds-sum', role: 'status' });
    const modeBox = el('div', { class: 'ds-ratios' });
    const raw = (inp) => parseInt(inp.value, 10);   // as typed: a negative or a too big value stays visible
    const int = (inp) => { const n = raw(inp); return Number.isFinite(n) ? Math.max(0, Math.min(100, n)) : NaN; };
    const validPct = (inp) => Number.isFinite(raw(inp)) && raw(inp) >= 0 && raw(inp) <= 100;
    const field = (label, input) => el('div', { class: 'field ds-num' }, el('label', { for: input.id }, label), input);
    /* the video options, shown once a preview says the chosen tasks have videos (#636) */
    const videoOpts = el('div', { class: 'ds-video-opts', hidden: true },
      field(t('ds.videos'), videosSel), el('p', { class: 'muted ds-help' }, t('ds.videos_help')),
      el('div', { class: 'field ds-switch' }, el('label', { for: 'ds-gap-on', class: 'ds-check' }, gapBox, el('span', null, t('ds.gap_switch')))),
      gapField, gapOff);

    function drawRatios() {
      const parts = st.mode === 'split'
        ? [['train', int(train)], ['val', int(val)], ['test', int(test)]]
        : [['fold-0', 100 - (int(ktestInput) || 0)], ['test', int(ktestInput) || 0]];
      fill(ratioBar, ...parts.map(([cl, n]) => el('span', { class: `ds-seg ds-${cl}`, style: `flex-grow:${Number.isFinite(n) ? n : 0}` })));
      if (st.mode === 'split') {
        const sum = [train, val, test].reduce((a, i) => a + (raw(i) || 0), 0);
        sumText.className = `ds-sum ${sum === 100 && [train, val, test].every(validPct) ? 'ok' : 'bad'}`;
        sumText.textContent = t('ds.sum', { n: sum });
      } else {
        sumText.className = 'ds-sum ok';
        sumText.textContent = t('ds.kfold_sum', { k: kInput.value || '?', train: 100 - (int(ktestInput) || 0), test: int(ktestInput) || 0 });
      }
    }
    function drawMode() {
      fill(modeBox,
        st.mode === 'split'
          ? el('div', { class: 'ds-nums' }, field(t('ds.pct_train'), train), field(t('ds.pct_val'), val), field(t('ds.pct_test'), test))
          : el('div', { class: 'ds-nums' }, field(t('ds.k'), kInput), field(t('ds.pct_test'), ktestInput)),
        st.mode === 'split' ? null : kHelp,
        ratioBar, sumText);
      drawRatios();
    }
    /* the three percentages stay linked: changing one adjusts the next so the sum is 100 */
    train.addEventListener('input', () => {
      const a = int(train);
      if (Number.isNaN(a)) return;
      const v = Math.min(int(val) || 0, 100 - a);
      val.value = String(v);
      test.value = String(100 - a - v);
    });
    val.addEventListener('input', () => {
      const v = int(val);
      if (Number.isNaN(v)) return;
      const m = Math.min(v, 100 - (int(train) || 0));
      if (m !== v) val.value = String(m);
      test.value = String(100 - (int(train) || 0) - m);
    });
    test.addEventListener('input', () => {
      const x = int(test);
      if (Number.isNaN(x)) return;
      const m = Math.min(x, 100 - (int(train) || 0));
      if (m !== x) test.value = String(m);
      val.value = String(100 - (int(train) || 0) - m);
    });

    /* the request body, or the reason it cannot be sent */
    function options() {
      if (!st.tasks.size) return { error: t('ds.err_tasks') };
      const sd = parseInt(seed.value, 10);
      if (!Number.isFinite(sd)) return { error: t('ds.err_seed') };
      const gap = gapBox.checked ? parseFloat(String(gapInput.value).replace(',', '.')) : 0;
      const badGap = !(gap >= 0 && gap <= 3600) || (gapBox.checked && gap === 0);
      if (badGap) gapInput.setAttribute('aria-invalid', 'true'); else gapInput.removeAttribute('aria-invalid');
      if (badGap) return { error: t('ds.err_gap') };
      const body = { tasks: [...st.tasks], background: cls ? false : st.bg, seed: sd, videos: videosSel.value, gap_seconds: gap };
      if (st.mode === 'split') {
        const s = { train: raw(train), val: raw(val), test: raw(test) };
        const bad = ['train', 'val', 'test'].find((k) => !validPct({ train, val, test }[k]));
        if (bad) return { error: t('ds.err_pct_range', { part: partName(bad) }) };
        if (s.train < 1) return { error: t('ds.err_train_min') };
        if (s.train + s.val + s.test !== 100) return { error: t('ds.err_sum_is', { n: s.train + s.val + s.test }) };
        body.split = s;
        body.kfold = null;
      } else {
        const k = parseInt(kInput.value, 10);
        const tp = int(ktestInput);
        if (st.maxK != null && st.maxK < 2) return { error: t('ds.k_none') };
        if (st.maxK != null && k > st.maxK) return { error: t('ds.err_k_max', { max: num(st.maxK) }) };
        if (!(k >= 2 && k <= 10)) return { error: t('ds.err_k') };
        if (!validPct(ktestInput) || tp > 90) return { error: t('ds.err_test_range') };
        body.split = { train: 100 - tp, val: 0, test: tp };
        body.kfold = k;
      }
      return { body };
    }

    // ---- live preview
    const previewBox = el('div', { class: 'ds-preview', 'aria-live': 'polite' });
    const showMsg = (text, cl = 'muted') => fill(previewBox, el('p', { class: cl, role: cl === 'ds-error' ? 'alert' : null }, text));

    /* the most folds the images allow (from the preview): limits K in the field and says why */
    function limitK(maxK) {
      st.maxK = Number.isFinite(maxK) ? maxK : null;
      if (st.maxK == null) return;
      kInput.max = String(Math.max(2, Math.min(10, st.maxK)));
      kHelp.textContent = st.maxK < 2 ? t('ds.k_none') : t('ds.k_max', { max: num(st.maxK) });
      kfoldRadio.disabled = st.maxK < 2;
    }

    function renderPreview(d) {
      const sizes = normSizes(d);
      const items = partItems(sizes, d.options);
      limitK(d.max_kfold);
      videoOpts.hidden = !(Array.isArray(d.videos) && d.videos.length);
      const blocked = (d.empty_parts || []).filter((p) => p === 'train' || p === 'val');
      st.blocked = blocked.length > 0;
      const reason = trainBlockReason(sizes);
      const emptyNote = blocked.length
        ? el('li', { class: 'report-line ds-warn ds-blocked', role: 'alert' }, icon('warning'),
          el('div', null,
            el('p', null, t('ds.empty_blocked', { parts: blocked.map((p) => lower(partName(p))).join(t('ds.and')) })),
            el('label', { for: allowBox.id, class: 'ds-check' }, allowBox, el('span', null, t('ds.allow_empty')))))
        : reason ? el('li', { class: 'report-line ds-warn ds-blocked' }, icon('warning'), reason) : null;
      const total = totalOf(items);
      const stats = d.stats || {};
      const per = items.map((x) => statFor(stats, x));
      const bgTotal = per.reduce((a, s) => a + (Number(s.background) || 0), 0);
      const names = [];
      per.forEach((s) => Object.keys(s.objects || {}).forEach((c) => { if (!names.includes(c)) names.push(c); }));
      const head = el('tr', null, el('th', { scope: 'col' }, ''), ...items.map((x) => el('th', { scope: 'col' }, x.label)));
      const row = (label, cells, cl = '') => el('tr', { class: cl }, el('th', { scope: 'row' }, label), ...cells.map((c) => el('td', null, num(c))));
      /* a class row: the count, the share of the class in each part, and a marked 0 in a part that has images */
      const classRow = (c, cells, zero) => {
        const sum = cells.reduce((a, n) => a + n, 0);
        return el('tr', { class: zero ? 'ds-rare' : '' }, el('th', { scope: 'row' }, c),
          ...cells.map((n, i) => (n === 0 && items[i].n > 0
            ? el('td', { class: 'ds-zero', title: t('ds.zero_help', { class: c, part: items[i].label }) }, icon('warning', 12), el('strong', null, '0'), el('span', { class: 'sr-only' }, ` ${t('ds.zero_sr')}`))
            : el('td', { title: sum ? t('ds.class_share', { pct: pctOf(n, sum) }) : null }, num(n), el('small', { class: 'ds-share' }, sum ? `${pctOf(n, sum)}%` : '')))));
      };
      /* the rarest classes first (those missing from a part before the others): they are the ones at risk */
      const ranked = names.map((c) => {
        const cells = per.map((s) => (s.objects || {})[c] || 0);
        const zero = cells.some((n, i) => n === 0 && items[i].n > 0);
        return { c, cells, zero, sum: cells.reduce((a, n) => a + n, 0) };
      }).sort((a, b) => (Number(b.zero) - Number(a.zero)) || (a.sum - b.sum) || a.c.localeCompare(b.c));
      const classList = ranked.map((r) => r.c);
      const classRows = ranked.map((r) => classRow(r.c, r.cells, r.zero));
      const SHOWN = 8;
      let all = false;
      const body = el('tbody');
      const balanceBox = el('div', { class: 'ds-balance' });
      const more = el('button', { type: 'button', class: 'btn small', 'aria-expanded': 'false', onclick: () => { all = !all; paint(); } });
      function paint() {
        fill(body,
          row(bgTotal > 0 ? t('ds.images_with_bg') : t('ds.images'), items.map((x, i) => (per[i].images != null ? per[i].images : x.n)), 'strong'),
          bgTotal > 0 ? row(t('ds.background'), per.map((s) => s.background || 0)) : null,
          classRows.length ? el('tr', { class: 'sub' }, el('th', { scope: 'colgroup', colspan: String(items.length + 1) }, cls ? t('ds.images_per_class') : t('ds.objects_per_class'))) : null,
          ...(all ? classRows : classRows.slice(0, SHOWN)));
        const shown = (all ? classRows : classRows.slice(0, SHOWN)).length;
        const chart = partBalance(items, per, classList.slice(0, shown));
        fill(balanceBox, chart ? el('h4', { class: 'section-title' }, t('dist.balance_title')) : null, chart ? el('p', { class: 'muted ds-help' }, t('dist.balance_help')) : null, chart);
        more.hidden = classRows.length <= SHOWN;
        more.textContent = all ? t('ds.show_less') : t('ds.show_all', { n: num(classRows.length) });
        more.setAttribute('aria-expanded', String(all));
      }
      paint();
      const warns = cleanWarnings(d.warnings, items).filter((w) => !(w.code === 'part_empty' && blocked.includes(w.part)))  // said above, with the choice
        .filter((w) => w.code !== 'video_parts_shared')
        .map((w) => el('li', { class: 'report-line ds-warn' }, icon('warning'), warnText(w, items)));
      const infos = videoInfos(d.warnings).map((w) => el('li', { class: 'report-line ds-info' }, icon('info'), videoText(w, d.videos)));
      fill(previewBox,
        el('h3', { class: 'section-title' }, t('ds.preview'), el('span', { class: 'muted' }, ` · ${tn('status.images_active', total)}`)),
        partsBar(items, barLabel(items)),
        legend(items),
        el('div', { class: 'ds-table-wrap' }, el('table', { class: 'ds-table' }, el('thead', null, head), body)),
        more,
        balanceBox,
        sharedAlert(sharedVideos(d.warnings)),
        warns.length || emptyNote ? el('ul', { class: 'ds-warns', 'aria-label': t('ds.warnings') }, emptyNote, ...warns) : el('p', { class: 'report-line ds-ok' }, icon('check'), t('ds.no_warnings')),
        infos.length ? el('ul', { class: 'ds-warns ds-infos' }, ...infos) : null);
    }

    async function runPreview() {
      const opts = options();
      if (opts.error) return;
      const mine = ++token;
      try {
        const d = await api('POST', `/projects/${projectId}/datasets/preview`, { json: opts.body });
        if (mine !== token) return;
        renderPreview(d);
        syncCreate();
      } catch (err) {
        if (mine !== token) return;
        create.disabled = true;  // a version the server refuses to preview cannot be created
        if (err.code === 'kfold_too_few_groups') limitK(err.params.groups);  // that many groups is the most folds
        showMsg(err.message || String(err), 'ds-error');
      }
    }
    function changed() {
      drawRatios();
      clearTimeout(timer);
      token += 1;
      allowBox.checked = false;  // a new choice of proportions needs a new "create anyway"
      st.blocked = false;
      const opts = options();
      create.disabled = !!opts.error;
      if (opts.error) { showMsg(opts.error, 'ds-error'); return; }
      showMsg(t('ds.previewing'));
      timer = setTimeout(runPreview, 400);
    }

    let kfoldRadio = null;
    const modeOpt = (mode, label) => {
      const input = el('input', { type: 'radio', name: 'ds-mode', id: `ds-mode-${mode}`, value: mode, class: 'sr-only', checked: st.mode === mode, onchange: () => { st.mode = mode; drawMode(); changed(); } });
      if (mode === 'kfold') kfoldRadio = input;
      return el('label', { for: input.id, class: 'seg-opt' }, input, label);
    };
    const bgRow = cls ? null : el('div', { class: 'field ds-switch' },
      el('label', { for: 'ds-bg', class: 'ds-check' }, bgBox, el('span', null, t('ds.background_switch'))),
      el('p', { class: 'muted ds-help' }, t('ds.background_help')));
    bgBox.addEventListener('change', () => { st.bg = bgBox.checked; changed(); });

    const form = el('form', { class: 'dialog-body form ds-form', method: 'dialog', novalidate: true },
      el('h2', null, t('ds.new_title')),
      el('div', { class: 'ds-form-grid' },
        el('div', { class: 'ds-opts' },
          el('fieldset', { class: 'field' }, el('legend', null, t('ds.tasks')), el('ul', { class: 'ds-tasks' }, ...taskBoxes)),
          el('fieldset', { class: 'field' }, el('legend', null, t('ds.split')),
            el('div', { class: 'seg', role: 'radiogroup', 'aria-label': t('ds.split') }, modeOpt('split', t('ds.mode_split')), modeOpt('kfold', t('ds.mode_kfold'))),
            modeBox),
          bgRow,
          el('details', { class: 'ds-adv field' }, el('summary', null, t('ds.advanced')),
            field(t('ds.seed'), seed), el('p', { class: 'muted ds-help' }, t('ds.seed_help')),
            videoOpts),
          el('div', { class: 'field' }, el('label', { for: 'ds-note' }, `${t('ds.note')} `, el('span', { class: 'muted' }, `(${t('step.optional')})`)), note)),
        el('aside', { class: 'ds-side', 'aria-label': t('ds.preview') }, previewBox)),
      el('div', { class: 'form-actions' }, create, el('button', { type: 'button', class: 'btn', onclick: () => dlg.close() }, t('common.cancel'))));
    const dlg = el('dialog', { class: 'dialog wide ds-dialog', 'aria-label': t('ds.new_title') }, form);
    [seed, kInput, ktestInput, train, val, test, gapInput].forEach((i) => i.addEventListener('input', changed));
    videosSel.addEventListener('change', changed);
    /* the most folds depends on the test share: an older preview's limit must not clamp the new K (#636) */
    ktestInput.addEventListener('input', () => { st.maxK = null; kInput.max = '10'; });
    kInput.addEventListener('change', () => {
      const k = parseInt(kInput.value, 10);
      if (st.maxK >= 2 && k > st.maxK) { kInput.value = String(st.maxK); changed(); }
    });

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const opts = options();
      if (opts.error) return;
      create.disabled = true;
      try {
        const body = { ...opts.body, note: note.value.trim() };
        if (st.blocked && allowBox.checked) body.allow_empty = true;
        const v = await api('POST', `/projects/${projectId}/datasets`, { json: body });
        dlg.close();
        toast(t('ds.created', { n: v.version }), 'success', 4000);
        await load();
      } catch (err) {
        create.disabled = false;
        toastError(err);
      }
    });
    dlg.addEventListener('close', () => { clearTimeout(timer); token += 1; dlg.remove(); returnFocus(trigger, opener, node.querySelector('.ds-empty-actions .btn')); });
    drawMode();
    syncGap();
    document.body.appendChild(dlg);
    openModal(dlg);
    changed();
    const first = dlg.querySelector('input');
    if (first) first.focus();
  }

  const newBtn = el('button', { type: 'button', class: 'btn primary', id: 'ds-new', onclick: once(() => openNew(newBtn)) }, icon('plus', 16), t('ds.new'));
  add(node,
    el('div', { class: 'ds-head' },
      el('div', null, el('h2', { id: 'h-datasets' }, t('step.datasets')), el('p', { class: 'muted' }, t('ds.intro'))),
      newBtn),
    listBox);

  panel.mount = load;
  panel.refresh = load;
  /** Number of versions for the step's status line, before the panel is opened. */
  panel.count = async () => {
    try {
      ctx.datasetCount = ((await api('GET', `/projects/${projectId}/datasets`)).versions || []).length;
      renderStepper();
    } catch (err) { /* the status stays "no versions" */ }
  };
  return panel;
}
