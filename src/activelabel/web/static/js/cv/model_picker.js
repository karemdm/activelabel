/* The model of a new run in three fields (#632): family, then architecture (or generation), then its size or
 * variant. The catalogue gives every model its `arch` and `variant`; the task comes from the project type, so
 * it is never a choice (it is said in the summary line). A family names the two fields in its own words:
 * Ultralytics "Generation" and "Size", torchvision and timm "Architecture" and "Variant", smp "Decoder" and
 * "Encoder", transformers "Model" and "Size". Short variant lists are a radio group (arrow keys move, a
 * 44 px target each); long ones a list. */
import { el, fill } from '../shell/dom.js';
import { num, nf1, t, tr } from '../shell/i18n.js';

/** Most options a radio group shows; a longer list (or a long name) is a select. */
export const MAX_RADIOS = 8;
/** Longest variant name a radio group shows ("Ti/14 336" fits; longer names go in a select). */
export const MAX_RADIO_TEXT = 9;

/** The architectures of a family, in catalogue order. `models` are the catalogue's; returns their names. */
export const archsOf = (models, family) => [...new Set(models.filter((m) => m.family === family).map((m) => m.arch || m.label))];

/** The models of one architecture of a family, in catalogue order (smallest first in the catalogue). */
export const variantsOf = (models, family, arch) => models.filter((m) => m.family === family && (m.arch || m.label) === arch);

/** The name of a model inside its architecture: its variant, or nothing when the architecture has only it. */
export const variantName = (m) => m.variant || '';

/** Whether the variants of an architecture fit a radio group (few, with short names). */
export const asRadios = (list) => list.length > 1 && list.length <= MAX_RADIOS && list.every((m) => m.variant && m.variant.length <= MAX_RADIO_TEXT);

/** The sizes of Ultralytics and of the other families that name them so, smallest first. */
export const SIZES = ['n', 's', 'm', 'l', 'x'];

/** Whether a list of variants goes from the smallest network to the largest (then "faster ... more accurate" is
 *  true of it): at least three, named n/s/m/l/x in that order, or every one with a known size never smaller than
 *  the one before. */
export const ordered = (list) => list.length >= 3 && (
  list.every((m, i) => i === 0 || SIZES.indexOf(m.variant) > SIZES.indexOf(list[i - 1].variant)) && SIZES.includes(list[0].variant)
  || list.every((m, i) => m.params_m != null && (i === 0 || m.params_m >= list[i - 1].params_m)));

/** The model a family proposes first: its default architecture (`defaultArch`, from the catalogue) or its first
 *  one, in the size this computer suits (`size`, from /training/environment) when the architecture has it,
 *  else its first variant. Returns the model, or null when the family has none. */
export function defaultModel(models, family, defaultArch = {}, size = null) {
  const archs = archsOf(models, family);
  if (!archs.length) return null;
  const arch = archs.includes(defaultArch[family]) ? defaultArch[family] : archs[0];
  const list = variantsOf(models, family, arch);
  return list.find((m) => size && m.variant === size) || list[0];
}

/** "25.9 M parameters · ~99 MB" (in the interface language), or '' when the catalogue does not know the size. */
export function sizeFacts(m) {
  if (m.params_m == null) return '';
  return t('tr.size_facts', { params: nf1.format(m.params_m), mb: num(Math.round(m.file_mb)) });
}

/** The name of a catalogue model on every screen, in one style whatever its family (#668): its architecture and
 *  its variant with a space ("ResNet 18", "EfficientNet B5", "YOLO26 m"); the catalogue label only when the model
 *  has no architecture. torchvision and timm name the same networks differently ("ResNet-18", "ResNet 18"). */
export const modelName = (m) => (m.arch ? [m.arch, m.variant].filter(Boolean).join(' ') : m.label || m.id);

/** Parameters (in millions) from which a model is "very large" for a computer without a GPU (#668): ViT-L, the
 *  largest RegNets and EfficientNets. */
export const LARGE_ON_CPU_M = 100;

/** Whether `m` is very large for training on the CPU. */
export const largeForCpu = (m) => m.params_m != null && m.params_m >= LARGE_ON_CPU_M;

/** One line that names the chosen model: "Ultralytics YOLO26 m · Instance". */
export function modelSummary(m) {
  const name = [tr(`tr.family.${m.family}`, m.family), m.arch || m.label, variantName(m)].filter(Boolean).join(' ');
  return `${name} · ${tr(`tr.kind.${m.task_kind}`, m.task_kind)}`;
}

/** The architecture and variant fields of the new-run dialog. `models` are the catalogue's for this project,
 *  `family` and `model` the starting choice; `onChange(model, sizeOnly)` is called when the person picks another
 *  model (`sizeOnly` when only the variant changed, inside the same architecture). Returns
 *  `{archField, variantField, summary, setFamily(family, model), setSuggested(size), get()}`. */
export function modelPicker({ models, family, model, onChange }) {
  const st = { family, model, suggested: null };
  const now = () => models.find((m) => m.id === st.model);
  const archSel = el('select', { id: 'tr-arch', class: 'tr-select', 'aria-describedby': 'tr-model-summary' });
  const archLabel = el('label', { for: 'tr-arch' });
  const archField = el('div', { class: 'field' }, archLabel, archSel);
  const variantBox = el('div');
  const facts = el('p', { class: 'hint tr-size-facts', id: 'tr-size-facts' });
  const variantField = el('fieldset', { class: 'field tr-variant' });
  const summary = el('p', { class: 'tr-model-summary', id: 'tr-model-summary' });

  function pick(id) {
    const before = now();
    st.model = id;
    const m = now();
    draw();
    onChange(m, (before.arch || before.label) === (m.arch || m.label));
  }
  function drawArchs() {
    const mine = models.filter((m) => m.family === st.family);
    const kinds = [...new Set(mine.map((m) => m.task_kind))];
    const option = (a) => el('option', { value: a }, a);
    /* a family with two kinds of model for this project (masks of instances and semantic masks) groups by kind */
    fill(archSel, ...(kinds.length > 1
      ? kinds.map((k) => el('optgroup', { label: tr(`tr.kind.${k}`, k) }, ...[...new Set(mine.filter((m) => m.task_kind === k).map((m) => m.arch || m.label))].map(option)))
      : archsOf(models, st.family).map(option)));
    archLabel.textContent = tr(`tr.arch.${st.family}`, t('tr.arch'));
  }
  function drawVariants() {
    const m = now();
    const list = variantsOf(models, st.family, m.arch || m.label);
    const legend = el('legend', null, tr(`tr.variant.${st.family}`, t('tr.variant')));
    variantField.hidden = !variantName(m);
    if (asRadios(list)) {
      const cols = list.length <= 5 ? list.length : Math.ceil(list.length / 2);   // a phone shows two rows of up to 4
      fill(variantBox, el('div', { class: 'tr-sizes', role: 'radiogroup', 'aria-describedby': 'tr-size-facts', style: `--n: ${list.length}; --n-phone: ${cols}` },
        ...list.map((v) => {
          const input = el('input', { type: 'radio', name: 'tr-size', id: `tr-size-${v.id}`, value: v.id, class: 'sr-only', onchange: () => pick(v.id) });
          input.checked = v.id === m.id;
          return el('label', { class: 'tr-size', for: input.id, title: [v.label, sizeFacts(v)].filter(Boolean).join(' · ') }, input, v.variant);
        })),
      ordered(list) ? el('p', { class: 'tr-size-scale', 'aria-hidden': 'true' }, el('span', null, t('tr.size_faster')), el('span', null, t('tr.size_accurate'))) : null);
    } else if (list.length > 1) {
      const select = el('select', { id: 'tr-variant', class: 'tr-select', 'aria-describedby': 'tr-size-facts' }, ...list.map((v) => el('option', { value: v.id }, v.variant)));
      select.value = m.id;
      select.addEventListener('change', () => pick(select.value));
      fill(variantBox, select);
    } else {
      fill(variantBox, el('p', { class: 'tr-variant-only' }, variantName(m)));   // the only one: said, not chosen
    }
    fill(variantField, legend, variantBox, facts);
    /* the scale drawn under the buttons is hidden from screen readers: they hear it here, after the facts */
    const said = [sizeFacts(m), m.variant && m.variant === st.suggested ? t('tr.size_suggested') : ''].filter(Boolean).join(' · ');
    const range = `${t('tr.size_faster')} → ${t('tr.size_accurate')}`;
    fill(facts, said, ordered(list) && asRadios(list) ? el('span', { class: 'sr-only' }, said ? ` (${range})` : range) : null);
  }
  function draw() {
    const m = now();
    archSel.value = m.arch || m.label;
    drawVariants();
    summary.textContent = modelSummary(m);
    summary.title = m.id;
  }
  /* another architecture: the variant of the same name when it has one (YOLO11 m -> YOLO26 m), else the size this
     computer suits, else its first */
  archSel.addEventListener('change', () => {
    const list = variantsOf(models, st.family, archSel.value);
    const same = list.find((v) => v.variant && v.variant === now().variant);
    pick((same || list.find((v) => v.variant === st.suggested) || list[0]).id);
  });

  drawArchs();
  draw();
  return {
    archField,
    variantField,
    summary,
    /** Shows another family, at `model` (its default). */
    setFamily(f, m) { st.family = f; st.model = m.id; drawArchs(); draw(); },
    /** The size this computer suits (n, s or m); proposes it when the person has not changed the model yet. */
    setSuggested(size, untouched) {
      st.suggested = size;
      const m = now();
      const there = variantsOf(models, st.family, m.arch || m.label).find((v) => v.variant === size);
      if (untouched && there && there.id !== m.id) { st.model = there.id; draw(); onChange(there, true); return; }
      draw();
    },
    get: now,
  };
}
