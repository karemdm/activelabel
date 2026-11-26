import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body></body></html>');
for (const [name, value] of Object.entries({ window, document: window.document, Node: window.Node })) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const picker = await import('../../../src/activelabel/web/static/js/cv/model_picker.js');

const { archsOf, variantsOf, asRadios, ordered, defaultModel, sizeFacts, modelSummary, modelPicker } = picker;

setMessages({
  'tr.size_facts': '{params} M parameters · ~{mb} MB',
  'tr.family.ultralytics': 'Ultralytics', 'tr.family.torchvision': 'torchvision',
  'tr.kind.detection': 'Detection', 'tr.kind.instance': 'Instance', 'tr.kind.semantic': 'Semantic',
  'tr.arch': 'Architecture', 'tr.variant': 'Variant', 'tr.arch.ultralytics': 'Generation', 'tr.variant.ultralytics': 'Size',
  'tr.size_faster': 'Faster', 'tr.size_accurate': 'More accurate', 'tr.size_suggested': 'Suggested for this computer',
}, 'en');

const mk = (id, family, arch, variant, params_m = null, file_mb = null, task_kind = 'detection') => (
  { id, family, label: `${arch} ${variant}`.trim(), arch, variant, task_kind, params_m, file_mb });

const SIZES = ['n', 's', 'm', 'l', 'x'];
const YOLO11_PARAMS = [2.6, 9.4, 20.1, 25.3, 56.9];
const YOLO11_MB = [5.4, 18.4, 38.8, 49, 109.3];
const CATALOGUE = [
  ...SIZES.map((s) => mk(`yolo26${s}`, 'ultralytics', 'YOLO26', s)),
  ...SIZES.map((s, i) => mk(`yolo11${s}`, 'ultralytics', 'YOLO11', s, YOLO11_PARAMS[i], YOLO11_MB[i])),
  mk('resnet18', 'torchvision', 'ResNet', '18', 11.7, 44.7),
  mk('resnet34', 'torchvision', 'ResNet', '34', 21.8, 83.3),
  mk('resnet50', 'torchvision', 'ResNet', '50', 25.6, 97.8),
  ...[0, 1, 2, 3, 4, 5, 6, 7].map((i) => mk(`effnet-b${i}`, 'torchvision', 'EfficientNet', `B${i}`, 5.3 + i * 10, 20 + i * 40)),
  mk('frcnn-r50', 'torchvision', 'Faster R-CNN', 'ResNet-50 FPN', 41.8, 160),
  mk('frcnn-mobile', 'torchvision', 'Faster R-CNN', 'MobileNetV3 FPN', 18.9, 74),
  mk('retinanet', 'torchvision', 'RetinaNet', 'ResNet-50 FPN', 34, 130),
  mk('googlenet', 'torchvision', 'GoogLeNet', '', 6.6, 50),
];
const byId = (id) => CATALOGUE.find((m) => m.id === id);

/** Mounts a picker with its three nodes in the document; `done()` removes them. */
function mount(opts) {
  const calls = [];
  const p = modelPicker({ models: CATALOGUE, onChange: (m, sizeOnly) => calls.push([m, sizeOnly]), ...opts });
  document.body.append(p.archField, p.variantField, p.summary);
  const done = () => { p.archField.remove(); p.variantField.remove(); p.summary.remove(); };
  return { p, calls, done };
}

test('archsOf lists the architectures of a family in catalogue order, once each (#632)', () => {
  assert.deepEqual(archsOf(CATALOGUE, 'ultralytics'), ['YOLO26', 'YOLO11']);
  assert.deepEqual(archsOf(CATALOGUE, 'torchvision'), ['ResNet', 'EfficientNet', 'Faster R-CNN', 'RetinaNet', 'GoogLeNet']);
  assert.deepEqual(archsOf(CATALOGUE, 'timm'), []);
  assert.deepEqual(archsOf([{ family: 'x', label: 'Only label' }], 'x'), ['Only label']);
});

test('variantsOf returns the models of one architecture of one family, smallest first (#632)', () => {
  assert.deepEqual(variantsOf(CATALOGUE, 'ultralytics', 'YOLO11').map((m) => m.variant), SIZES);
  assert.deepEqual(variantsOf(CATALOGUE, 'torchvision', 'Faster R-CNN').map((m) => m.id), ['frcnn-r50', 'frcnn-mobile']);
  assert.deepEqual(variantsOf(CATALOGUE, 'torchvision', 'YOLO11'), []);
  assert.deepEqual(variantsOf(CATALOGUE, 'ultralytics', 'ResNet'), []);
});

test('asRadios is true for n to x and B0 to B7 and false for one model, too many options or long names (#632)', () => {
  assert.equal(asRadios(variantsOf(CATALOGUE, 'ultralytics', 'YOLO26')), true);
  assert.equal(asRadios(variantsOf(CATALOGUE, 'torchvision', 'EfficientNet')), true);
  assert.equal(asRadios(variantsOf(CATALOGUE, 'torchvision', 'ResNet')), true);
  assert.equal(asRadios(variantsOf(CATALOGUE, 'torchvision', 'RetinaNet')), false);   // one model
  assert.equal(asRadios(variantsOf(CATALOGUE, 'torchvision', 'GoogLeNet')), false);
  assert.equal(asRadios(variantsOf(CATALOGUE, 'torchvision', 'Faster R-CNN')), false);   // "ResNet-50 FPN" is too long
  assert.equal(asRadios([]), false);
  const many = (n) => Array.from({ length: n }, (_, i) => mk(`v${i}`, 'x', 'A', `V${i}`));
  assert.equal(asRadios(many(8)), true);
  assert.equal(asRadios(many(9)), false);   // more than eight options
  assert.equal(asRadios([mk('a', 'x', 'A', 'abcdefghi'), mk('b', 'x', 'A', 'b')]), true);   // nine characters fit
  assert.equal(asRadios([mk('a', 'x', 'A', 'abcdefghij'), mk('b', 'x', 'A', 'b')]), false);   // ten do not
});

test('ordered is true for n/s/m/l/x and for growing sizes, false for fewer than three or unknown sizes (#632)', () => {
  assert.equal(ordered(variantsOf(CATALOGUE, 'ultralytics', 'YOLO26')), true);   // no params, but n/s/m/l/x
  assert.equal(ordered(variantsOf(CATALOGUE, 'ultralytics', 'YOLO26').slice(0, 3)), true);
  assert.equal(ordered(variantsOf(CATALOGUE, 'ultralytics', 'YOLO11')), true);
  assert.equal(ordered(variantsOf(CATALOGUE, 'torchvision', 'ResNet')), true);   // growing params
  assert.equal(ordered(variantsOf(CATALOGUE, 'torchvision', 'EfficientNet')), true);
  assert.equal(ordered(variantsOf(CATALOGUE, 'torchvision', 'Faster R-CNN')), false);   // only two
  assert.equal(ordered(variantsOf(CATALOGUE, 'ultralytics', 'YOLO26').slice(0, 2)), false);
  assert.equal(ordered([]), false);
  const unknown = ['18', '34', '50'].map((v) => mk(`r${v}`, 'x', 'R', v));
  assert.equal(ordered(unknown), false);   // neither sizes nor params
  const shuffled = ['s', 'n', 'm'].map((v) => mk(`y${v}`, 'x', 'Y', v));
  assert.equal(ordered(shuffled), false);
  const shrinking = [mk('a', 'x', 'A', 'a', 30), mk('b', 'x', 'A', 'b', 20), mk('c', 'x', 'A', 'c', 40)];
  assert.equal(ordered(shrinking), false);
});

test('defaultModel takes the default architecture, else the first one, in the suggested size when it has it (#632)', () => {
  assert.equal(defaultModel(CATALOGUE, 'ultralytics', { ultralytics: 'YOLO11' }).id, 'yolo11n');
  assert.equal(defaultModel(CATALOGUE, 'ultralytics', { ultralytics: 'YOLO11' }, 'm').id, 'yolo11m');
  assert.equal(defaultModel(CATALOGUE, 'ultralytics').id, 'yolo26n');   // no map: the first architecture
  assert.equal(defaultModel(CATALOGUE, 'ultralytics', { ultralytics: 'Nope' }).id, 'yolo26n');   // unknown default
  assert.equal(defaultModel(CATALOGUE, 'ultralytics', {}, 's').id, 'yolo26s');
  assert.equal(defaultModel(CATALOGUE, 'ultralytics', {}, 'z').id, 'yolo26n');   // the size is not there: the first variant
  assert.equal(defaultModel(CATALOGUE, 'torchvision', { torchvision: 'Faster R-CNN' }, 'm').id, 'frcnn-r50');
  assert.equal(defaultModel(CATALOGUE, 'timm', {}, 'm'), null);
});

test('sizeFacts is empty without params and says parameters and megabytes with them (#632)', () => {
  assert.equal(sizeFacts(byId('yolo26n')), '');
  assert.equal(sizeFacts(byId('yolo11n')), '2.6 M parameters · ~5 MB');
  assert.equal(sizeFacts(byId('yolo11x')), '56.9 M parameters · ~109 MB');
});

test('modelSummary names the family, architecture, variant and task in one line (#632)', () => {
  assert.equal(modelSummary(byId('yolo26m')), 'Ultralytics YOLO26 m · Detection');
  assert.equal(modelSummary(byId('googlenet')), 'torchvision GoogLeNet · Detection');   // no variant: nothing between
  assert.equal(modelSummary({ family: 'zzz', label: 'Label', variant: '', task_kind: 'odd' }), 'zzz Label · odd');   // unknown texts
});

test('the architecture label and the variant legend use the family own words when it has them (#632)', () => {
  const a = mount({ family: 'ultralytics', model: 'yolo26m' });
  assert.equal(a.p.archField.querySelector('label').textContent, 'Generation');
  assert.equal(a.p.variantField.querySelector('legend').textContent, 'Size');
  a.done();
  const b = mount({ family: 'torchvision', model: 'resnet18' });
  assert.equal(b.p.archField.querySelector('label').textContent, 'Architecture');
  assert.equal(b.p.variantField.querySelector('legend').textContent, 'Variant');
  b.done();
});

test('a short variant list is a radio group named tr-size with exactly one checked (#632)', () => {
  const { p, done } = mount({ family: 'ultralytics', model: 'yolo26m' });
  const group = p.variantField.querySelector('[role="radiogroup"]');
  assert.ok(group);
  const inputs = [...group.querySelectorAll('input')];
  assert.equal(inputs.length, 5);
  assert.ok(inputs.every((i) => i.type === 'radio' && i.name === 'tr-size'));
  assert.deepEqual(inputs.filter((i) => i.checked).map((i) => i.value), ['yolo26m']);
  assert.deepEqual(inputs.map((i) => i.parentNode.textContent), SIZES);
  assert.ok(p.variantField.querySelector('.tr-size-scale'));   // n..x goes from faster to more accurate
  assert.equal(p.variantField.hidden, false);
  assert.equal(p.variantField.querySelector('select'), null);
  done();
});

test('picking another size radio calls onChange with that model and sizeOnly true (#632)', () => {
  const { p, calls, done } = mount({ family: 'ultralytics', model: 'yolo26m' });
  const input = p.variantField.querySelector('input[value="yolo26l"]');
  input.checked = true;
  input.dispatchEvent(new window.Event('change'));
  assert.equal(calls.length, 1);
  assert.equal(calls[0][0].id, 'yolo26l');
  assert.equal(calls[0][1], true);
  assert.equal(p.get().id, 'yolo26l');
  const checked = [...p.variantField.querySelectorAll('input')].filter((i) => i.checked);
  assert.deepEqual(checked.map((i) => i.value), ['yolo26l']);
  assert.equal(p.summary.textContent, 'Ultralytics YOLO26 l · Detection');
  done();
});

test('changing the architecture keeps the variant name in the other generation and reports sizeOnly false (#632)', () => {
  const { p, calls, done } = mount({ family: 'ultralytics', model: 'yolo26m' });
  const select = p.archField.querySelector('select#tr-arch');
  assert.equal(select.value, 'YOLO26');
  select.value = 'YOLO11';
  select.dispatchEvent(new window.Event('change'));
  assert.equal(calls.length, 1);
  assert.equal(calls[0][0].id, 'yolo11m');
  assert.equal(calls[0][1], false);
  assert.equal(select.value, 'YOLO11');
  assert.equal(p.summary.textContent, 'Ultralytics YOLO11 m · Detection');
  assert.equal(p.summary.title, 'yolo11m');
  done();
});

test('changing to an architecture without the same variant picks its first one (#632)', () => {
  const { p, calls, done } = mount({ family: 'torchvision', model: 'resnet50' });
  const select = p.archField.querySelector('select#tr-arch');
  select.value = 'EfficientNet';
  select.dispatchEvent(new window.Event('change'));
  assert.equal(calls[0][0].id, 'effnet-b0');
  assert.equal(calls[0][1], false);
  done();
});

test('a long-name architecture gets a select #tr-variant that picks inside the architecture (#632)', () => {
  const { p, calls, done } = mount({ family: 'torchvision', model: 'frcnn-r50' });
  assert.equal(p.variantField.querySelector('[role="radiogroup"]'), null);
  const select = p.variantField.querySelector('select#tr-variant');
  assert.ok(select);
  assert.deepEqual([...select.options].map((o) => o.textContent), ['ResNet-50 FPN', 'MobileNetV3 FPN']);
  assert.equal(select.value, 'frcnn-r50');
  select.value = 'frcnn-mobile';
  select.dispatchEvent(new window.Event('change'));
  assert.equal(calls[0][0].id, 'frcnn-mobile');
  assert.equal(calls[0][1], true);
  done();
});

test('an architecture with one variant says it instead of offering a choice (#632)', () => {
  const { p, done } = mount({ family: 'torchvision', model: 'retinanet' });
  assert.equal(p.variantField.querySelector('.tr-variant-only').textContent, 'ResNet-50 FPN');
  assert.equal(p.variantField.querySelector('select'), null);
  assert.equal(p.variantField.querySelector('input'), null);
  assert.equal(p.variantField.hidden, false);
  done();
});

test('an empty variant hides the variant fieldset (#632)', () => {
  const { p, done } = mount({ family: 'torchvision', model: 'googlenet' });
  assert.equal(p.variantField.hidden, true);
  assert.equal(p.summary.textContent, 'torchvision GoogLeNet · Detection');
  done();
});

test('more than one kind of model groups the architectures by kind, one kind does not (#632)', () => {
  const seg = [
    mk('maskrcnn', 'torchvision', 'Mask R-CNN', 'ResNet-50 FPN', null, null, 'instance'),
    mk('fcn', 'torchvision', 'FCN', 'ResNet-50', null, null, 'semantic'),
    mk('deeplab50', 'torchvision', 'DeepLabV3', 'ResNet-50', null, null, 'semantic'),
    mk('deeplab101', 'torchvision', 'DeepLabV3', 'ResNet-101', null, null, 'semantic'),
  ];
  const grouped = modelPicker({ models: seg, family: 'torchvision', model: 'maskrcnn', onChange: () => {} });
  const groups = [...grouped.archField.querySelectorAll('optgroup')];
  assert.deepEqual(groups.map((g) => g.label), ['Instance', 'Semantic']);
  assert.deepEqual([...groups[1].querySelectorAll('option')].map((o) => o.value), ['FCN', 'DeepLabV3']);
  const plain = modelPicker({ models: CATALOGUE, family: 'ultralytics', model: 'yolo26n', onChange: () => {} });
  assert.equal(plain.archField.querySelectorAll('optgroup').length, 0);
  assert.deepEqual([...plain.archField.querySelectorAll('option')].map((o) => o.value), ['YOLO26', 'YOLO11']);
});

test('setSuggested moves an untouched picker to the suggested size and calls onChange (#632)', () => {
  const { p, calls, done } = mount({ family: 'ultralytics', model: 'yolo26n' });
  p.setSuggested('s', true);
  assert.equal(calls.length, 1);
  assert.equal(calls[0][0].id, 'yolo26s');
  assert.equal(calls[0][1], true);
  assert.equal(p.get().id, 'yolo26s');
  assert.deepEqual([...p.variantField.querySelectorAll('input')].filter((i) => i.checked).map((i) => i.value), ['yolo26s']);
  assert.ok(p.variantField.querySelector('.tr-size-facts').textContent.includes('Suggested for this computer'));
  p.setSuggested('s', true);   // already there: nothing to report
  assert.equal(calls.length, 1);
  done();
});

test('setSuggested leaves a touched picker, or a size the architecture lacks, where it is (#632)', () => {
  const { p, calls, done } = mount({ family: 'ultralytics', model: 'yolo26n' });
  p.setSuggested('s', false);
  assert.equal(calls.length, 0);
  assert.equal(p.get().id, 'yolo26n');
  p.setSuggested('q', true);
  assert.equal(calls.length, 0);
  assert.equal(p.get().id, 'yolo26n');
  done();
});

test('the summary title is the model id and setFamily shows another family (#632)', () => {
  const { p, done } = mount({ family: 'ultralytics', model: 'yolo26m' });
  assert.equal(p.summary.title, 'yolo26m');
  p.setFamily('torchvision', byId('resnet18'));
  assert.equal(p.summary.title, 'resnet18');
  assert.equal(p.summary.textContent, 'torchvision ResNet 18 · Detection');
  assert.equal(p.archField.querySelector('label').textContent, 'Architecture');
  assert.equal(p.get().id, 'resnet18');
  done();
});

test('modelName names every family in one style, and largeForCpu marks networks of 100 M parameters or more (#668)', () => {
  const { modelName, largeForCpu } = picker;
  assert.equal(modelName({ id: 'resnet18', label: 'ResNet-18', arch: 'ResNet', variant: '18' }), 'ResNet 18');
  assert.equal(modelName({ id: 'timm_resnet18', label: 'ResNet 18', arch: 'ResNet', variant: '18' }), 'ResNet 18');
  assert.equal(modelName({ id: 'tv_efficientnet_b5', label: 'EfficientNet-B5', arch: 'EfficientNet', variant: 'B5' }), 'EfficientNet B5');
  assert.equal(modelName({ id: 'yolo26m', label: 'YOLO26m', arch: 'YOLO26', variant: 'm' }), 'YOLO26 m');
  assert.equal(modelName({ id: 'googlenet', label: 'GoogLeNet', arch: 'GoogLeNet', variant: '' }), 'GoogLeNet');
  assert.equal(modelName({ id: 'x', label: 'Only a label' }), 'Only a label');
  assert.equal(largeForCpu({ params_m: 644.8 }), true);
  assert.equal(largeForCpu({ params_m: 100 }), true);
  assert.equal(largeForCpu({ params_m: 60.2 }), false);
  assert.equal(largeForCpu({ params_m: null }), false);
});
