import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body></body></html>');
for (const [name, value] of Object.entries({ window, document: window.document, Node: window.Node })) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const names = await import('../../../src/activelabel/web/static/js/cv/names.js');

const { TASKS, ISSUES, SKIP_FILES, taskName, issueName, issueTag, partName, taskPicker, taskBadge, nextTaskNumber } = names;

setMessages({
  'task.detection': 'Detection', 'task.detection.help': 'Boxes', 'issue.blurry': 'Blurry images', 'tag.dark': 'Dark',
  'issue.dark': 'Dark images', 'ds.part_train': 'Training', 'ds.part_val': 'Validation', 'ds.part_test': 'Test',
}, 'en');

test('the four task types and the eight curation issues, in the order the screens show them', () => {
  assert.deepEqual(TASKS, ['classification', 'detection', 'segmentation', 'keypoints']);
  assert.equal(ISSUES.length, 8);
  assert.ok(SKIP_FILES.has('thumbs.db'));
});

test('names fall back to the raw id when the text is missing', () => {
  assert.equal(taskName('detection'), 'Detection');
  assert.equal(taskName('unknown'), 'unknown');
  assert.equal(taskName(undefined), '?');
  assert.equal(issueTag('dark'), 'Dark');
  assert.equal(issueTag('blurry'), 'Blurry images');  // no tag text: the issue name
  assert.equal(issueName('odd'), 'odd');
  assert.deepEqual(['train', 'val', 'test', 'other', null].map(partName), ['Training', 'Validation', 'Test', 'other', '']);
});

test('taskPicker is a radio group of four cards; set and a change move the selection', () => {
  const picker = taskPicker('t', 'classification');
  document.body.appendChild(picker.node);
  assert.equal(picker.node.getAttribute('role'), 'radiogroup');
  assert.equal(picker.node.querySelectorAll('input[type="radio"]').length, 4);
  assert.equal(picker.get(), 'classification');
  const input = picker.node.querySelector('input[value="detection"]');
  input.checked = true;
  input.dispatchEvent(new window.Event('change'));
  assert.equal(picker.get(), 'detection');
  assert.ok(picker.node.querySelector('label.selected[for="t-detection"]'));
  picker.set('keypoints');
  assert.ok(picker.node.querySelector('input[value="keypoints"]').checked);
  assert.equal(taskPicker('u').get(), 'detection');  // the default
  picker.node.remove();
});

test('taskBadge shows the icon and the name of the task', () => {
  const badge = taskBadge('detection');
  assert.equal(badge.className, 'badge task-badge task-detection');
  assert.ok(badge.querySelector('.icon-detection'));
  assert.equal(badge.textContent, 'Detection');
});

test('annotationStatus says how many images are annotated, or empty, or none', () => {
  setMessages({ 'status.annotated_one': '{n} annotated', 'status.annotated_other': '{n} annotated', 'status.empty_one': '{n} empty',
    'status.empty_other': '{n} empty', 'status.annotations_none': 'No annotations' }, 'en');
  assert.equal(names.annotationStatus({ annotated: 2, empty: 1 }), '2 annotated');
  assert.equal(names.annotationStatus({ annotated: 0, empty: 3 }), '3 empty');
  assert.equal(names.annotationStatus({}), 'No annotations');
});

test('deviceName names the devices of the training environment', () => {
  setMessages({ 'tr.device_auto': 'Automatic', 'tr.device_cpu': 'CPU', 'tr.device_gpu': 'GPU {n}', 'tr.device_mps': 'Apple GPU (MPS)' }, 'en');
  assert.deepEqual(['auto', 'cpu', 'cuda:1', 'mps', 'tpu'].map(names.deviceName), ['Automatic', 'CPU', 'GPU 1', 'Apple GPU (MPS)', 'tpu']);
});

test('nextTaskNumber follows the names like the server does, not the ids (#554)', () => {
  assert.equal(nextTaskNumber([]), 1);
  assert.equal(nextTaskNumber([{ id: 1, name: 'Campo' }]), 1);
  assert.equal(nextTaskNumber([{ id: 5, name: 'Task 2' }, { id: 6, name: 'task 7' }, { id: 7, name: 'Task 3 b' }]), 8);
});
