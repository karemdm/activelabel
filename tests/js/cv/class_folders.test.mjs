import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' });
for (const [name, value] of Object.entries({ window, document: window.document, Node: window.Node, localStorage: window.localStorage })) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const { checkClassFolders, classFoldersCard, classRows, mayHaveClassFolders } = await import('../../../src/activelabel/web/static/js/cv/class_folders.js');

setMessages({
  'images.folders_title': 'Class folders found', 'images.folders_use': 'Use folder names as classes', 'images.folders_skip': 'Not now',
  'images.folders_existing': 'Existing classes are reused', 'images.folders_split': 'Left out: {names}',
  'images.folders_labelled_one': '{n} image got its class', 'images.folders_labelled_other': '{n} images got their class',
  'images.folders_new_classes_one': '{n} new class', 'images.folders_new_classes_other': '{n} new classes',
  'ann.new': 'new',
}, 'en');

const report = {
  found: true, token: 'abcd', split_folders: ['train', 'val'],
  classes_found: ['cat', 'dog', 'unused'], classes_new: ['dog'],
  counts: { images_matched: 5, images_without_labels: 0, labels_per_class: { cat: 3, dog: 2 } },
  problems: [],
};

test('an upload may hold class folders when a name is inside a folder or a zip is sent', () => {
  assert.equal(mayHaveClassFolders([{ name: 'a.png' }, { name: 'b.jpg' }]), false);
  assert.equal(mayHaveClassFolders([{ name: 'pets/cat/a.png' }]), true);
  assert.equal(mayHaveClassFolders([{ name: 'pets\\cat\\a.png' }]), true);   // a Windows path
  assert.equal(mayHaveClassFolders([{ name: 'SET.ZIP' }]), true);
  assert.equal(mayHaveClassFolders([{ name: '/a.png' }]), false);
  assert.equal(mayHaveClassFolders([]), false);
});

test('the classes of a report are the ones in use, with their count and whether they are new', () => {
  assert.deepEqual(classRows(report), [
    { name: 'cat', count: 3, isNew: false },
    { name: 'dog', count: 2, isNew: true },
  ]);   // "unused" labels no image
  assert.deepEqual(classRows({}), []);
});

function stubFetch(answers) {
  const calls = [];
  const real = globalThis.fetch;
  globalThis.fetch = async (url, init = {}) => {
    calls.push({ url, method: init.method, body: init.body });
    const answer = answers[`${init.method} ${url}`];
    return { ok: true, status: 200, text: async () => JSON.stringify(answer === undefined ? {} : answer) };
  };
  return { calls, restore: () => { globalThis.fetch = real; } };
}

test('the check answers null when the images are not in class folders', async () => {
  const stub = stubFetch({ 'POST /api/projects/3/images/class-folders': { found: false } });
  try {
    assert.equal(await checkClassFolders(3), null);
  } finally { stub.restore(); }
  assert.equal(stub.calls[0].method, 'POST');
});

test('the card previews the classes, marks the new ones and says what was left out', () => {
  const card = classFoldersCard({ projectId: 3, report, onApplied: async () => {} });
  assert.match(card.textContent, /Class folders found/);
  assert.deepEqual([...card.querySelectorAll('.chip')].map((c) => c.firstChild.textContent), ['cat', 'dog']);
  assert.equal(card.querySelectorAll('.chip.is-new').length, 1);
  assert.match(card.textContent, /Existing classes are reused/);
  assert.match(card.textContent, /Left out: train, val/);
  assert.ok(card.querySelector('#class-folders-use') && card.querySelector('#class-folders-skip'));
});

test('no "existing classes" note and no split note when the project had no classes and no split folders', () => {
  const plain = { ...report, classes_new: [], split_folders: [] };
  const card = classFoldersCard({ projectId: 3, report: plain, onApplied: async () => {} });
  assert.doesNotMatch(card.textContent, /Existing classes/);
  assert.doesNotMatch(card.textContent, /Left out/);
});

test('"Use folder names as classes" imports in merge mode and reads the project again', async () => {
  const stub = stubFetch({ 'POST /api/projects/3/annotations/abcd/apply': { images_added: 5, classes_added: ['cat', 'dog'] } });
  let applied = 0;
  const card = classFoldersCard({ projectId: 3, report, onApplied: async () => { applied += 1; } });
  try {
    card.querySelector('#class-folders-use').click();
    await new Promise((resolve) => setTimeout(resolve, 20));
  } finally { stub.restore(); }
  assert.deepEqual(JSON.parse(stub.calls[0].body), { action: 'import', mode: 'merge' });
  assert.equal(applied, 1);
  assert.match(card.textContent, /5 images got their class/);
  assert.match(card.textContent, /2 new classes: cat, dog/);
});

test('"Not now" throws the staged check away and removes the card', async () => {
  const stub = stubFetch({});
  const holder = document.createElement('div');
  const card = classFoldersCard({ projectId: 3, report, onApplied: async () => { throw new Error('not applied'); } });
  holder.appendChild(card);
  try {
    card.querySelector('#class-folders-skip').click();
    await new Promise((resolve) => setTimeout(resolve, 20));
  } finally { stub.restore(); }
  assert.equal(stub.calls[0].method, 'DELETE');
  assert.equal(stub.calls[0].url, '/api/projects/3/annotations/abcd');
  assert.equal(holder.children.length, 0);
});
