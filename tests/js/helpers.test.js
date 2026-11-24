'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { loadApp } = require('./helpers/load-app');

const en = loadApp('en');
const pt = loadApp('pt');

test('fmt fills {name} placeholders and keeps unknown ones', () => {
  assert.equal(en.fmt('{a} and {b}', { a: 1, b: 'x' }), '1 and x');
  assert.equal(en.fmt('{a} and {b}', { a: 1 }), '1 and {b}');
  assert.equal(en.fmt('no placeholders'), 'no placeholders');
});

test('t returns the translation, or the key itself when it is unknown', () => {
  assert.equal(en.t('projects.title'), 'Projects');
  assert.equal(pt.t('projects.title'), 'Projetos');
  assert.equal(en.t('no.such.key'), 'no.such.key');
});

test('tr returns the fallback for an unknown key', () => {
  assert.equal(en.tr('task.classification', 'x'), 'Classification');
  assert.equal(pt.tr('task.classification', 'x'), 'Classificação');
  assert.equal(en.tr('task.no_such_task', 'fallback'), 'fallback');
});

test('tn: exactly 1 is singular, everything else (0 included) is plural, in both languages', () => {
  assert.equal(en.tn('images.added', 1), '1 image added');
  assert.equal(en.tn('images.added', 3), '3 images added');
  assert.equal(en.tn('images.added', 0), '0 images added');
  assert.match(pt.tn('images.added', 0), /^0 imagens/);
  assert.match(pt.tn('images.added', 1), /^1 imagem /);
});

test('tn formats the number with the interface language', () => {
  assert.equal(en.tn('images.added', 1234), '1,234 images added');
  assert.match(pt.tn('images.added', 1234), /^1\.234 imagens/);
});

test('num formats numbers and turns anything that is not a number into 0', () => {
  assert.equal(en.num(1234567), '1,234,567');
  assert.equal(pt.num(1234567), '1.234.567');
  assert.equal(en.num(undefined), '0');
  assert.equal(en.num('abc'), '0');
});

test('fmtBytes picks B, KB or MB and one decimal at most', () => {
  assert.equal(en.fmtBytes(512), '512 B');
  assert.equal(en.fmtBytes(1536), '1.5 KB');
  assert.equal(en.fmtBytes(5 * 1024 * 1024), '5 MB');
  assert.equal(pt.fmtBytes(1536), '1,5 KB');
});

test('esc escapes the characters that matter in HTML', () => {
  assert.equal(en.esc('<a href="x">\'&\'</a>'), '&lt;a href=&quot;x&quot;&gt;&#39;&amp;&#39;&lt;/a&gt;');
});

test('isHttp accepts only http and https addresses', () => {
  assert.equal(en.isHttp('https://example.org'), true);
  assert.equal(en.isHttp('HTTP://example.org'), true);
  assert.equal(en.isHttp('javascript:alert(1)'), false);
  assert.equal(en.isHttp(null), false);
});
