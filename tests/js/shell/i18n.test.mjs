import test from 'node:test';
import assert from 'node:assert/strict';
import * as i18n from '../../../src/activelabel/web/static/js/shell/i18n.js';

const { setMessages, fmt, t, tr, tn, num, fmtDate, fmtBytes, lower } = i18n;

test('setMessages changes the texts and the locale that importers see (live bindings)', () => {
  setMessages({ hello: 'Olá, {name}' }, 'pt');
  assert.equal(i18n.LOCALE, 'pt');
  assert.equal(i18n.I18N.hello, 'Olá, {name}');
  assert.equal(t('hello', { name: 'Ana' }), 'Olá, Ana');
  assert.equal(num(1234.5), i18n.nf.format(1234.5));
  setMessages(null);
  assert.equal(i18n.LOCALE, 'en');
  assert.deepEqual(i18n.I18N, {});
});

test('fmt fills known placeholders and keeps the others', () => {
  assert.equal(fmt('{a}/{b}', { a: 1 }), '1/{b}');
});

test('t and tr: an unknown key gives the key itself or the fallback', () => {
  setMessages({ known: 'yes' }, 'en');
  assert.equal(t('missing.key'), 'missing.key');
  assert.equal(tr('missing.key', 'Fallback'), 'Fallback');
  assert.equal(tr('known', 'Fallback'), 'yes');
});

test('tn uses _one only for exactly 1 and formats the number in the locale', () => {
  setMessages({ 'img_one': '{n} imagem', 'img_other': '{n} imagens' }, 'pt');
  assert.equal(tn('img', 1), '1 imagem');
  assert.equal(tn('img', 0), '0 imagens');
  assert.equal(tn('img', 1500), '1.500 imagens');
});

test('fmtBytes, fmtDate and lower follow the locale', () => {
  setMessages({}, 'pt');
  assert.equal(fmtBytes(1536), '1,5 KB');
  assert.equal(fmtDate('not a date'), '');
  assert.notEqual(fmtDate('2026-01-02T00:00:00'), '');
  assert.equal(lower('ÁRVORE'), 'árvore');
});
