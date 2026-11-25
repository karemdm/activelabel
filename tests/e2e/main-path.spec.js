'use strict';
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { test, expect } = require('@playwright/test');
const { writePng } = require('./png');

// Raw translation keys that show on screen when a text is missing (for example "step.curation", "error.x").
const RAW_KEY = /\b(?:step|error|project|nav|common|task)\.[a-z_]+/;

async function expectNoRawKeys(page) {
  const text = await page.evaluate(() => document.body.innerText);
  expect(text).not.toMatch(RAW_KEY);
}

async function expectNoHorizontalScroll(page) {
  const widths = await page.evaluate(() => ({
    window: window.innerWidth,
    page: document.documentElement.scrollWidth,
    body: document.body.scrollWidth,
  }));
  expect(widths.page, `page is wider than the window: ${JSON.stringify(widths)}`).toBeLessThanOrEqual(widths.window);
  expect(widths.body, `body is wider than the window: ${JSON.stringify(widths)}`).toBeLessThanOrEqual(widths.window);
}

test.describe.serial('main path', () => {
  const projectName = `Smoke ${Date.now()}`;
  let projectUrl = '';
  let images = [];

  test.beforeAll(() => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'activelabel-e2e-images-'));
    images = [1, 2, 3].map((n) => writePng(path.join(dir, `img${n}.png`), n));
  });

  test('home page opens in Portuguese without raw keys', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByRole('heading', { level: 1, name: 'Projetos', exact: true })).toBeVisible();
    await expectNoRawKeys(page);
  });

  test('create a classification project', async ({ page }) => {
    await page.goto('/projects/new');
    await page.locator('#f-name').fill(projectName);
    await page.locator('label[for="task-classification"]').click();
    await expect(page.locator('#task-classification')).toBeChecked();
    await page.locator('#create-btn').click();
    await page.waitForURL(/\/projects\/\d+/);
    projectUrl = new URL(page.url()).pathname;
    await expect(page.getByRole('heading', { name: projectName })).toBeVisible();
  });

  test('upload three generated PNGs', async ({ page, request }) => {
    await page.goto(projectUrl);
    await page.locator('#file-input').setInputFiles(images);
    const id = projectUrl.split('/').pop();
    await expect
      .poll(async () => (await (await request.get(`/api/projects/${id}`)).json()).counts.active, {
        message: 'the project should end up with 3 active images',
        timeout: 30_000,
      })
      .toBe(3);
  });

  test('curation step opens, in Portuguese and in English, without raw keys', async ({ page }) => {
    await page.goto(projectUrl);
    const step = page.locator('.step[data-step="curation"]');
    await step.click();
    await expect(step).toHaveAttribute('aria-current', 'step');
    await expect(step).toContainText('Curadoria');
    await expectNoRawKeys(page);

    await page.locator('.lang-switch a[data-lang="en"]').click();
    await page.waitForLoadState('load');
    await expect(page.locator('html')).toHaveAttribute('lang', 'en');
    const stepEn = page.locator('.step[data-step="curation"]');
    await stepEn.click();
    await expect(stepEn).toContainText('Curation');
    await expectNoRawKeys(page);

    await page.locator('.lang-switch a[data-lang="pt"]').click();
    await page.waitForLoadState('load');
    await expect(page.locator('html')).toHaveAttribute('lang', 'pt');
  });

  test('no horizontal scroll at 390 px on the home page, the new-project form and the project', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto('/');
    await expect(page.getByRole('heading', { level: 1, name: 'Projetos', exact: true })).toBeVisible();
    await expectNoHorizontalScroll(page);

    await page.goto('/projects/new');
    await expect(page.locator('#create-btn')).toBeVisible();
    await expectNoHorizontalScroll(page);

    await page.goto(projectUrl);
    await expect(page.getByRole('heading', { name: projectName })).toBeVisible();
    await page.locator('.step[data-step="curation"]').click();
    await expect(page.locator('.step[data-step="curation"]')).toHaveAttribute('aria-current', 'step');
    await expectNoHorizontalScroll(page);
    await expectNoRawKeys(page);
  });
});
