'use strict';
const os = require('node:os');
const path = require('node:path');
const { defineConfig, devices } = require('@playwright/test');

// A port and a workspace of the tests' own: never the default port (8090) and never ~/.activelabel.
const PORT = 8107;
const WORKSPACE = path.join(os.tmpdir(), 'activelabel-e2e-workspace');
const PYTHON = process.env.ACTIVELABEL_E2E_PYTHON || 'python';

module.exports = defineConfig({
  testDir: 'tests/e2e',
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  timeout: 60_000,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : [['list']],
  use: {
    // the app picks the language from the browser's Accept-Language before ACTIVELABEL_LANG, so ask for Portuguese
    locale: 'pt-BR',
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  // PW_CHANNEL=chrome (CI: the runner's own Google Chrome) or msedge (Windows) avoids downloading a browser.
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'], channel: process.env.PW_CHANNEL || undefined } }],
  webServer: {
    // isolated_server.py gives the server its own mongod and FiftyOne folders (never ~/.fiftyone or a shared service).
    command: `${PYTHON} tests/helpers/isolated_server.py --port ${PORT} --workspace "${WORKSPACE}" --lang pt`,
    url: `http://127.0.0.1:${PORT}/`,
    reuseExistingServer: false,
    timeout: 60_000,
    stdout: 'pipe',
    stderr: 'pipe',
  },
});
