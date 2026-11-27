// The FiftyOne App plugin that opens the image window in Annotate mode (plain script, run in a sandbox).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";

const SCRIPT = readFileSync(
  new URL(
    "../../../../src/activelabel/fiftyone_plugin/activelabel_annotate/dist/index.umd.js",
    import.meta.url,
  ),
  "utf8",
);

class Operator {}
class OperatorConfig {
  constructor(options) {
    Object.assign(this, options);
  }
}

function app(info, controller) {
  const calls = { registered: [], executed: [], annotate: 0, warnings: [], reloaded: 0 };
  const window = {
    location: { reload: () => (calls.reloaded += 1) },
    console: { warn: (...args) => calls.warnings.push(args) },
    __foo__: {
      Operator,
      OperatorConfig,
      registerOperator: (cls, plugin) => calls.registered.push({ cls, plugin }),
      executeOperator: (uri) => calls.executed.push(uri),
    },
    __fos__: {
      dataset: "dataset-atom",
      useModalModeController:
        controller ?? (() => ({ activateAnnotateMode: () => (calls.annotate += 1) })),
    },
    recoil: { useRecoilValue: (atom) => (atom === "dataset-atom" ? { info } : null) },
  };
  vm.runInNewContext(SCRIPT, { window });
  return calls;
}

async function run(calls) {
  const operator = new calls.registered[1].cls();
  await operator.execute({ hooks: operator.useHooks() });
}

test("registers an unlisted operator that runs when a dataset opens, and runs it once at load", () => {
  const calls = app({ activelabel_editable: true });
  assert.equal(calls.registered.length, 2);
  assert.ok(calls.registered.every((r) => r.plugin === "@activelabel/annotate"));
  const config = new calls.registered[1].cls().config;
  assert.equal(config.name, "open_in_annotate");
  assert.equal(config.unlisted, true);
  assert.equal(config.onDatasetOpen, true);
  assert.deepEqual(calls.executed, ["@activelabel/annotate/open_in_annotate"]);
});

test("an editable dataset opens its images in Annotate mode", async () => {
  const calls = app({ activelabel_editable: true });
  await run(calls);
  assert.equal(calls.annotate, 1);
});

test("a dataset the App cannot edit (keypoints) or not made by ActiveLabel stays in Explore mode", async () => {
  for (const info of [{ activelabel_editable: false }, {}, null]) {
    const calls = app(info);
    await run(calls);
    assert.equal(calls.annotate, 0);
  }
});

test("another version of the App, without the modules it needs, is left alone with a warning", () => {
  const warnings = [];
  const window = { __foo__: undefined, __fos__: {}, recoil: {}, console: { warn: (m) => warnings.push(m) } };
  assert.doesNotThrow(() => vm.runInNewContext(SCRIPT, { window }));
  assert.equal(warnings.length, 1);
  assert.match(warnings[0], /click the Annotate tab/);
});

test("an App whose mode controller changed (no activateAnnotateMode) warns and stays usable", async () => {
  const calls = app({ activelabel_editable: true }, () => ({}));
  await assert.doesNotReject(run(calls));
  assert.equal(calls.annotate, 0);
  assert.equal(calls.warnings.length, 1);
});

test("an App whose mode switch fails warns instead of breaking", async () => {
  const failing = () => ({
    activateAnnotateMode: () => {
      throw new Error("changed");
    },
  });
  const calls = app({ activelabel_editable: true }, failing);
  await assert.doesNotReject(run(calls));
  assert.equal(calls.warnings.length, 1);
});

/** A page of the App with the image window open: the label list or the edit form, and the select tool. */
function page({ list = false, select = true } = {}) {
  const listeners = { keydown: [], click: [] };
  const clicks = [];
  const timers = [];
  const state = { list };
  const document = {
    addEventListener: (type, fn, capture) => {
      assert.equal(capture, true);
      listeners[type].push(fn);
    },
    querySelector: (selector) => {
      if (selector.includes('sidebar-group-Labels-field-count')) return state.list ? {} : null;
      if (selector.includes('select-action')) return select ? { click: () => clicks.push('select') } : null;
      return null;
    },
  };
  const window = {
    document,
    console: { warn: () => {} },
    setTimeout: (fn) => timers.push(fn),
    __foo__: undefined,
    __fos__: {},
    recoil: {},
  };
  vm.runInNewContext(SCRIPT, { window });
  const fire = (type, event) => listeners[type].forEach((fn) => fn(event));
  const settle = () => timers.splice(0).forEach((fn) => fn());
  return { state, clicks, fire, settle };
}

test("after Delete on an open label the select tool is picked again", () => {
  const app = page({ list: false });
  app.fire("keydown", { key: "Delete", target: { tagName: "CANVAS" } });
  app.state.list = true; // the label is gone: the list shows again
  app.settle();
  assert.ok(app.clicks.length >= 1);
});

test("after Delete label in the menu the select tool is picked again", () => {
  const app = page({ list: false });
  const item = { closest: (sel) => (sel.includes("label-menu-delete") ? {} : null) };
  app.fire("click", { target: item });
  app.state.list = true;
  app.settle();
  assert.ok(app.clicks.length >= 1);
});

test("Backspace while drawing a polygon (the form stays open) keeps the drawing tool", () => {
  const app = page({ list: false });
  app.fire("keydown", { key: "Backspace", target: { tagName: "CANVAS" } });
  app.settle(); // the form is still shown: a point was removed, not a label
  assert.deepEqual(app.clicks, []);
});

test("Delete with no label open, or typed in a text field, changes nothing", () => {
  const listShown = page({ list: true });
  listShown.fire("keydown", { key: "Delete", target: { tagName: "CANVAS" } });
  listShown.settle();
  assert.deepEqual(listShown.clicks, []);
  const typing = page({ list: false });
  typing.fire("keydown", { key: "Backspace", target: { tagName: "INPUT" } });
  typing.state.list = true;
  typing.settle();
  assert.deepEqual(typing.clicks, []);
});

test("other keys and clicks are ignored", () => {
  const app = page({ list: false });
  app.fire("keydown", { key: "a", target: { tagName: "CANVAS" } });
  app.fire("click", { target: { closest: () => null } });
  app.state.list = true;
  app.settle();
  assert.deepEqual(app.clicks, []);
});

test("reload_page reloads the page, so the App reads images changed by the Python operators", async () => {
  const calls = app({});
  const reload = calls.registered.find((r) => new r.cls().config.name === "reload_page");
  assert.ok(reload && new reload.cls().config.unlisted);
  await new reload.cls().execute({});
  assert.equal(calls.reloaded, 1);
});
