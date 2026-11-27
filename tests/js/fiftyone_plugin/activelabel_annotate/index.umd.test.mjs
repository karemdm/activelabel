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
  const calls = { registered: [], executed: [], annotate: 0, warnings: [] };
  const window = {
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
  const operator = new calls.registered[0].cls();
  await operator.execute({ hooks: operator.useHooks() });
}

test("registers an unlisted operator that runs when a dataset opens, and runs it once at load", () => {
  const calls = app({ activelabel_editable: true });
  assert.equal(calls.registered.length, 1);
  assert.equal(calls.registered[0].plugin, "@activelabel/annotate");
  const config = new calls.registered[0].cls().config;
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
