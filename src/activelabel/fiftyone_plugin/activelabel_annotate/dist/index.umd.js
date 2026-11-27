/*
 * ActiveLabel plugin for the FiftyOne App: open the image window in Annotate mode.
 *
 * In Explore mode (the App's default) the labels are only shown. In Annotate mode a click selects a box
 * or a polygon, and dragging it, its corners or its points changes it. ActiveLabel marks the datasets the
 * App can edit with ``activelabel_editable`` in the dataset info; for those this operator runs when the
 * dataset opens and switches the image window to Annotate mode. The person can still go back to Explore.
 *
 * The App runs the ``onDatasetOpen`` operators it knows when a dataset is opened; the first dataset is often
 * opened before this script is loaded, so the script also runs the operator once itself.
 *
 * Plain script, no build step: it uses the modules the App puts on ``window``.
 */
(function () {
  var operators = window.__foo__;
  var state = window.__fos__;
  var recoil = window.recoil;
  if (!operators || !state || !recoil || !state.useModalModeController || !operators.registerOperator) {
    return; // another version of the App: it keeps its own default
  }

  class OpenInAnnotate extends operators.Operator {
    get config() {
      return new operators.OperatorConfig({
        name: "open_in_annotate",
        label: "ActiveLabel: open images ready to edit",
        unlisted: true,
        onDatasetOpen: true,
      });
    }

    useHooks() {
      return {
        dataset: recoil.useRecoilValue(state.dataset),
        mode: state.useModalModeController(),
      };
    }

    async execute(ctx) {
      var info = (ctx.hooks.dataset && ctx.hooks.dataset.info) || {};
      if (info.activelabel_editable === true) {
        ctx.hooks.mode.activateAnnotateMode();
      }
    }
  }

  operators.registerOperator(OpenInAnnotate, "@activelabel/annotate");
  operators.executeOperator("@activelabel/annotate/open_in_annotate");
})();
