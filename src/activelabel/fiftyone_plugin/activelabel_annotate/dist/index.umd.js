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
 * Plain script, no build step: it uses the modules the App puts on ``window``. Verified with FiftyOne
 * 1.22.1 (``fiftyone_plugin.VERIFIED_FIFTYONE``). ``useModalModeController`` is not a public API: when a
 * later App lacks it or it fails, the plugin only writes a warning in the console and the App works as
 * usual (images open in Explore).
 *
 * After a label is deleted the App keeps its drawing tool on, so the next click on the image draws a new box
 * instead of selecting one. ``watchDeletions`` picks the select tool (the arrow) again after a deletion. It
 * only reads the page (the ``data-cy`` names of the App's elements); when they are missing it does nothing.
 */

(function () {
  /**
   * Pick the App's select tool again after a label is deleted.
   *
   * A deletion is the *Delete* or *Backspace* key, or *Delete label* in the label's menu, while a label is
   * open in the edit form (the list of labels is hidden); it is done when the list shows again. Backspace
   * while drawing a polygon removes its last point and keeps the form open, so it is not taken for a
   * deletion. A key typed in a text field is left alone.
   *
   * @param {Window} win the App's window.
   */
  function watchDeletions(win) {
    var doc = win.document;
    if (!doc || typeof doc.addEventListener !== "function") return;
    var LIST = '[data-cy="sidebar-group-Labels-field-count"]';
    var SELECT = '[data-cy="select-action"]';
    var listShown = function () {
      return doc.querySelector(LIST) !== null;
    };
    var formShown = function () {
      return doc.querySelector(SELECT) !== null && !listShown();
    };
    var selectWhenDone = function () {
      [250, 700].forEach(function (ms) {
        win.setTimeout(function () {
          var select = doc.querySelector(SELECT);
          if (select && listShown()) select.click();
        }, ms);
      });
    };
    doc.addEventListener(
      "keydown",
      function (event) {
        if (event.key !== "Delete" && event.key !== "Backspace") return;
        var target = event.target || {};
        if (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName || "")) return;
        if (formShown()) selectWhenDone();
      },
      true,
    );
    doc.addEventListener(
      "click",
      function (event) {
        var target = event.target;
        if (target && target.closest && target.closest('[data-cy="label-menu-delete"]')) selectWhenDone();
      },
      true,
    );
  }

  try {
    watchDeletions(window);
  } catch (error) {
    window.console.warn("ActiveLabel: could not watch deletions in the App.", error);
  }
  var operators = window.__foo__;
  var state = window.__fos__;
  var recoil = window.recoil;
  var WARNING = "ActiveLabel: this FiftyOne App version cannot open images in Annotate mode; click the Annotate tab to edit.";
  if (!operators || !state || !recoil || !state.useModalModeController || !operators.registerOperator) {
    // another version of the App: it keeps its own default (Explore), everything else works
    window.console.warn(WARNING);
    return;
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
      if (info.activelabel_editable !== true) return;
      var mode = ctx.hooks.mode;
      if (!mode || typeof mode.activateAnnotateMode !== "function") {
        window.console.warn(WARNING);
        return;
      }
      try {
        mode.activateAnnotateMode();
      } catch (error) {
        window.console.warn(WARNING, error);
      }
    }
  }

  /**
   * Reload the page, so the App reads the images again after ActiveLabel's Python operators changed them.
   * The image window comes back (its id is in the address) in the mode it had.
   */
  class ReloadPage extends operators.Operator {
    get config() {
      return new operators.OperatorConfig({ name: "reload_page", label: "ActiveLabel: reload", unlisted: true });
    }

    async execute() {
      window.location.reload();
    }
  }

  try {
    operators.registerOperator(ReloadPage, "@activelabel/annotate");
    operators.registerOperator(OpenInAnnotate, "@activelabel/annotate");
    operators.executeOperator("@activelabel/annotate/open_in_annotate");
  } catch (error) {
    window.console.warn(WARNING, error);
  }
})();
