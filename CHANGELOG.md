# Changelog

Notable changes, newest first. The project has no tagged release yet.

## Unreleased

### Removed

- **Label Studio support.** Importing a project from Label Studio, its synchronisation, its edit sessions
  and reading its JSON export are gone. A workspace made before keeps its data: an old link, an
  imported-from record or a session of that tool is shown as belonging to a **discontinued tool**
  ("Linked to a discontinued tool", "Imported from a discontinued tool"); a link can only be **unlinked**
  and a session can only be **closed**, which changes nothing in that tool.
- **CVAT edit sessions.** Review in CVAT goes only through the synchronisation of the linked project
  (docs/SYNC.md); the *FiftyOne* tab keeps the edit sessions (docs/ANNOTATION.md). Old CVAT sessions can
  only be closed.

### Added

- Read-only audit of the tasks on the CVAT server and a guard that refuses to create a CVAT task outside a
  project (docs/SYNC.md).
- Stable error codes next to the English `detail` on API errors, limits on names, classes, job size,
  archives and video frames, and one FiftyOne dataset name per project (docs/DESIGN.md).
- Pre-annotation with instance and semantic segmentation models; data-split rules for left-out labels,
  empty parts and versions in use; saved models take the weights of their run (docs/PREANNOTATE.md,
  docs/DATASETS.md, docs/TRAINING.md).
