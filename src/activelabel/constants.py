"""Names shared by several packages: job states and image states.

One place for the words that the API, the files and the interface agree on, so a state is spelled once.
``activelabel.jobs`` and ``activelabel.images.manifest`` re-export these for the code that already imports
them from there.
"""

from __future__ import annotations

# State of a job (a background activity, a training run, a pre-annotation run)
QUEUED, RUNNING, DONE, FAILED, CANCELLED = "queued", "running", "done", "failed", "cancelled"
FINISHED = (DONE, FAILED, CANCELLED)

# State of an image in a project's manifest
ACTIVE = "active"
EXCLUDED = "excluded"
