from activelabel import constants, jobs
from activelabel.images import manifest


def test_job_states_are_the_words_the_api_uses():
    assert (constants.QUEUED, constants.RUNNING, constants.DONE, constants.FAILED, constants.CANCELLED) == (
        "queued",
        "running",
        "done",
        "failed",
        "cancelled",
    )
    assert constants.FINISHED == (constants.DONE, constants.FAILED, constants.CANCELLED)


def test_jobs_reexports_the_job_states():
    assert jobs.QUEUED is constants.QUEUED
    assert jobs.RUNNING is constants.RUNNING
    assert jobs.FINISHED is constants.FINISHED


def test_manifest_reexports_the_image_states():
    assert manifest.ACTIVE == constants.ACTIVE == "active"
    assert manifest.EXCLUDED == constants.EXCLUDED == "excluded"
