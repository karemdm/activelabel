from pathlib import Path

from activelabel.config import Settings


def test_from_env(tmp_path):
    settings = Settings.from_env({"ACTIVELABEL_WORKSPACE": str(tmp_path), "ACTIVELABEL_LANG": "pt"})
    assert settings.workspace == tmp_path and settings.language == "pt"
    assert settings.projects_dir == tmp_path / "projects"
    assert settings.import_root is None
    assert (
        Settings.from_env({"ACTIVELABEL_IMPORT_ROOT": str(tmp_path)}).import_root == Path(tmp_path).resolve()
    )


def test_empty_workspace_means_not_set():
    default = Settings().workspace
    assert Settings.from_env({"ACTIVELABEL_WORKSPACE": ""}).workspace == default
    assert Settings.from_env({"ACTIVELABEL_WORKSPACE": "  "}).workspace == default
    assert Settings.from_env({"ACTIVELABEL_LANG": ""}).language == "en"


def test_allowed_hosts_are_a_comma_list():
    assert Settings.from_env({}).allowed_hosts == ()
    env = {"ACTIVELABEL_ALLOWED_HOSTS": " my-pc.local, ,Other.lan "}
    assert Settings.from_env(env).allowed_hosts == ("my-pc.local", "Other.lan")


def test_queue_limits_default_to_one_and_ignore_bad_values():
    assert (Settings().queue_cpu, Settings().queue_gpu) == (1, 1)
    assert (Settings.from_env({}).queue_cpu, Settings.from_env({}).queue_gpu) == (1, 1)
    good = Settings.from_env({"ACTIVELABEL_QUEUE_CPU": " 3 ", "ACTIVELABEL_QUEUE_GPU": "2"})
    assert (good.queue_cpu, good.queue_gpu) == (3, 2)
    for bad in ("", "  ", "abc", "0", "-4", "1.5"):
        settings = Settings.from_env({"ACTIVELABEL_QUEUE_CPU": bad, "ACTIVELABEL_QUEUE_GPU": bad})
        assert (settings.queue_cpu, settings.queue_gpu) == (1, 1), bad
