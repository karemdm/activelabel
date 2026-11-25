import json
import re
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / ".devcontainer" / "devcontainer.json"


def _load() -> dict:
    """Read devcontainer.json, which allows ``//`` comment lines.

    Returns:
        The parsed configuration.
    """
    text = "\n".join(
        line for line in CONFIG.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("//")
    )
    return json.loads(text)


def test_devcontainer_is_valid_and_pins_the_image():
    config = _load()
    assert config["image"].count(":") == 1 and not config["image"].endswith(":latest")


def test_forwarded_port_matches_the_server_command():
    config = _load()
    command = config["postStartCommand"]
    port = int(re.search(r"--port (\d+)", command).group(1))
    assert "--host 0.0.0.0" in command
    assert config["forwardPorts"][0] == port  # the app first; 5161 is the FiftyOne app
    assert config["portsAttributes"][str(port)]["label"] == "ActiveLabel"


def test_workspace_is_set_and_no_secrets():
    config = _load()
    assert config["containerEnv"]["ACTIVELABEL_WORKSPACE"].startswith("/workspaces/")
    assert not re.search(r"token|secret|password|key", json.dumps(config), re.I)
