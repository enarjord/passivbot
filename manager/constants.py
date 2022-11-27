from getpass import getuser
import logging
import sys
import os


logger = logging
logger.basicConfig(stream=sys.stdout, level=logging.INFO,
                   format="%(message)s")


def get_python_executable():
    if sys.version_info[0] == 3:
        return "python3"
    return None


PYTHON_EXC_ALIAS = get_python_executable()
if PYTHON_EXC_ALIAS is None:
    logging.error("Unsupported python version")
    sys.exit(1)


MANAGER_PATH = os.path.dirname(os.path.abspath(__file__))
MANAGER_CONFIG_PATH = os.path.join(MANAGER_PATH, "config.yaml")
MANAGER_CONFIG_SETTINGS_PATH = os.path.join(
    MANAGER_PATH, "config.settings.yaml")

PASSIVBOT_PATH = os.path.dirname(MANAGER_PATH)
USER = getuser()
if USER == "root":
    logging.error("Do not run this script as root")
    sys.exit(1)

# relative to passivbot.py
CONFIGS_PATH = os.path.join(PASSIVBOT_PATH, "configs/live")
SERVICES_PATH = "/etc/systemd/system"

INSTANCE_SIGNATURE_BASE = [PYTHON_EXC_ALIAS, "-u",
                           os.path.join(PASSIVBOT_PATH, "passivbot.py")]
