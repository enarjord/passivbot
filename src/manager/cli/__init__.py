from sys import path, argv
path.append(".")

from .cli import CLI
from .commands.init import Init
from .commands.help import Help
from .commands.list import List
from .commands.info import Info
from .commands.sync import Sync
from .commands.stop import Stop
from .commands.start import Start
from .commands.restart import Restart


class ManagerCLI(CLI):
    def __init__(self) -> None:
        super().__init__()
        self.add_command("start", Start)
        self.add_command("stop", Stop)
        self.add_command("restart", Restart)
        self.add_command("sync", Sync)
        self.add_command("list", List)
        self.add_command("info", Info)
        self.add_command("init", Init)
        self.add_command("help", Help)

        self.add_flag("all", ["-a", "--all"],
                      "perform action on all instances")

        self.add_flag("unsynced", ["-u", "--unsynced"],
                      "perform action on unsynced instances")

        self.add_flag("yes", ["-y", "--yes"], "skip confirmation")

        self.add_flag("silent", ["-s", "--silent"],
                      "disiable logging for affected instances")

        self.add_flag("force", ["-f", "--force"], "force an action")

        self.add_flag("modifiers", ["-m", "--modify"],
                      "modify flags of affected instances", type=str)

        self.add_flag("config_path", ["-c", "--config"],
                      "specify an absolute path to a manager config", type=str)

        self.add_flag("help", ["-h", "--help"], "show help for a command")


if __name__ == "__main__":
    ManagerCLI().run(argv[1:])
