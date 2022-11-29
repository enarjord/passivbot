from constants import INSTANCE_SIGNATURE_BASE, PASSIVBOT_PATH
from typing import Dict, List, Any
from pm import ProcessManager
import os


class Instance:
    def __init__(self, config: Dict):
        self.user = config.get("user")
        self.symbol = config.get("symbol")
        self.config = config.get("config")

        self.flags = config.get("flags", {})

        self.is_in_config_ = bool(config.get("is_in_config", True))
        self.is_running_ = None
        self.pid_ = None

    def get_args(self) -> List[str]:
        return [self.user, self.symbol, self.config]

    def get_flags(self) -> List[str]:
        flags = []
        for k, v in self.flags.items():
            if v is not None:
                flags.extend([k, str(v)])

        return flags

    def get_id(self) -> str:
        return f"{self.get_user()}-{self.get_symbol()}"

    def get_symbol(self) -> str:
        return self.symbol

    def get_user(self) -> str:
        return self.user

    def get_config(self) -> str:
        return self.config

    def get_pid_signature(self) -> str:
        signature = INSTANCE_SIGNATURE_BASE.copy()
        signature.extend([self.user, self.symbol])
        return f"^{' '.join(signature)}"

    def get_pid(self) -> int:
        if self.pid_ is None:
            self.pid_ = ProcessManager.get_pid(self.get_pid_signature())

        return self.pid_

    def get_pid_str(self) -> str:
        pid = self.get_pid()
        return str(pid) if pid is not None else "-"

    def get_cmd(self) -> List[str]:
        cmd = INSTANCE_SIGNATURE_BASE.copy()
        cmd.extend(self.get_args())
        cmd.extend(self.get_flags())
        return cmd

    def get_status(self) -> str:
        return "running" if self.is_running() else "stopped"

    def is_running(self) -> bool:
        if self.is_running_ is None:
            self.is_running_ = ProcessManager.is_running(
                self.get_pid_signature())

        return self.is_running_

    def is_in_config(self, value=None) -> bool:
        if value is not None:
            self.is_in_config_ = bool(value)

        return self.is_in_config_

    def match(self, query: List[str], exact: bool = False) -> bool:
        parameters = {
            "id": self.get_id(),
            "pid": self.get_pid_str(),
            "symbol": self.get_symbol(),
            "user": self.get_user(),
            "status": self.get_status(),
        }

        if not exact:
            parameters = {k: v.lower() for k, v in parameters.items()}
            query = [q.lower() for q in query]

        matches = 0
        for condition in query:
            if "=" in condition:
                k, v = condition.split("=")
                if k in parameters and parameters[k].startswith(v):
                    matches += 1
                    continue

            if any(condition in v for v in parameters.values()):
                matches += 1

        return matches == len(query)

    def apply_flags(self, flags: Dict[str, Any]):
        if flags is None:
            return

        for key, value in flags.items():
            self.flags[key] = value

    # ---------------------------------------------------------------------------- #
    #                                 state methods                                #
    # ---------------------------------------------------------------------------- #

    def reset_state(self):
        self.is_running_ = None
        self.pid_ = None

    def start(self, silent: bool = False) -> bool:
        self.reset_state()

        log_file = os.path.join(
            PASSIVBOT_PATH, f"logs/{self.get_user()}/{self.get_symbol()}.log")

        try:
            if not os.path.exists(os.path.dirname(log_file)):
                os.makedirs(os.path.dirname(log_file))
        except:
            return False

        cmd = self.get_cmd()

        if silent is True:
            log_file = "/dev/null"

        ProcessManager.add_nohup_process(cmd, log_file)
        self.proc_id = ProcessManager.wait_pid_start(self.get_pid_signature())
        if self.proc_id is None:
            return False

        return True

    def stop(self, force=False) -> bool:
        self.reset_state()
        if not self.is_running():
            return False

        pid = ProcessManager.get_pid(self.get_pid_signature())
        if pid is None:
            return False

        ProcessManager.kill(pid, force)
        return True

    def restart(self, force=False, silent=False) -> bool:
        self.reset_state()
        if self.is_running():
            stopped = self.stop(force)
            if not stopped:
                return False

        return self.start(silent)
