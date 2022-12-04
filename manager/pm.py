from typing import List, Union
from constants import USER
from time import sleep
import subprocess
import os


class ProcessManager:
    @staticmethod
    def add(command: List[str]) -> int:
        """
        Launch a new process with the given command.
        :param command: The command to run.
        :return: Error code of the add command.
        """
        return os.system(" ".join(command))

    @staticmethod
    def add_nohup_process(command: List[str], log_file_path: str = None) -> int:
        """
        Launch a new no hang up process with the given target script path.
        :param target_script_path: The path to the target script.
        :param log_file_path: The path to the log file.
        :return: Error code of the add command.

        man: https://linux.die.net/man/1/nohup
        """
        nohup_command = ["nohup"]
        nohup_command.extend(command)
        if log_file_path is not None:
            nohup_command.extend([">", log_file_path, "2>&1", "&"])
        else:
            nohup_command.extend([">", "/dev/null", "2>&1", "&"])
        return ProcessManager.add(nohup_command)

    @staticmethod
    def get_pid(signature: str, all_matches: bool = False) -> Union[int, None, List[int]]:
        """
        Use pgrep to get the process id of the process with the given query string.
        :param signature: The signature to search for.
        :return: The process id of the process with the given query string.

        man: https://man7.org/linux/man-pages/man1/pgrep.1.html
        """
        try:
            cmd = ["pgrep", "-U", USER, "-f", signature]
            pids = subprocess.check_output(cmd).decode("utf-8").strip()
        except subprocess.CalledProcessError:
            if all_matches:
                return []

            return None

        matches = pids.split("\n")
        if len(matches) == 0:
            if all_matches:
                return []

            return None

        if all_matches:
            return [int(pid) for pid in matches]
        else:
            return int(matches[0])

    @staticmethod
    def wait_pid_start(signature: str, retries: int = 5, cooldown: float = 0.5) -> Union[int, None]:
        for i in range(0, retries):
            pid = ProcessManager.get_pid(signature)
            if pid is not None:
                return pid

            sleep(cooldown)

        return None

    @staticmethod
    def is_running(signature: str) -> bool:
        """
        Check if the process with the given signature is running.
        :param signature: The signature to check.
        :return: True if the process with the given signature is running.
        """
        return ProcessManager.get_pid(signature) is not None

    @staticmethod
    def info(pid: int) -> str:
        """
        Get the info of the process with the given pid.
        :param pid: The process id of the process to get the info of.
        :return: The info of the process with the given pid.
        """
        cmd = ["ps", "-p", str(pid), "-o", "args="]
        try:
            return subprocess.check_output(cmd).decode("utf-8").strip()
        except subprocess.CalledProcessError:
            return None

    @staticmethod
    def kill(pid: int, force: bool = False, max_retries: int = -1) -> bool:
        """
        Kill the process with the given pid.
        :param pid: The process id of the process to kill.
        :param force: If True, kill the process with the given pid with SIGKILL.
        :return: The error code of the kill command.
        """
        cmd = ["kill"]
        if force:
            cmd.append("-9")

        cmd.append(str(pid))
        os.system(" ".join(cmd))
        while ProcessManager.info(pid) is not None:
            sleep(0.15)
            max_retries -= 1
            if max_retries == 0:
                return False

        return True
