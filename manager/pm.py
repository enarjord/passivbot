import os
import sys
from time import sleep
from typing import List, Union
import subprocess
from constants import UNELEVATED_USER


class ProcessManager:
    def add(self, command: List[str]) -> int:
        '''
        Launch a new process with the given command.
        :param command: The command to run.
        :return: Error code of the add command.
        '''
        return os.system(' '.join(command))

    def add_nohup_process(self, command: List[str], log_file_path: str = None) -> int:
        '''
        Launch a new no hang up process with the given target script path.
        :param target_script_path: The path to the target script.
        :param log_file_path: The path to the log file.
        :return: Error code of the add command.

        man: https://linux.die.net/man/1/nohup
        '''
        nohuo_command = ['sudo', '-u', UNELEVATED_USER, 'nohup']
        nohuo_command.extend(['sh', '-c', '"{}"'.format(' '.join(command))])
        if log_file_path is not None:
            nohuo_command.extend(['>', log_file_path, '2>&1', '&'])
        else:
            nohuo_command.extend(['>', '/dev/null', '2>&1', '&'])
        return self.add(nohuo_command)

    def get_pid(self, signature: str, all_matches: bool = False, retries: int = 5) -> List[int]:
        '''
        Use pgrep to get the process id of the process with the given query string.
        :param signature: The signature to search for.
        :return: The process id of the process with the given query string.

        man: https://man7.org/linux/man-pages/man1/pgrep.1.html
        '''
        for i in range(retries):
            try:
                cmd = ['pgrep', '-U', UNELEVATED_USER, '-f', signature]
                pids = subprocess.check_output(cmd).decode('utf-8').strip()
                if len(pids) > 0:
                    break
                else:
                    sleep(0.2)
            except subprocess.CalledProcessError:
                if i == retries - 1:
                    if all_matches:
                        return []
                    return None
                else:
                    continue

        matches = pids.split('\n')
        if all_matches:
            return [int(pid) for pid in matches]
        else:
            return int(matches[0])

    def is_running(self, signature: str) -> bool:
        '''
        Check if the process with the given signature is running.
        :param signature: The signature to check.
        :return: True if the process with the given signature is running.
        '''
        return self.get_pid(signature) is not None

    def info(self, pid: int) -> str:
        '''
        Get the info of the process with the given pid.
        :param pid: The process id of the process to get the info of.
        :return: The info of the process with the given pid.
        '''
        cmd = ['sudo', '-u', UNELEVATED_USER,
               'ps', '-p', str(pid), '-o', 'args=']
        try:
            return subprocess.check_output(cmd).decode('utf-8').strip()
        except subprocess.CalledProcessError:
            return None

    def kill(self, pid: int, force: bool = False):
        '''
        Kill the process with the given pid.
        :param pid: The process id of the process to kill.
        :param force: If True, kill the process with the given pid with SIGKILL.
        :return: The error code of the kill command.
        '''
        if force:
            cmd_arr = ['sudo', '-u', UNELEVATED_USER, 'kill', '-9', str(pid)]
        else:
            cmd_arr = ['sudo', '-u', UNELEVATED_USER, 'kill', str(pid)]

        cmd = ' '.join(cmd_arr)
        os.system(cmd)
        retries = 10
        while retries > 0:
            if not self.info(pid):
                break
            sleep(0.2)
            retries -= 1
