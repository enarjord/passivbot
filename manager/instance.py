import os
import sys
from typing import Dict, List
from constants import CONFIGS_PATH, INSTANCE_SIGNATURE_BASE, PASSIVBOT_PATH
from pm import ProcessManager


class Instance:
    def __init__(self, config):
        self.prooc_id = None
        self.user = str(config['user'])
        self.symbol = str(config['symbol'])

        live_config_path = config['live_config_path']
        if type(live_config_path) == str and len(live_config_path) > 0:
            self.live_config_path = live_config_path
        else:
            self.live_config_path = os.path.join(
                CONFIGS_PATH, config['live_config_name'])

        if not os.path.exists(self.live_config_path):
            self.say('Config file does not exist: {}'.format(
                self.live_config_path))
            sys.exit(1)

        self.market_type = str(config['market_type']) or 'futures'
        self.lw = float(config['long_wallet_exposure_limit']) or 0.0
        self.sw = float(config['short_wallet_exposure_limit']) or 0.0
        self.ab = float(config['assigned_balance']) or 0
        self.lm = str(config['long_mode']) or 'n'
        self.sm = str(config['short_mode']) or 'm'

    def get_id(self):
        return '{}:{}'.format(self.user, self.symbol)

    def say(self, message):
        print('{}: {}'.format(self.get_id(), message))

    def get_args(self):
        return [self.user, self.symbol, self.live_config_path]

    def get_flags(self):
        flags = {
            '-m': {
                'value': self.market_type,
                'valid': self.market_type != 'futures'
            },
            '-lw': {
                'value': self.lw,
                'valid': self.lw > 0.0
            },
            '-sw': {
                'value': self.sw,
                'valid': self.sw > 0.0
            },
            '-ab': {
                'value': self.ab,
                'valid': self.ab > 0.0
            },
            '-lm': {
                'value': self.lm,
                'valid': self.lm != 'n'
            },
            '-sm': {
                'value': self.sm,
                'valid': self.sm != 'm'
            }
        }

        valid_flags = []
        for k, v in flags.items():
            if v['valid'] is True:
                valid_flags.append(k)
                valid_flags.append(str(v['value']))

        return valid_flags

    def get_pid_signature(self):
        signature = INSTANCE_SIGNATURE_BASE.copy()
        signature.extend([self.user, self.symbol])
        return '^{}'.format(' '.join(signature))

    def get_cmd(self):
        cmd = INSTANCE_SIGNATURE_BASE.copy()
        cmd.extend(self.get_args())
        cmd.extend(self.get_flags())
        return cmd

    def start(self, silent=False) -> bool:
        log_file = os.path.join(PASSIVBOT_PATH, 'logs/{}.log'.format(self.get_id()))
        cmd = self.get_cmd()

        if silent is True:
            log_file = '/dev/null'

        pm = ProcessManager()
        pm.add_nohup_process(cmd, log_file)
        self.proc_id = pm.get_pid(self.get_pid_signature(), retries=10)
        if self.proc_id is None:
            self.say('Failed to get process id. See {} for more info.'
                     .format(log_file))
            return False

        return True

    def stop(self, force=False) -> bool:
        if not self.is_running():
            return False

        pm = ProcessManager()
        pid = pm.get_pid(self.get_pid_signature())
        if pid is None:
            return False

        pm.kill(pid, force)
        return True

    def restart(self, force=False, silent=False) -> bool:
        if self.is_running():
            stopped = self.stop(force)
            if not stopped:
                return False

        return self.start(silent)

    def is_running(self):
        pm = ProcessManager()
        return pm.is_running(self.get_pid_signature())


def instances_from_config(config: Dict, defaults: Dict) -> List[Instance]:
    instances = []
    for symbol in config['symbols']:
        cfg = defaults.copy()
        cfg['symbol'] = symbol
        cfg['user'] = config['user']
        for k, v in config.items():
            if k in cfg:
                cfg[k] = v
        instances.append(Instance(cfg))

    return instances
