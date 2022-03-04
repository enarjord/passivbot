from time import sleep
import yaml
from constants import INSTANCE_SIGNATURE_BASE, MANAGER_CONFIG_PATH
from instance import Instance, instances_from_config
from pm import ProcessManager


class Manager:
    def __init__(self):
        self.defaults = {}
        self.instances = []
        self.sync_config()

    def sync_config(self):
        '''Sync manger with config file'''
        self.instances = []
        with open(MANAGER_CONFIG_PATH, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.defaults = config['defaults']
        for instance in config['instances']:
            self.instances.extend(instances_from_config(
                instance, self.defaults))

    def start_all(self):
        for instance in self.instances:
            self.start(instance)

    def start(self, instance):
        instance.start()

    def stop_all(self):
        stopped_instances = []
        for instance in self.get_running_instances():
            stopped = self.stop(instance)
            if stopped:
                stopped_instances.append(instance.get_id())

        return stopped_instances

    def stop(self, instance: Instance) -> bool:
        return self.stop_attempt(instance, retries=3)

    def stop_attempt(self, instance: Instance, retries: int = 1):
        '''
        Attempt to stop instance several times
        :param instance: Instance to stop
        :param retries: Number of retries
        :return: True if instance was stopped, False otherwise
        '''
        while retries > 0:
            if instance.stop():
                return True
            retries -= 1
            sleep(1)

        return False

    def get_instances(self):
        return self.instances

    def get_instance_by_id(self, instance_id):
        for instance in self.instances:
            if instance.get_id() == instance_id:
                return instance

        return None

    def get_running_instances(self):
        running_instances = []
        for instance in self.instances:
            if instance.is_running():
                running_instances.append(instance)

        return running_instances

    def get_all_passivbot_instances(self):
        '''Get all passivbot instances running on this machine'''
        pm = ProcessManager()
        signature = '^{}'.format(' '.join(INSTANCE_SIGNATURE_BASE))
        pids = pm.get_pid(signature, all_matches=True)
        if len(pids) == 0:
            return []

        instances_cmds = [pm.info(pid) for pid in pids]
        instanaces = []
        for cmd in instances_cmds:
            args = cmd.split(' ')
            if len(args) > 3:
                args = args[3:]
            else:
                continue

            user = args[0]
            symbol = args[1]
            live_config_path = args[2]
            cfg = self.defaults.copy()
            cfg['user'] = user
            cfg['symbol'] = symbol
            cfg['live_config_path'] = live_config_path
            instance = Instance(cfg)
            if instance.is_running():
                instanaces.append(instance)

        return instanaces
