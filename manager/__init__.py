from manager.constants import INSTANCE_SIGNATURE_BASE
from typing import List, Dict, Union, Callable, Tuple
from manager.config.parser import ConfigParser
from manager.instance import Instance
from manager.pm import ProcessManager
from itertools import groupby


class Manager:
    def __init__(self, config_path: str = None):
        self.instances = None
        self.config_path = config_path
        self.config_parser = ConfigParser(self.config_path)

    def load_instances(self):
        """Sync manger with instances in the config and unsynced ones"""
        self.instances = self.config_parser.get_instances()

        for instance in self.find_unsynced_instances():
            iid = instance.get_id()
            if self.instances.get(iid) is None:
                instance.is_in_config(False)
                self.instances[iid] = instance

    def get_instances(self) -> List[Instance]:
        if self.instances is None:
            self.load_instances()

        return self.instances.values()

    def filter_instances(self, filter) -> List[Instance]:
        if not callable(filter):
            return []

        instances = []
        for instance in self.get_instances():
            if filter(instance):
                instances.append(instance)

        return instances

    def get_running_instances(self) -> List[Instance]:
        return self.filter_instances(lambda i: i.is_running())

    def count(self, filter_fn: Callable, instances: List[Instance] = None) -> Tuple[int, int]:
        iterable = instances
        if iterable is None:
            iterable = self.get_instances()

        return len(list(filter(filter_fn, iterable))), len(iterable)

    def count_running(self, instances: List[Instance] = None, format=False) -> Union[int, str]:
        def filter(i): return i.is_running()
        running, total = self.count(filter, instances)

        if not format:
            return running

        return f"{running}/{total} running"

    def count_unsynced(self, instances: List[Instance] = None, format: bool = False) -> Union[int, str]:
        def filter(i): return not i.is_in_config()
        unsynced, total = self.count(filter, instances)

        if not format:
            return unsynced

        return f"{unsynced}/{total} unsynced"

    def get_synced_instances(self) -> List[Instance]:
        return self.filter_instances(lambda i: i.is_in_config())

    def get_unsynced_instances(self) -> List[Instance]:
        return self.filter_instances(lambda i: not i.is_in_config())

    def query_instances(self, query: List[str]) -> List[Instance]:
        """Query instances by query string"""
        instances = []
        for instance in self.get_instances():
            if instance.match(query):
                instances.append(instance)

        return instances

    def find_unsynced_instances(self) -> List[Instance]:
        """Get all passivbot instances running on this machine"""
        signature = f"^{' '.join(INSTANCE_SIGNATURE_BASE)}"
        pids = ProcessManager.get_pid(signature, all_matches=True)
        if len(pids) == 0:
            return []

        instances_cmds = [ProcessManager.info(pid) for pid in pids]
        instanaces = []
        for cmd in instances_cmds:
            args = cmd.split(" ")
            if len(args) <= 3:
                continue

            args = args[3:]
            user = args[0]
            symbol = args[1]
            config = args[2]
            flags = {}

            if len(args[3:]) > 0:
                it = iter(args[3:])
                flags = dict(zip(it, it))

            instance = Instance({
                "user": user,
                "symbol": symbol,
                "config": config,
                "flags": flags
            })
            if instance.is_running():
                instanaces.append(instance)

        return instanaces

    def group_instances_by_user(self, instances: List[Instance]) -> Dict[str, List[Instance]]:
        groups = {}
        for key, group in groupby(instances, lambda i: i.get_user()):
            groups[key] = list(group)
        return groups
