from constants import INSTANCE_SIGNATURE_BASE
from config_parser import ConfigParser
from instance import Instance
from pm import ProcessManager
from typing import List


class Manager:
    def __init__(self):
        self.instances = {}
        self.sync_instances()

    def sync_instances(self):
        """Sync manger with instances in the config and unsynced ones"""
        cp = ConfigParser()
        cp.get_config()
        self.instances = cp.get_instances()

        for instance in self.get_all_passivbot_instances():
            iid = instance.get_id()
            if self.instances.get(iid) is None:
                instance.is_in_config(False)
                self.instances[iid] = instance

    def get_instances(self) -> List[Instance]:
        return self.instances.values()

    def get_instances_length(self) -> int:
        return len(self.get_instances())

    def get_instance_by_id(self, instance_id) -> Instance:
        return self.instances[instance_id]

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

    def get_all_passivbot_instances(self) -> List[Instance]:
        """Get all passivbot instances running on this machine"""
        signature = "^{}".format(" ".join(INSTANCE_SIGNATURE_BASE))
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
