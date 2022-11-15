from typing import Any, Dict, List, Callable
from constants import MANAGER_PATH
from instance import Instance
from manager import Manager
from shutil import copyfile
from sys import argv
from os import path
import logging
import sys


class CLI:
    def __init__(self):
        self.commands_available = {
            "sync": self.sync,
            "list": self.list,
            "info": self.info,
            "start": self.start,
            "stop": self.stop,
            "restart": self.restart,
            "init": self.init,
            "help": self.help,
        }

        self.flags_available = {
            "all": {
                "variants": ["-a", "--all"],
                "docs": "perform action on all instances",
            },
            "unsynced": {
                "variants": ["-u", "--unsynced"],
                "docs": "perform action on unsynced instances",
            },
            "yes": {
                "variants": ["-y", "--yes"],
                "docs": "do not ask for confirmation",
            },
            "silent": {
                "variants": ["-s", "--silent"],
                "docs": "disable logging to file",
            },
            "force": {
                "variants": ["-f", "--force"],
                "docs": "force action",
            },
            "help": {
                "variants": ["-h", "--help"],
                "docs": "show help for a command",
            },
        }

        self.manager = None
        self.args = []
        self.flags = {}

        self.instance_formatter = "  {:<15} {:<10} {:<10} {:<10} {:<30}"

    # ---------------------------------------------------------------------------- #
    #                                   commands                                   #
    # ---------------------------------------------------------------------------- #

    def sync(self):
        """Sync instances with config
        This will perform stop and start
        on all instances. See help for those
        commands to understand consequences.
        Args:  [ query ]
        Flags: [ -y ] [ -s ]"""

        self.flags["all"] = True
        self.flags["force"] = True

        delimiter = "-" * 20
        logging.info("Syncing instances...")
        logging.info(delimiter)
        self.stop()
        logging.info(delimiter)
        self.access_manager().sync_instances()
        self.start()

        logging.info(delimiter)
        logging.info("Sync complete")
        logging.info(delimiter)
        self.list()

    def list(self):
        """List all instances, if no query is given.
        If a query is given, only matching instances
        will be shown.
        Args:  [ query ]"""

        instances_synced = self.access_manager().get_synced_instances()
        instances_unsynced = self.access_manager().get_unsynced_instances()

        if len(instances_synced) > 0:
            self.print_instances(instances_synced, title="Instances:")

        if len(instances_unsynced) > 0:
            self.print_instances(instances_unsynced, title="\nUnsynced:")

        logging.info('\nUse "manager info <query>" to get more info')

    def info(self):
        """Get detailed info about an instance.
        If multiple instances are matched,
        only the first one will be shown.
        Args:  <query>"""
        instances = self.get_instances_for_action()
        if len(instances) == 0:
            return

        instance = instances[0]
        flags = instance.get_flags()

        lines = []
        lines.extend(
            [
                "Instance {}:".format(instance.get_id()),
                "  user: {}".format(instance.user),
                "  symbol: {}".format(instance.symbol),
                "  config: {}".format(instance.config),
                "  pid: {}".format(instance.get_pid_str()),
                "  status: {}".format(instance.get_status()),
                "Flags:",
            ]
        )

        lines.extend(["  {}: {}".format(flags[i], flags[i + 1])
                     for i in range(0, len(flags), 2)])

        for line in lines:
            logging.info(line)

    def start(self):
        """Start instances that match
        the given arguments.
        Args:  [ query ]
        Flags: [ -a ] [ -s ] [ -y ]"""

        silent = self.flags.get("silent", False)

        logging.info("Seeking for stopped instances...")

        instances_to_start = self.get_instances_for_action(lambda i: not i.is_running())
        if len(instances_to_start) == 0:
            return

        if not self.confirm_action("start", instances_to_start):
            return

        logging.info("Starting instances...")
        started_instances = []
        for instance in instances_to_start:
            if instance.is_running():
                continue
            started = instance.start(silent)
            if started:
                started_instances.append(instance.get_id())

        logging.info("Started {} instance(s)".format(len(started_instances)))
        for instance_id in started_instances:
            logging.info("- {}".format(instance_id))

    def stop(self):
        """Stop instances that match
        the given arguments.
        You will be prompted to stop
        instances that are out of sync.
        Args:  [ query ]
        Flags: [ -a ] [ -u ] [ -y ] [ -f ]"""

        force = self.flags.get("force", False)

        logging.info("Seeking for running instances...")

        instances_to_stop = self.get_instances_for_action(lambda i: i.is_running())
        if len(instances_to_stop) == 0:
            return

        if not self.confirm_action("stop", instances_to_stop):
            return

        logging.info("Stopping instances. This may take a while...")
        stopped_instances = []
        for instance in instances_to_stop:
            if instance.stop(force):
                stopped_instances.append(instance.get_id())

        logging.info("Stopped {} instance(s)".format(len(stopped_instances)))
        for instance_id in stopped_instances:
            logging.info("- {}".format(instance_id))

    def restart(self):
        """Restart instances that match
        the given arguments.
        Args:  [ query ]
        Flags: [ -a ] [ -u ] [ -s ] [ -f ] [ -y ]"""
        force = self.flags.get("force", False)
        silent = self.flags.get("silent", False)

        instances_to_restart = self.get_instances_for_action()
        if len(instances_to_restart) == 0:
            return

        if not self.confirm_action("restart", instances_to_restart):
            return

        logging.info("Restarting instances. This may take a while...")
        restarted_instances = []
        for instance in instances_to_restart:
            if instance.restart(force, silent):
                restarted_instances.append(instance.get_id())

        logging.info("Restarted {} instance(s)".format(
            len(restarted_instances)))
        for instance_id in restarted_instances:
            logging.info("- {}".format(instance_id))

    def init(self):
        """Create config file"""
        source = path.join(MANAGER_PATH, "config.example.yaml")
        target = path.join(MANAGER_PATH, "config.yaml")

        if path.exists(target):
            logging.info("Config file already exists -> {}".format(target))
            return

        copyfile(source, target)
        logging.info("Created a config file -> {}".format(target))

    def help(self):
        """Show help
        Args:  [ command ]"""

        if len(self.args) > 0:
            command = self.commands_available.get(self.args[0], None)
            if command is None:
                logging.info("No such command: {}".format(self.args[0]))
                return

            logging.info("Help for {}:".format(self.args[0]))
            doc_lines = self.prepare_function_doc(command)
            for line in doc_lines:
                logging.info("  {}".format(line))
            return

        lines = [
            "Usage: manager <command> [args]",
            "\n  CLI for managing instances of PassivBot\n",
            "Commands:",
        ]

        for command, func in self.commands_available.items():
            doc_lines = self.prepare_function_doc(func)

            lines.append("  {:<8} - {}".format(command, doc_lines[0]))
            for line in doc_lines[1:]:
                lines.append("  {:<8} {}".format("", line))

        lines.extend(
            [
                "\nArgs:",
                "  <arg>  - required argument",
                "  [arg]  - optional argument",
                "  query    - a query to match instances.",
                "           These params for queries:",
                "           user, symbol, status, pid, id.",
                "           You can use several params at once,",
                "           separate them with spaces for that.",
                "           Examples:",
                "             - user=passivbot symbol=btc",
                "             - symbol=btcusd status=running",
                "             - passivbot stopped",
            ]
        )

        lines.append("\nFlags:")
        for flag in self.flags_available.values():
            variants = ", ".join(flag["variants"])
            lines.append(
                "  {:<15} - {}".format(
                    variants,
                    flag["docs"],
                )
            )

        for line in lines:
            logging.info(line)

    # ---------------------------------------------------------------------------- #
    #                                    helpers                                   #
    # ---------------------------------------------------------------------------- #

    def access_manager(self) -> Manager:
        if self.manager is None:
            self.manager = Manager()

        return self.manager

    def print_instances(self, instances, **kwargs):
        if len(instances) == 0:
            return

        def log(*args):
            lines.extend(args)

        def format_instance(instance):
            return self.instance_formatter.format(
                instance.get_user(),
                instance.get_symbol(),
                instance.get_status(),
                instance.get_pid_str(),
                instance.get_id())

        title = kwargs.get("title", "Instances:")
        headers = kwargs.get("headers", True)

        lines = []
        log(title)

        if headers:
            log(self.instance_formatter.format(
                "user", "symbol", "status", "pid", "id"))

        log(*[format_instance(instance) for instance in instances])

        for line in lines:
            logging.info(line)

    def confirm_action(self, action, instances) -> bool:
        if len(instances) == 0:
            return False

        if self.flags.get("yes", False):
            return True

        def action_message(message):
            return 'Action "{}" will be perfromed on {}'.format(action, message)

        logging.info(action_message(
            '{} instance(s). Continue?'.format(len(instances))))

        while True:
            try:
                raw_answer = input(
                    'yes or y to confirm\nlist or l to list affected instances\nCtrl+C or exit to cancel: ')
            except KeyboardInterrupt:
                logging.info("\nAborted")
                sys.exit(1)

            answer = raw_answer.lower()

            if answer in ["yes", "y"]:
                return True
            elif answer in ["list", "l"]:
                self.print_instances(
                    instances, title="\n{}".format(action_message("these instances:")))
            elif answer == "exit":
                logging.info("Aborted")
                return False

    def get_instances_for_action(self, filter: Callable = None) -> List[Instance]:
        if self.access_manager().get_instances_length() == 0:
            logging.warn("You have no instances configured")
            return []

        if self.flags.get("all", False):
            instances = self.access_manager().get_instances()
        elif self.flags.get("unsynced", False):
            instances = self.access_manager().get_unsynced_instances()
        else:
            instances = self.access_manager().query_instances(self.args)

        if callable(filter):
            instances = [
                instance for instance in instances if filter(instance)]

        if len(instances) == 0:
            logging.warn("No instances matched the given arguments")

        return instances

    def parse_args(self, args=[]) -> Dict[str, Any]:
        def flag_key(arg):
            if "=" in arg:
                arg = arg.split("=")[0]
            return arg

        def flag_value(arg):
            if "=" not in arg:
                return True
            return arg.split("=")[1]

        passed_flags = {}
        clean_args = []
        for arg in args:
            if len(arg) < 2 or arg[0] != "-":
                clean_args.append(arg)
                continue

            value = flag_value(arg)
            arg = flag_key(arg)

            if arg[1] == "-":
                passed_flags[arg] = value
                continue

            for flag in list(arg[1:]):
                passed_flags["-" + flag] = value
            continue

        flags = {}
        for flag_name in self.flags_available.keys():
            flag = self.flags_available[flag_name]
            for variant in flag["variants"]:
                passed = passed_flags.get(variant, False)
                if passed:
                    flags[flag_name] = passed
                    break

        return {"flags": flags, "args": clean_args}

    def prepare_function_doc(self, func) -> List[str]:
        doc = func.__doc__
        if doc is None:
            return []

        doc = doc.split("\n")
        return [line.strip() for line in doc]

    def run_command(self, args=[]):
        if len(args) == 0:
            self.help()
            return

        command = self.commands_available.get(args[0], None)
        if command is None:
            logging.info("No such command: {}".format(args[0]))
            return

        parsed_args = self.parse_args(args[1:])
        self.flags = parsed_args["flags"]
        self.args = parsed_args["args"]

        if self.flags.get("help", False):
            self.args = [args[0]]
            self.help()
            return

        try:
            command()
        except KeyboardInterrupt:
            logging.info("\nAborted")


if __name__ == "__main__":
    CLI().run_command(argv[1:])
