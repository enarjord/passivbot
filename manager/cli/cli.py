from typing import Dict, List, Any, Callable, Union
from traceback import format_exc as traceback
from manager.cli.progress import Progress
from manager.instance import Instance
from manager.constants import logger
from manager.cli.color import Color
from manager import Manager
from threading import Event


class CLI:
    def __init__(self) -> None:
        self.commands_available: Dict[str, CLICommand] = {}
        self.flags_available: Dict[str, Dict] = {}

        self.args = []
        self.flags = {}

        self.threads_: List[Progress] = []
        self.manager_: Union[Manager, None] = None

    def parse_input(self, args=[]) -> Dict[str, Any]:
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

    def add_command(self, command: str, executor: "CLICommand"):
        self.commands_available[command] = executor

    def add_flag(self, name: str, variants: List[str], doc: str):
        self.flags_available[name] = {
            "variants": variants,
            "doc": doc
        }

    def add_progress(self, initial_message: str = "") -> Progress:
        thread = Progress(Event(), initial_message)
        self.threads_.append(thread)
        return thread

    def run_command(self, command: str):
        executor = self.commands_available.get(command)
        if executor is None:
            logger.info("Unknown command: {}".format(command))
            return

        try:
            executor.run(self)
        except KeyboardInterrupt:
            self.on_exit()
            logger.info("\nAborted")
        except:
            self.on_exit()
            logger.error("\n{}".format(traceback()))

    def run(self, args: List):
        if len(args) == 0:
            return

        parsed_input = self.parse_input(args[1:])
        self.flags = parsed_input.get("flags")
        self.args = parsed_input.get("args")
        self.run_command(args[0])

    def on_exit(self):
        for thread in self.threads_:
            if not thread.is_alive():
                continue

            thread.finished.set()
            thread.join()

    # ---------------------------------------------------------------------------- #
    #                                   utilities                                  #
    # ---------------------------------------------------------------------------- #

    def format_instnaces(self, instances: List[Instance], **kwargs) -> List[str]:
        if len(instances) == 0:
            return []

        lines = []
        title = kwargs.get("title")

        if title:
            lines.append("{}:".format(title))

        groups = self.manager.group_instances_by_user(instances)
        for user, group in groups.items():
            running = self.manager.count_running(group, format=True)
            user = Color.apply(Color.CYAN, user)
            lines.append("- {} ({})".format(user, running))
            lines.extend([self.format_instance(instance)
                          for instance in group])

        return lines

    def format_instance_like(self, *args) -> str:
        return "    {:1} {:<15}".format(*args)

    def format_instance(self, instance: Instance) -> str:
        color = Color.RED
        if instance.is_running():
            color = Color.GREEN

        status = Color.apply(color, "●")
        symbol = instance.get_symbol()
        return self.format_instance_like(status, symbol)

    def get_instances_for_action(self, filter: Callable = None) -> List[Instance]:
        if self.manager.get_instances_length() == 0:
            logger.warn("You have no instances configured")
            return []

        if self.flags.get("all", False):
            instances = self.manager.get_instances()
        elif self.flags.get("unsynced", False):
            instances = self.manager.get_unsynced_instances()
        else:
            instances = self.manager.query_instances(self.args)

        if callable(filter):
            instances = [
                instance for instance in instances if filter(instance)]

        if len(instances) == 0:
            logger.warn("No instances matched the given arguments")

        return instances

    def confirm_action(self, action, instances) -> bool:
        if len(instances) == 0:
            return False

        if self.flags.get("yes", False):
            return True

        def action_message(message):
            return 'Action "{}" will be perfromed on {}'.format(action, message)

        logger.info(action_message(
            '{} instance(s). Continue?'.format(len(instances))))

        while True:
            try:
                prompts = [
                    '{} or {} to confirm'.format(
                        *Color.apply(Color.LIGHT_PURPLE, "yes", "y")),
                    '{} or {} to see affected instances'.format(
                        *Color.apply(Color.LIGHT_PURPLE, "list", "l")),
                    '{}, {} or Ctrl+C to abort:'.format(
                        *Color.apply(Color.LIGHT_PURPLE, "no", "n")),
                ]
                raw_answer = input("\n".join(prompts))
            except KeyboardInterrupt:
                logger.info("\nAborted")
                return False

            answer = raw_answer.lower()

            if answer in ["yes", "y"]:
                return True
            elif answer in ["list", "l"]:
                lines = self.format_instnaces(
                    instances,  title="\n{}".format(action_message("these instances")))
                for line in lines:
                    logger.info(line)
            elif answer in ["no", "n"]:
                logger.info("Aborted")
                return False

    # ---------------------------------------------------------------------------- #
    #                                  properties                                  #
    # ---------------------------------------------------------------------------- #

    @property
    def manager(self) -> Manager:
        if self.manager_ is None:
            self.manager_ = Manager()

        return self.manager_


class CLICommand:
    doc: str
    args_required: List[str]
    args_optional: List[str]
    flags: List[str]

    @staticmethod
    def run(cli: CLI):
        pass

    @classmethod
    def get_docs(self) -> List[str]:
        lines = []

        if self.doc != "":
            lines.extend(["{}".format(line.strip())
                         for line in self.doc.split("\n")])

        args_line = []
        if hasattr(self, "args_required") and len(self.args_required) > 0:
            args_line.extend(["<{}>".format(arg.strip())
                              for arg in self.args_required])

        if hasattr(self, "args_optional") and len(self.args_optional) > 0:
            args_line.extend(["[ {} ]".format(arg.strip())
                              for arg in self.args_optional])

        if len(args_line) > 0:
            lines.append("Args: {}".format(" ".join(args_line)))

        if hasattr(self, "flags") and len(self.flags) > 0:
            lines.append("Flags: [ {} ]".format(" ] [ ".join(self.flags)))

        return lines
