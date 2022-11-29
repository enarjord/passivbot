from typing import Dict, List, Any, Callable, Union, Type
from manager.cli.progress import Progress
from manager.instance import Instance
from manager.constants import logger
from manager.cli.color import Color
from argparse import ArgumentParser
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

    def parse_input(self, input=[]) -> Dict[str, Any]:

        def base_parser() -> ArgumentParser:
            parser = ArgumentParser(add_help=False, allow_abbrev=False)
            return parser

        parser = base_parser()
        parser.add_argument("args", nargs="*")
        parsed_args = vars(parser.parse_known_args()[0])
        parsed_args = parsed_args.get("args")

        args = []
        if len(parsed_args) > 1:
            args = parsed_args[1:]

        parser = base_parser()
        for name, flag in self.flags_available.items():
            variants = flag.get("variants", [])
            if len(variants) == 0:
                continue

            type = flag.get("type")
            if type:
                parser.add_argument(*variants, type=type, dest=name)
            else:
                parser.add_argument(*variants, action="store_true", dest=name)

        flags = parser.parse_known_args()[0]

        return {"args": args, "flags": vars(flags)}

    def add_command(self, command: str, executor: "CLICommand"):
        self.commands_available[command] = executor

    def add_flag(self, name: str, variants: List[str], doc: str, type: Type = None):
        self.flags_available[name] = {
            "variants": variants,
            "doc": doc,
            "type": type,
        }

    def add_progress(self, initial_message: str = "") -> Progress:
        thread = Progress(Event(), initial_message)
        self.threads_.append(thread)
        return thread

    def run_command(self, command: str):
        executor = self.commands_available.get(command)
        if executor is None:
            logger.info(f"Unknown command: {command}")
            return

        try:
            executor.run(self)
        except KeyboardInterrupt:
            self.on_exit()
            logger.info("\nAborted")
        except:
            self.on_exit()
            logger.error(
                "\nSomething went wrong, there should be possible reasons above")

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
            lines.append(f"{title}:")

        groups = self.manager.group_instances_by_user(instances)
        for user, group in groups.items():
            running = self.manager.count_running(group, format=True)
            user = Color.apply(Color.CYAN, user)
            lines.append(f"- {user} ({running})")
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
        if len(self.manager.get_instances()) == 0:
            logger.info("You have no instances configured")
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
            logger.warn(
                f"No instances matched the given arguments: {' '.join(self.args)}")

        return instances

    def confirm_action(self, action, instances) -> bool:
        if len(instances) == 0:
            return False

        if self.flags.get("yes", False):
            return True

        def action_message(message):
            return f"Action \"{action}\" will be perfromed on {message}"

        logger.info(action_message(f"{len(instances)} instance(s). Continue?"))

        def variant(text: str) -> str:
            return Color.apply(Color.LIGHT_PURPLE, text)

        while True:
            try:
                prompts = [
                    f"{variant('yes')} or {variant('y')} to confirm",
                    f"{variant('list')} or {variant('l')} to see affected instances",
                    f"{variant('no')}, {variant('n')} or Ctrl+C to abort:",
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
                    instances,  title=f"\n{action_message('these instances')}")
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
            config_path = self.flags.get("config_path", None)
            self.manager_ = Manager(config_path)

        return self.manager_

    @property
    def modifiers(self) -> Dict[str, Any]:
        flag_value = self.flags.get("modifiers")
        if flag_value is None:
            return {}

        it = iter(flag_value.split(" "))
        return dict(zip(it, it))


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
            lines.extend([f"{line.strip()}"
                         for line in self.doc.split("\n")])

        args_line = []
        if hasattr(self, "args_required") and len(self.args_required) > 0:
            args_line.extend([f"<{arg.strip()}>"
                              for arg in self.args_required])

        if hasattr(self, "args_optional") and len(self.args_optional) > 0:
            args_line.extend([f"[ {arg.strip()} ]"
                              for arg in self.args_optional])

        if len(args_line) > 0:
            lines.append(f"Args: {' '.join(args_line)}")

        if hasattr(self, "flags") and len(self.flags) > 0:
            lines.append(f"Flags: [ {' ] [ '.join(self.flags)} ]")

        return lines
