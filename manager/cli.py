import logging
from sys import argv
from typing import Dict
from manager import Manager


class CLI:
    def __init__(self):
        self.commands = {
            'sync': self.sync,
            'list': self.list,
            'start': self.start,
            'stop': self.stop,
            'restart': self.restart,
            'help': self.help,
        }

        self.flags = {
            'all': {
                'variants': ['-a', '--all'],
                'docs': 'perform action on all instances',
            },
            'yes': {
                'variants': ['-y', '--yes'],
                'docs': 'do not ask for confirmation',

            },
            'silent': {
                'variants': ['-s', '--silent'],
                'docs': 'disable logging to file',
            },
            'force': {
                'variants': ['-f', '--force'],
                'docs': 'force action',
            }
        }

        self.manager = Manager()

    # ---------------------------------------------------------------------------- #
    #                                   commands                                   #
    # ---------------------------------------------------------------------------- #

    def sync(self, args=[]):
        '''Sync instances with config
        This will perform stop and start 
        on all instances. See help for those 
        commands to understand consequences.
        Flags: [ -y ] [ -s ] [ -f ]'''

        delimiter = '-' * 20
        self.stop(['-a'] + args)
        logging.info(delimiter)
        self.manager.sync_config()
        self.start(['-a'] + args)
        logging.info(delimiter)
        logging.info('Sync complete')
        logging.info(delimiter)
        self.list()

    def list(self, args=[]):
        '''List running instances,
        or get detailed info about an instance.
        Args: [ instance_id ]'''
        if len(args) > 0:
            self.instance_info(args)
            return

        instances = self.manager.get_instances()
        format_str = '  {:<15} {:<10} {:<10} {:<10} {:<30}'
        logging.info('Instances:')
        logging.info(format_str.format(
            'user', 'symbol', 'status', 'pid', 'id'))
        for instance in instances:
            pid = instance.get_pid()
            logging.info(format_str.format(
                instance.user,
                instance.symbol,
                instance.get_status(),
                str(pid) if pid is not None else '-',
                instance.get_id(),
            ))

        logging.info('\nUse "manager info <instance_id>" to get more info')

    def instance_info(self, args=[]):
        instances = self.get_instances_for_action(args)
        if len(instances) == 0:
            return
        instance = instances[0]
        pid = instance.get_pid()
        pid = str(pid) if pid is not None else '-'
        logging.info('Instance {}:'.format(instance.get_id()))
        logging.info('  user: {}'.format(instance.user))
        logging.info('  symbol: {}'.format(instance.symbol))
        logging.info('  config: {}'.format(instance.live_config_path))
        logging.info('  pid: {}'.format(pid))
        logging.info('  status: {}'.format(instance.get_status()))
        logging.info('Flags:')
        flags = instance.get_flags()
        for i in range(0, len(flags), 2):
            logging.info('  {}: {}'.format(flags[i], flags[i + 1]))

    def start(self, args=[]):
        '''Start a new instance.
        Args: [ instance_id ]
        Flags: [ -a ] [ -s ]'''
        if len(args) == 0:
            self.help(['start'])
            return

        flags = self.parse_flags(args)
        silent = flags.get('silent', False)
        instances_to_start = self.get_instances_for_action(args)
        if len(instances_to_start) == 0:
            return

        logging.info('Starting instances...')
        started_instances = []
        for instance in instances_to_start:
            if instance.is_running():
                continue
            started = instance.start(silent)
            if started:
                started_instances.append(instance.get_id())

        logging.info('Started {} instance(s)'.format(len(started_instances)))
        for instance_id in started_instances:
            logging.info('- {}'.format(instance_id))

    def stop(self, args=[]):
        '''Stop running instance(s).
        You will be prompted to stop
        instances that are out of sync.
        Args: [ instance_id ]
        Flags: [ -a ] [ -y ] [ -f ]'''
        if len(args) == 0:
            self.help(['stop'])
            return

        flags = self.parse_flags(args)
        force = flags.get('force', False)
        instances_to_stop = self.get_instances_for_action(args)
        if len(instances_to_stop) > 0:
            logging.info('Stopping instances. This may take a while...')
            stopped_instances = []
            for instance in instances_to_stop:
                if instance.stop(force):
                    stopped_instances.append(instance.get_id())

            logging.info('Stopped {} instance(s)'.format(
                len(stopped_instances)))
            for instance_id in stopped_instances:
                logging.info('- {}'.format(instance_id))

        if flags.get('all', False):
            self.prompt_stop_unsynced(confirm=flags.get('yes', False))

    def restart(self, args=[]):
        '''Restart instance(s).
        Args: [ instance_id ]
        Flags: [ -a ] [ -s ] [ -f ]'''
        if len(args) == 0:
            self.help(['restart'])
            return

        flags = self.parse_flags(args)
        force = flags.get('force', False)
        silent = flags.get('silent', False)
        instances_to_restart = self.get_instances_for_action(args)
        if len(instances_to_restart) == 0:
            return

        logging.info('Restarting instances. This may take a while...')
        restarted_instances = []
        for instance in instances_to_restart:
            if instance.restart(force, silent):
                restarted_instances.append(instance.get_id())

        logging.info('Restarted {} instance(s)'.format(
            len(restarted_instances)))
        for instance_id in restarted_instances:
            logging.info('- {}'.format(instance_id))

    def help(self, args=[]):
        '''Show help'''
        if len(args) > 0:
            command = args[0]
        else:
            command = None

        if command is not None and command in self.commands:
            logging.info('Help for {}:'.format(command))
            logging.info(self.commands[command].__doc__)
            return

        logging.info('Usage: manager <command> [args]\n')
        logging.info('  CLI for managing instances of PassivBot\n')
        logging.info('Commands:')
        for command in self.commands.keys():
            doc_lines = self.commands[command].__doc__.split('\n')
            doc_lines = [line.strip() for line in doc_lines]

            logging.info('  {:<8} - {}'.format(command, doc_lines[0]))
            for line in doc_lines[1:]:
                logging.info('  {:<8} {}'.format('', line))

        logging.info('\nFlags:')
        for flag in self.flags.values():
            logging.info('  {:<13} - {}'.format(
                ', '.join(flag['variants']),
                flag['docs'],
            ))

    # ---------------------------------------------------------------------------- #
    #                                    helpers                                   #
    # ---------------------------------------------------------------------------- #

    def get_instances_for_action(self, args=[]):
        if len(args) == 0:
            logging.warn('No instances specified')
            return []

        total_instances = self.manager.get_instances_length()
        if total_instances == 0:
            logging.warn('You have no instances configured')
            return []

        flags = self.parse_flags(args)
        if flags.get('all', False):
            return self.manager.get_instances()

        instance = self.manager.get_instance_by_id(args[0])
        if instance is None:
            logging.warn('No instances matced the given id')
            return []
        return [instance]

    def prompt_stop_unsynced(self, confirm=False):
        unsynced = self.manager.get_all_passivbot_instances()
        if len(unsynced) == 0:
            return

        if not confirm:
            print('These instances are out of sync:')
            for instance in unsynced:
                print('- {} {} {}'
                      .format(instance.user,
                              instance.symbol,
                              instance.live_config_path))
            try:
                stop = input('Do you want to stop them? (y/n) ')
                if stop.lower() == 'y':
                    confirm = True
            except KeyboardInterrupt:
                confirm = False

        if not confirm:
            return

        stopped_instances = []
        for instance in unsynced:
            if instance.stop():
                stopped_instances.append(instance.get_id())

        if len(stopped_instances) > 0:
            print('Successfully stopped these instances:')
        for instance_id in stopped_instances:
            print('- {}'.format(instance_id))

    def parse_flags(self, args=[]) -> Dict[str, bool]:
        flags = {}
        for arg in args:
            for flag in self.flags.keys():
                if arg in self.flags[flag]['variants']:
                    flags[flag] = True
                    break
        return flags

    def run_command(self, args=[]):
        if len(args) == 0:
            self.help()
            return

        command = args[0]
        if command in self.commands:
            self.commands[command](args[1:])
        else:
            self.help()


if __name__ == '__main__':
    CLI().run_command(argv[1:])
