from time import sleep
from manager import Manager
from pm import ProcessManager


class CLI:
    def __init__(self):
        self.commands = {
            'sync': self.sync,
            'list': self.list,
            'info': self.info,
            'start-all': self.start_all,
            'start': self.start,
            'stop-all': self.stop_all,
            'stop': self.stop,
            'help': self.help,
        }

        self.manager = Manager()

    def help(self, args=[]):
        '''Show help'''
        if len(args) > 0:
            command = args[0]
        else:
            command = None

        if command is not None and command in self.commands:
            print('Help for {}:'.format(command))
            print(self.commands[command].__doc__)
            return

        print('Usage:')
        print('    manager <command> [args]')
        print('Commands:')
        for i, command in enumerate(self.commands.keys()):
            print('    {}) {} - {}'
                  .format(i + 1, command, self.commands[command].__doc__))

    def info(self, args=[]):
        '''Show detailed info about instasnce
        Args: <instance_id>'''
        if len(args) == 0:
            self.help(['info'])
            return

        instance_id = args[0]
        instance = self.manager.get_instance_by_id(instance_id)
        if instance is None:
            print('Instance {} not found'.format(instance_id))
            return

        print('Instance {}:'.format(instance_id))
        print('\tuser: {}'.format(instance.user))
        print('\tsymbol: {}'.format(instance.symbol))
        print('\tlive config path: {}'.format(instance.live_config_path))
        print('Flags:')
        # pring flags in pairs
        flags = instance.get_flags()
        for i in range(0, len(flags), 2):
            print('\t{}: {}'.format(flags[i], flags[i + 1]))

    def list(self, args=[]):
        '''List running instances'''
        instances = self.manager.get_instances()
        print('Instances:')
        format_str = '  {:<15} {:<10} {:<10} {:<10} {:<30}'
        print(format_str.format('user', 'symbol', 'status', 'pid', 'id'))
        for instance in instances:
            pm = ProcessManager()
            pid = pm.get_pid(instance.get_pid_signature())
            status = 'running' if pid is not None else 'stopped'
            print(format_str.format(
                instance.user,
                instance.symbol,
                status,
                str(pid) if pid is not None else '-',
                instance.get_id(),
            ))

        print('\nUse "manager info <instance_id>" to get more info.')

    def start_all(self, args=[]):
        '''Start all instances'''
        self.manager.start_all()

    def start(self, args=[]):
        '''Start a new instance
        Args: <instance_id>'''
        if len(args) == 0:
            self.help(['start'])
            return

        instance_id = args[0]
        instance = self.manager.get_instance_by_id(instance_id)
        if instance is None:
            print('Instance {} not found'.format(instance_id))
            return

        if instance.is_running():
            print('Instance {} is already running'.format(instance_id))
            return

        instance.start()
        print('Started instance {}'.format(instance_id))

    def stop_all(self, args=[]):
        '''Stop all running instances'''

        stop_unsynced = False
        if len(args) > 0 and args[0] == '-y':
            stop_unsynced = True

        total_instances = self.manager.get_instances_length()
        if total_instances == 0:
            print('No instances running')
            return

        print('Stopping all instances. This may take a while...')
        stopped = self.manager.stop_all()
        len_stopped = len(stopped)
        print('Stopped {} instance(s)'.format(len_stopped))
        if len_stopped > 0:
            for instance_id in stopped:
                print('- {}'.format(instance_id))

        unsynced = self.manager.get_all_passivbot_instances()
        if len(unsynced) > 0 and not stop_unsynced:
            print('These instances are out of sync:')
            for instance in unsynced:
                print('- {} {} {}'
                      .format(instance.user,
                              instance.symbol,
                              instance.live_config_path))
            try:
                stop = input('Do you want to stop them? (y/n) ')
                if stop.lower() == 'y':
                    stop_unsynced = True
            except KeyboardInterrupt:
                stop_unsynced = False

        if stop_unsynced:
            stopped_instances = []
            for instance in unsynced:
                stopped = self.manager.stop(instance)
                if stopped:
                    stopped_instances.append(instance.get_id())

            if len(stopped_instances) > 0:
                print('Successfully stopped these instances:')
            for instance_id in stopped_instances:
                print('- {}'.format(instance_id))

    def stop(self, args=[]):
        '''Stop a running instance
        Args: <instance_id>'''
        if len(args) == 0:
            self.help(['stop'])
            return

        instance_id = args[0]
        instance = self.manager.get_instance_by_id(instance_id)
        if instance is None:
            print('Instance {} not found'.format(instance_id))
            return

        if instance.is_running():
            instance.stop()
            print('Stopped instance {}'.format(instance_id))
        else:
            print('Instance {} is not running'.format(instance_id))

    def sync(self, args=[]):
        '''Sync instances with config
        Flags:
            -y: confirm stop of unsynced instances

        This will restart all instances!
        You will be prompted to stop all instances that are not in config anymore.'''

        self.stop_all(args)
        self.manager.sync_config()
        self.manager.start_all()
        print('Sync complete.')
        self.list()

    def run_command(self, args=[]):
        if len(args) == 0:
            self.help()
            return

        command = args[0]
        if command in self.commands:
            self.commands[command](args[1:])
        else:
            self.help()
