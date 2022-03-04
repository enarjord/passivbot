from manager import Manager
from pm import ProcessManager


class CLI:
    def __init__(self):
        self.commands = {
            'sync': self.sync,
            'list': self.list,
            'start-all': self.start_all,
            'start': self.start,
            'stop-all': self.stop_all,
            'stop': self.stop,
            'help': self.help,
        }

        self.manager = Manager()

    def help(self, args):
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
        for command in self.commands:
            print('    {} - {}'.format(command,
                  self.commands[command].__doc__))

    def sync(self, args):
        '''Sync instances with config
        Flags:
            -y: confirm stop of unsynced instances
        This will restart all instances from current config!
        You will be prompted to stop all instances that are not in config anymore.'''

        if len(args) > 0:
            if '-y' in args:
                stop_unsynced = True

        self.manager.stop_all()

        isntances_out_of_sync = self.manager.get_all_passivbot_instances()
        if len(isntances_out_of_sync) > 0 and not stop_unsynced:
            print('These instances are out of sync:')
            for instance in isntances_out_of_sync:
                print('- {} {} {}'
                      .format(instance.user,
                              instance.symbol,
                              instance.live_config_path))
            stop = input('Do you want to stop them? (y/n) ')
            if stop.lower() == 'y':
                stop_unsynced = True

        if stop_unsynced:
            stopped_instances = []
            for instance in isntances_out_of_sync:
                stopped = self.manager.stop_attempt(instance)
                if stopped:
                    stopped_instances.append(instance.get_id())
                    print('Successfully stopped these instances:')
                    for instance_id in stopped_instances:
                        print('- {}'.format(instance_id))

        self.manager.sync_config()
        self.manager.start_all()
        print('Sync complete.')
        self.list()

    def list(self, args=None):
        '''List running instances'''
        instances = self.manager.get_instances()
        print('Instances:')
        format_str = '  {:<15} {:<10} {:<10} {:<10}'
        print(format_str.format('user', 'symbol', 'status', 'pid'))
        for instance in instances:
            pm = ProcessManager()
            pid = pm.get_pid(instance.get_pid_signature())
            status = 'running' if pid is not None else 'stopped'
            print(format_str.format(
                instance.user,
                instance.symbol,
                status,
                str(pid) if pid is not None else '-'
            ))

    def start_all(self, args=None):
        '''Start all instances'''
        self.manager.start_all()

    def start(self, args=None):
        '''Start a new instance
        Args: instance_id (account:symbol)
        Example: start binance_01:BTCUSDT'''
        if len(args) == 0:
            self.help('start')
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

    def stop_all(self, args=None):
        '''Stop all running instances'''

        stopped = self.manager.stop_all()
        len_stopped = len(stopped)
        print('Stopped {} instance(s)'.format(len_stopped))
        if len_stopped > 0:
            for instance_id in stopped:
                print('- {}'.format(instance_id))

    def stop(self, args=None):
        '''Stop a running instance
        Args: instance_id (account:symbol)
        Example: stop binance_01:BTCUSDT'''
        if len(args) == 0:
            self.help('stop')
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

    def run_command(self, args=None):
        if len(args) == 0:
            self.help()
            return

        command = args[0]
        if command in self.commands:
            self.commands[command](args[1:])
        else:
            self.help()
