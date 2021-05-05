from datetime import datetime, timedelta
import json

import git
import sys
from prettytable import PrettyTable, HEADER
from telegram import KeyboardButton, ParseMode, ReplyKeyboardMarkup
from telegram.ext import Updater, CommandHandler
from time import time

import passivbot


class Telegram:
    def __init__(self, token: str, chat_id: str, bot, loop):
        self._bot = bot
        self.loop = loop
        self._chat_id = chat_id
        self._updater = Updater(token=token)
        self.config_reload_ts = 0.0

        keyboard_buttons = [
            [KeyboardButton('/balance'), KeyboardButton('/orders'), KeyboardButton('/position')],
            [KeyboardButton('/graceful_stop'), KeyboardButton('/show_config'), KeyboardButton('/reload_config')],
            [KeyboardButton('/closed_trades'), KeyboardButton('/daily'), KeyboardButton('/help')]]
        self._keyboard = ReplyKeyboardMarkup(keyboard_buttons, resize_keyboard=True)

        dispatcher = self._updater.dispatcher
        dispatcher.add_handler(CommandHandler('balance', self._balance))
        dispatcher.add_handler(CommandHandler('orders', self._orders))
        dispatcher.add_handler(CommandHandler('position', self._position))
        dispatcher.add_handler(CommandHandler('graceful_stop', self._graceful_stop))
        dispatcher.add_handler(CommandHandler('show_config', self.show_config))
        dispatcher.add_handler(CommandHandler('reload_config', self._reload_config))
        dispatcher.add_handler(CommandHandler('closed_trades', self._closed_trades))
        dispatcher.add_handler(CommandHandler('daily', self._daily))
        dispatcher.add_handler(CommandHandler('help', self._help))
        self._updater.start_polling()

    def _help(self, update=None, context=None):
        msg = '<pre><b>The following commands are available:</b></pre>\n' \
              '/balance: the equity & wallet balance in the configured account\n' \
              '/orders: a list of all buy & sell orders currently open\n' \
              '/graceful_stop: instructs the bot to no longer open new positions and exit gracefully\n' \
              '/position: information about the current position(s)\n' \
              '/show_config: the active configuration used\n' \
              '/reload_config: reload the configuration from disk, based on the file initially used\n' \
              '/closed_trades: a brief overview of the last 10 closed trades\n' \
              '/daily: an overview of daily profit\n' \
              '/help: This help page\n'
        self.send_msg(msg)

    def _orders(self, update=None, context=None):
        open_orders = self._bot.open_orders
        order_table = PrettyTable(["l/s", "b/s", "price", "qty"])

        for order in open_orders:
            price = passivbot.round_dynamic(order['price'], 3)
            qty = passivbot.round_dynamic(order['qty'], 3)
            side = order['side']
            position_side = order['position_side']
            order_table.add_row([position_side, side, price, qty])

        table_msg = order_table.get_string(sortby="price", border=True, padding_width=1,
                                           junction_char=' ', vertical_char=' ', hrules=HEADER)
        msg = f'<pre>{table_msg}</pre>'
        self.send_msg(msg)

    def _position(self, update=None, context=None):
        position_table = PrettyTable(['', 'Long', 'Short'])
        if 'long' in self._bot.position:
            long_position = self._bot.position['long']
            shrt_position = self._bot.position['shrt']

            position_table.add_row(['Size', passivbot.round_dynamic(long_position['size'], 3),
                                    passivbot.round_dynamic(shrt_position['size'], 3)])
            position_table.add_row(['Price', passivbot.round_dynamic(long_position['price'], 3),
                                    passivbot.round_dynamic(shrt_position['price'], 3)])
            position_table.add_row(['Leverage', long_position['leverage'], shrt_position['leverage']])
            position_table.add_row(['Liq.price', passivbot.round_dynamic(long_position['liquidation_price'], 3),
                                    passivbot.round_dynamic(shrt_position['liquidation_price'], 3)])
            position_table.add_row(['Liq.diff', passivbot.round_dynamic(float(long_position['liq_diff']) * 100, 3),
                                    passivbot.round_dynamic(float(shrt_position['liq_diff']) * 100, 3)])
            position_table.add_row(['UPNL', passivbot.round_dynamic(float(long_position['upnl']), 3),
                                    passivbot.round_dynamic(float(shrt_position['upnl']), 3)])

            table_msg = position_table.get_string(border=True, padding_width=1,
                                                  junction_char=' ', vertical_char=' ',
                                                  hrules=HEADER)
            self.send_msg(f'<pre>{table_msg}</pre>')
        else:
            self.send_msg("Position not initialized yet, please try again later")

    def _balance(self, update=None, context=None):
        if bool(self._bot.position):
            msg = '<pre><b>Balance:</b></pre>\n' \
                  f'Equity: {passivbot.round_dynamic(self._bot.position["equity"], 3)}\n' \
                  f'Locked margin: {passivbot.round_dynamic(self._bot.position["used_margin"], 3)}\n' \
                  f'Available margin: {passivbot.round_dynamic(self._bot.position["available_margin"], 3)}'
        else:
            msg = 'Balance not retrieved yet, please try again later'
        self.send_msg(msg)

    def _graceful_stop(self, update=None, context=None):
        self._bot.set_config_value('do_long', False)
        self._bot.set_config_value('do_shrt', False)

        self.send_msg(
            'No longer opening new long or short positions, existing positions will be closed gracefully')

    def _reload_config(self, update=None, context=None):
        if self.config_reload_ts > 0.0:
            if time() - self.config_reload_ts < 60 * 5:
                self.send_msg('Config reload in progress, please wait')
                return
        self.config_reload_ts = time()
        self.send_msg('Reloading config...')

        try:
            config = json.load(open(sys.argv[3]))
        except Exception:
            self.send_msg("Failed to load config file")
            self.config_reload_ts = 0.0
            return

        self._bot.pause()
        self._bot.set_config(config)

        def init_finished(task):
            self._bot.resume()
            self.log_start()
            self.config_reload_ts = 0.0

        task = self.loop.create_task(self._bot.init_indicators())
        task.add_done_callback(init_finished)

    def _closed_trades(self, update=None, context=None):
        from binance import BinanceBot
        if isinstance(self._bot, BinanceBot):
            async def send_closed_trades_async():
                trades = await self._bot.fetch_trades(limit=100)
                open_trades = [t for t in trades if float(t['realizedPnl']) > 0]
                open_trades.sort(key=lambda x: x['time'], reverse=True)

                table = PrettyTable(['Date', 'Pos.', 'Price', 'Pnl'])

                for trade in open_trades[:10]:
                    trade_date = datetime.datetime.fromtimestamp(trade['time'] / 1000)
                    table.add_row(
                        [trade_date.strftime('%d-%m %H:%M'), trade['positionSide'], trade['price'],
                         passivbot.round_dynamic(float(trade['realizedPnl']), 3)])

                msg = f'Closed trades:\n' \
                      f'<pre>{table.get_string(border=True, padding_width=1, junction_char=" ", vertical_char=" ", hrules=HEADER)}</pre>'
                self.send_msg(msg)

            self.loop.create_task(send_closed_trades_async())
        else:
            self.send_msg('This command is not supported (yet) on Bybit')

    def _daily(self, update=None, context=None):
        from binance import BinanceBot
        if isinstance(self._bot, BinanceBot):
            async def send_daily_async():
                table = PrettyTable(['Date', 'Profit'])
                nr_of_days = 7
                today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                for idx, item in enumerate(range(0, nr_of_days)):
                    start_of_day = today - timedelta(days=idx)
                    end_of_day = start_of_day + timedelta(days=1)
                    start_time = int(start_of_day.timestamp()) * 1000
                    end_time = int(end_of_day.timestamp()) * 1000
                    daily_trades = await self._bot.fetch_trades(start_time=start_time, end_time=end_time)
                    pln_summary = 0
                    for trade in daily_trades:
                        pln_summary += float(trade['realizedPnl'])
                    table.add_row([start_of_day.strftime('%d-%m'), passivbot.round_dynamic(pln_summary, 3)])

                msg = f'<pre>{table.get_string(border=True, padding_width=1, junction_char=" ", vertical_char=" ", hrules=HEADER)}</pre>'
                self.send_msg(msg)

            self.loop.create_task(send_daily_async())
        else:
            self.send_msg('This command is not supported (yet) on Bybit')

    def show_config(self, update=None, context=None):
        try:
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha
            sha_short = repo.git.rev_parse(sha, short=True)
        except:
            sha_short = 'UNKNOWN'

        msg = f'<pre><b>Version:</b></pre> {sha_short},\n' \
              f'<pre><b>Config:</b></pre> \n' \
              f'{json.dumps(self._bot.config, indent=4)}'
        self.send_msg(msg)

    def log_start(self):
        self.send_msg('<b>Passivbot started!</b>')
        self.show_config()

    def send_msg(self, msg: str):
        try:
            self._updater.bot.send_message(
                self._chat_id,
                text=msg,
                parse_mode=ParseMode.HTML,
                reply_markup=self._keyboard,
                disable_notification=False
            )
        except:
            print('Failed to send telegram message')

    def exit(self, signum, frame):
        try:
            self._updater.stop()
            print("Succesfully shutdown telegram bot")
        except:
            print("Failed to shutdown telegram bot. Please make sure it is correctly terminated")
        raise KeyboardInterrupt
