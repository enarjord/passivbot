import json
import signal

import git
import sys
from prettytable import PrettyTable, HEADER
from telegram import KeyboardButton, ParseMode, ReplyKeyboardMarkup
from telegram.ext import Updater, CommandHandler

class Telegram:
    def __init__(self, token: str, chat_id: str, bot):
        self._bot = bot
        self._chat_id = chat_id
        self._updater = Updater(token=token)

        keyboard_buttons = [
            [KeyboardButton('/balance'), KeyboardButton('/orders'), KeyboardButton('/position')],
            [KeyboardButton('/stop_buy'), KeyboardButton('/show_config'), KeyboardButton('/help')]]
        self._keyboard = ReplyKeyboardMarkup(keyboard_buttons, resize_keyboard=True)

        dispatcher = self._updater.dispatcher
        dispatcher.add_handler(CommandHandler('balance', self._balance))
        dispatcher.add_handler(CommandHandler('orders', self._orders))
        dispatcher.add_handler(CommandHandler('position', self._position))
        dispatcher.add_handler(CommandHandler('stop_buy', self._stop_buy))
        dispatcher.add_handler(CommandHandler('show_config', self.show_config))
        dispatcher.add_handler(CommandHandler('help', self._help))
        self._updater.start_polling()

    def _help(self, update=None, context=None):
        msg = '<pre><b>The following commands are available:</b></pre>\n' \
              '/balance: the equity & wallet balance in the configured account\n' \
              '/orders: a list of all buy & sell orders currently open\n' \
              '/stop_buy: instructs the bot to no longer open new positions and exit gracefully\n' \
              '/position: information about the current position(s)\n' \
              '/show_config: the config used\n' \
              '/help: This help page\n'
        self.send_msg(msg)

    def _orders(self, update=None, context=None):
        open_orders = self._bot.open_orders
        order_table = PrettyTable(["l/s", "b/s", "price", "qty"])

        for order in open_orders:
            price = order['price']
            qty = order['qty']
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

            position_table.add_row(['Size', long_position['size'], shrt_position['size']])
            position_table.add_row(['Price', long_position['price'], shrt_position['price']])
            position_table.add_row(['Leverage', long_position['leverage'], shrt_position['leverage']])
            position_table.add_row(
                ['Liq.price', long_position['liquidation_price'], shrt_position['liquidation_price']])
            position_table.add_row(['Liq.diff', long_position['liq_diff'], shrt_position['liq_diff']])
            position_table.add_row(['UPNL', long_position['upnl'], shrt_position['upnl']])

            table_msg = position_table.get_string(border=True, padding_width=1,
                                                  junction_char=' ', vertical_char=' ', hrules=HEADER)
            self.send_msg(f'<pre>{table_msg}</pre>')
        else:
            self.send_msg("Position not initialized yet, please try again later")

    def _balance(self, update=None, context=None):
        if bool(self._bot.position):
            msg = '<pre><b>Balance:</b></pre>\n' \
                  f'Equity: {self._bot.position["equity"]}\n' \
                  f'Used margin: {self._bot.position["used_margin"]}\n' \
                  f'Available margin: {self._bot.position["available_margin"]}'
        else:
            msg = 'Balance not retrieved yet, please try again later'
        self.send_msg(msg)

    def _stop_buy(self, update=None, context=None):
        self._bot.set_config_value('do_long', False)
        self._bot.set_config_value('do_short', False)

        self.send_msg('No longer buying long or short, existing positions will be closed gracefully')

    def show_config(self, update=None, context=None):
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        sha_short = repo.git.rev_parse(sha, short=True)

        msg = f'<pre><b>Version:</b></pre> {sha_short},\n' \
              f'<pre><b>Config:</b></pre> \n' \
              f'{json.dumps(self._bot.config, indent=4)}'
        self.send_msg(msg)

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
            print('Failed to send telegram message',)

    def exit(self, signum, frame):
        try:
            self._updater.stop()
            print("Succesfully shutdown telegram bot")
        except:
            print("Failed to shutdown telegram bot. Please make sure it is correctly terminated")
        raise KeyboardInterrupt
