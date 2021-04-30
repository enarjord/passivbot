import json

import git
from prettytable import PrettyTable
from telegram import KeyboardButton, ParseMode, ReplyKeyboardMarkup
from telegram.ext import Updater, CommandHandler

import passivbot


class Telegram:
    def __init__(self, token: str, chat_id: str, bot: passivbot.Bot):
        self._bot = bot
        self._chat_id = chat_id
        self._updater = Updater(token=token)

        keyboard_buttons = [
            [KeyboardButton('/balance'), KeyboardButton('/orders'), KeyboardButton('/position')],
            [KeyboardButton('/show_config'), KeyboardButton('/help')]]
        self._keyboard = ReplyKeyboardMarkup(keyboard_buttons, resize_keyboard=True)

        dispatcher = self._updater.dispatcher
        dispatcher.add_handler(CommandHandler('balance', self._balance))
        dispatcher.add_handler(CommandHandler('orders', self._orders))
        dispatcher.add_handler(CommandHandler('position', self._position))
        dispatcher.add_handler(CommandHandler('show_config', self.show_config))
        dispatcher.add_handler(CommandHandler('help', self._help))
        self._updater.start_polling()

    def _help(self, update=None, context=None):
        msg = '<pre><b>The following commands are available:</b></pre>\n' \
              '/balance: the equity & wallet balance in the configured account\n' \
              '/orders: a list of all buy & sell orders currently open\n' \
              '/position: information about the current position(s)\n' \
              '/show_config: the config used\n' \
              '/help: This help page\n'
        self.send_msg(msg)

    def _orders(self, update=None, context=None):
        open_orders = self._bot.open_orders
        order_table = PrettyTable(["long/short", "buy/sell", "Price", "Quantity"])

        for order in open_orders:
            price = order['price']
            qty = order['qty']
            side = order['side']
            position_side = order['position_side']
            order_table.add_row([position_side, side, price, qty])

        msg = f'<pre>{order_table.get_string(sortby="Price")}</pre>'
        self.send_msg(msg)

    def _position(self, update=None, context=None):
        position_table = PrettyTable(
            ["long/short", "size", "price", "leverage", "liquidation price", "upnl",
             "liquidation difference"])
        long_position = self._bot.position['long']
        position_table.add_row(
            ['long', long_position['size'], long_position['price'], long_position['leverage'],
             long_position['liquidation_price'], long_position['upnl'], long_position['liq_diff']])

        shrt_position = self._bot.position['shrt']
        position_table.add_row(
            ['short', shrt_position['size'], shrt_position['price'], shrt_position['leverage'],
             shrt_position['liquidation_price'], shrt_position['upnl'], shrt_position['liq_diff']])

        self.send_msg(f'<pre>{position_table}</pre>')

    def _balance(self, update=None, context=None):
        if bool(self._bot.position):
            msg = '<pre><b>Balance:</b></pre>\n' \
                  f'Equity: {self._bot.position["equity"]}\n' \
                  f'Used margin: {self._bot.position["used_margin"]}\n' \
                  f'Available margin: {self._bot.position["available_margin"]}'
        else:
            msg = 'Balance not retrieved yet, please try again later'
        self.send_msg(msg)

    def show_config(self, update=None, context=None):
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        sha_short = repo.git.rev_parse(sha, short=True)

        msg = f'<pre><b>Version:</b></pre> {sha_short},\n' \
              f'<pre><b>Config:</b></pre> \n' \
              f'{json.dumps(self._bot.settings, indent=4)}'
        self.send_msg(msg)

    def send_msg(self, msg: str):
        self._updater.bot.send_message(
            self._chat_id,
            text=msg,
            parse_mode=ParseMode.HTML,
            reply_markup=self._keyboard,
            disable_notification=False
        )
