import json
from datetime import datetime, timedelta
from time import time

try:
    import git
except Exception as e:
    print(e)
    print("Unable to import git module. This is probably a result of a missing git installation."
          "As a result of this, the version will not be shown in applicable messages. To fix this,"
          "please make sure you have git installed properly. The bot will work fine without it.")
from prettytable import PrettyTable, HEADER
from telegram import KeyboardButton, ParseMode, ReplyKeyboardMarkup, Update
from telegram.ext import Updater, CommandHandler, ConversationHandler, CallbackContext, \
    MessageHandler, Filters

from jitted import compress_float, round_, round_dynamic


class Telegram:
    def __init__(self, token: str, chat_id: str, bot, loop):
        self._bot = bot
        self.loop = loop
        self._chat_id = chat_id
        self._updater = Updater(token=token)
        self.config_reload_ts = 0.0
        self.n_trades = 10

        first_keyboard_buttons = [
            [KeyboardButton('\U0001F4B8 /daily'), KeyboardButton('\U0001F4CB /open_orders'), KeyboardButton('\U0001F4CC /position')],
            [KeyboardButton('\U0000274E /closed_trades'), KeyboardButton('\U0001F4DD /show_config'), KeyboardButton('\U0000267B /reload_config')],
            [KeyboardButton('\U0001F4B3 /balance'), KeyboardButton('\U00002753 /help'), KeyboardButton('\U000023E9 /next')]]
        second_keyboard_buttons = [
            [KeyboardButton('\U000026A1 /set_leverage')],
            [KeyboardButton('\U000023EA /previous'), KeyboardButton('\U000026D4 /stop')]
        ]
        self._keyboard_idx = 0
        self._keyboards = [ReplyKeyboardMarkup(first_keyboard_buttons, resize_keyboard=True),
                           ReplyKeyboardMarkup(second_keyboard_buttons, resize_keyboard=True)]

        self.add_handlers(self._updater)
        self._updater.start_polling()

    def add_handlers(self, updater):
        dispatcher = updater.dispatcher
        dispatcher.add_handler(MessageHandler(Filters.regex('.*/balance'), self._balance))
        dispatcher.add_handler(MessageHandler(Filters.regex('.*/open_orders'), self._open_orders))
        dispatcher.add_handler(MessageHandler(Filters.regex('.*/position'), self._position))
        dispatcher.add_handler(MessageHandler(Filters.regex('.*/show_config'), self.show_config))
        dispatcher.add_handler(
            MessageHandler(Filters.regex('.*/reload_config'), self._reload_config))
        dispatcher.add_handler(
            MessageHandler(Filters.regex('.*/closed_trades'), self._closed_trades))
        dispatcher.add_handler(MessageHandler(Filters.regex('.*/daily'), self._daily))
        dispatcher.add_handler(MessageHandler(Filters.regex('.*/help'), self._help))
        dispatcher.add_handler(ConversationHandler(
            entry_points=[MessageHandler(Filters.regex('.*/stop'), self._begin_stop)],
            states={
                1: [MessageHandler(Filters.regex('(graceful|freeze|shutdown|cancel)'),
                                   self._stop_mode_chosen)],
                2: [MessageHandler(Filters.regex('(confirm|abort)'), self._verify_stop_confirmation)],
            },
            fallbacks=[CommandHandler('cancel', self._abort)]
        ))
        dispatcher.add_handler(ConversationHandler(
            entry_points=[MessageHandler(Filters.regex('.*/set_leverage'), self._begin_set_leverage)],
            states={
                1: [MessageHandler(Filters.regex('([0-9]*|cancel)'), self._leverage_chosen)],
                2: [MessageHandler(Filters.regex('(confirm|abort)'), self._verify_leverage_confirmation)],
            },
            fallbacks=[CommandHandler('cancel', self._abort)]
        ))
        dispatcher.add_handler(MessageHandler(Filters.regex('.*/next'), self._next_page))
        dispatcher.add_handler(MessageHandler(Filters.regex('.*/previous'), self._previous_page))

    def _begin_set_leverage(self, update: Update, _: CallbackContext) -> int:
        self.stop_mode_requested = ''
        reply_keyboard = [['1', '3', '4'],
                          ['6', '7', '10'],
                          ['15', '20', 'cancel']]
        update.message.reply_text(
            text='To modify the leverage, please pick the desired leverage using the buttons below,'
                 'or type in the desired leverage yourself. Note that the maximum leverage that can'
                 f'be entered is {self._bot.max_leverage}, and that <b>this change is not persisted between restarts!</b>\n'
                 'Or send /cancel to abort modifying the leverage',
            parse_mode=ParseMode.HTML,
            reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
        )
        return 1

    def _leverage_chosen(self, update: Update, _: CallbackContext) -> int:
        if update.message.text == 'cancel':
            self.send_msg('Request for changing leverage was cancelled')
            return ConversationHandler.END

        try:
            self.leverage_chosen = int(update.message.text)
            if self.leverage_chosen < 1 or self.leverage_chosen > self._bot.max_leverage:
                self.send_msg(f'Invalid leverage provided. The leverage must be between 1 and {self._bot.max_leverage}')
                return ConversationHandler.END
        except:
            self.send_msg(f'Invalid leverage provided. The leverage must be between 1 and {self._bot.max_leverage}')
            return ConversationHandler.END

        reply_keyboard = [['confirm', 'abort']]
        update.message.reply_text(
            text=f'You have chosen to change the leverage to <pre>{update.message.text}</pre>.\n'
                 f'Please confirm that you want to activate this replying with either <pre>confirm</pre> or <pre>abort</pre>\n'
                 f'<b>Please be aware that this setting is not persisted between restarts!</b>',
            parse_mode=ParseMode.HTML,
            reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
        )
        return 2

    def _verify_leverage_confirmation(self, update: Update, _: CallbackContext) -> int:
        if update.message.text == 'confirm':
            async def _set_leverage_async():
                if self._bot.exchange == 'bybit':
                    return ''
                result = await self._bot.execute_leverage_change()
                self.send_msg(f'Result of leverage change: {result}')
            self._bot.set_config_value('leverage', self.leverage_chosen)
            task = self.loop.create_task(_set_leverage_async())
            task.add_done_callback(lambda fut: True) #ensures task is processed to prevent warning about not awaiting
            self.send_msg(
                f'Leverage set to {self.leverage_chosen} activated.\n'
                'Please be aware that this change is NOT persisted between restarts. To reset the leverage, you can use <pre>/reload_config</pre>')
        elif update.message.text == 'abort':
            self.leverage_chosen = ''
            self.send_msg(
                'Request for setting leverage was aborted')
        else:
            self.leverage_chosen = ''
            update.message.reply_text(text=f'Something went wrong, either <pre>confirm</pre> or <pre>abort</pre> was expected, but {update.message.text} was sent',
                                      parse_mode=ParseMode.HTML,
                                      reply_markup=self._keyboards[self._keyboard_idx])
        return ConversationHandler.END

    def _previous_page(self, update=None, context=None):
        if self._keyboard_idx > 0:
            self._keyboard_idx = self._keyboard_idx - 1
        self.send_msg('Previous')

    def _next_page(self, update=None, context=None):
        if self._keyboard_idx + 1 < len(self._keyboards):
            self._keyboard_idx = self._keyboard_idx + 1
        self.send_msg('Next')

    def _begin_stop(self, update: Update, _: CallbackContext) -> int:
        self.stop_mode_requested = ''
        reply_keyboard = [['graceful', 'freeze'],
                          ['shutdown', 'cancel']]
        update.message.reply_text(
            text='To stop the bot, please choose one of the following modes:\n'
            '<pre>graceful</pre>: prevents the bot from opening new positions, but completes the existing position as usual\n'
            '<pre>freeze</pre>: prevents the bot from opening positions, and cancels all open orders to open/reenter positions\n'
            '<pre>shutdown</pre>: immediately shuts down the bot, not making any further modifications to the current orders or positions\n'
            'Or send /cancel to abort stop-mode activation',
            parse_mode=ParseMode.HTML,
            reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
        )
        return 1

    def _stop_mode_chosen(self, update: Update, _: CallbackContext) -> int:
        if update.message.text == 'cancel':
            self.stop_mode_requested = ''
            self.send_msg('Request for activating stop-mode was cancelled')
            return ConversationHandler.END
        self.stop_mode_requested = update.message.text
        reply_keyboard = [['confirm', 'abort']]

        if self.stop_mode_requested == 'shutdown':
            msg = f'You have chosen to shut down the bot.\n' \
                  f'Please confirm that you want to activate this stop mode by replying with either <pre>confirm</pre> or <pre>abort</pre>\n' \
                  f'\U00002757<b>Be aware that you cannot restart or control the bot from telegram anymore after confirming!!</b>\U00002757'
        else:
            msg = f'You have chosen to activate the stop mode <pre>{update.message.text}</pre>\n' \
                  f'Please confirm that you want to activate this stop mode by replying with either <pre>confirm</pre> or <pre>abort</pre>'

        update.message.reply_text(
            text=msg,
            parse_mode=ParseMode.HTML,
            reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
        )
        return 2

    def _verify_stop_confirmation(self, update: Update, _: CallbackContext) -> int:
        if update.message.text == 'confirm':
            if self.stop_mode_requested == 'graceful':
                self._bot.set_config_value('do_long', False)
                self._bot.set_config_value('do_shrt', False)
                self.send_msg(
                    'Graceful stop mode activated. No longer opening new long or short positions, existing positions will still be managed.'
                    'Please be aware that this change is NOT persisted between restarts. To clear the stop-mode, you can use <pre>/reload_config</pre>')
            elif self.stop_mode_requested == 'freeze':
                self._bot.set_config_value('do_long', False)
                self._bot.set_config_value('do_shrt', False)
                self._bot.stop_mode = 'freeze'
                self.send_msg(
                    'Freeze stop mode activated. No longer opening new long or short positions, all orders for reentry will be cancelled.'
                    'Please be aware that this change is NOT persisted between restarts. To clear the stop-mode, you can use <pre>/reload_config</pre>')
            elif self.stop_mode_requested == 'shutdown':
                self._bot.stop_websocket = True
                self.send_msg(
                    'To restart the bot, you will need to manually start it again from a console.\n'
                    'Bot is being shut down...')
        elif update.message.text == 'abort':
            self.stop_mode_requested = ''
            self.send_msg(
                'Request for activating stop-mode was aborted')
        else:
            self.stop_mode_requested = ''
            update.message.reply_text(text=f'Something went wrong, either <pre>confirm</pre> or <pre>abort</pre> was expected, but {update.message.text} was sent',
                                      parse_mode=ParseMode.HTML,
                                      reply_markup=self._keyboards[self._keyboard_idx])
        return ConversationHandler.END

    def _abort(self, update: Update, _: CallbackContext) -> int:
        update.message.reply_text('Action aborted', reply_markup=self._keyboards[self._keyboard_idx])
        return ConversationHandler.END

    def _help(self, update=None, context=None):
        msg = '<pre><b>The following commands are available:</b></pre>\n' \
              '/balance: the equity & wallet balance in the configured account\n' \
              '/open_orders: a list of all buy & sell orders currently open\n' \
              '/stop: initiates a conversation via which the user can activate a stop mode\n' \
              '/position: information about the current position(s)\n' \
              '/show_config: the active configuration used\n' \
              '/reload_config: reload the configuration from disk, based on the file initially used\n' \
              "/closed_trades: a brief overview of bot's last 10 closed trades\n" \
              '/daily: an overview of daily profit\n' \
              '/set_leverage: initiates a conversion via which the user can modify the active leverage\n' \
              '/help: This help page\n'
        self.send_msg(msg)

    def _open_orders(self, update=None, context=None):
        open_orders = self._bot.open_orders
        order_table = PrettyTable(["Pos.", "Side", "Price", "Qty"])

        for order in open_orders:
            price = round_(order['price'], self._bot.price_step)
            qty = round_(order['qty'], self._bot.qty_step)
            side = order['side']
            position_side = order['position_side']
            order_table.add_row([position_side, side, price, qty])

        table_msg = order_table.get_string(sortby="Price", border=True, padding_width=1,
                                           junction_char=' ', vertical_char=' ', hrules=HEADER)
        msg = f'Current price: {round_(self._bot.price, self._bot.price_step)}\n' \
              f'<pre>{table_msg}</pre>'
        self.send_msg(msg)

    def _position(self, update=None, context=None):
        position_table = PrettyTable(['', 'Long', 'Short'])
        if 'long' in self._bot.position:
            long_position = self._bot.position['long']
            shrt_position = self._bot.position['shrt']

            position_table.add_row([f'Size', round_dynamic(long_position['size'], 3),
                                    round_dynamic(shrt_position['size'], 3)])
            position_table.add_row(['Price', round_dynamic(long_position['price'], 3),
                                    round_dynamic(shrt_position['price'], 3)])
            position_table.add_row(['Curr.price', round_dynamic(self._bot.price, 3),
                                    round_dynamic(self._bot.price, 3)])
            position_table.add_row(['Leverage', compress_float(long_position['leverage'], 3),
                                     compress_float(shrt_position['leverage'], 3)])
            position_table.add_row(['Liq.price', round_dynamic(long_position['liquidation_price'], 3),
                 round_dynamic(shrt_position['liquidation_price'], 3)])
            position_table.add_row(['Liq.diff.%', round_dynamic(float(long_position['liq_diff']) * 100, 3),
                 round_dynamic(float(shrt_position['liq_diff']) * 100, 3)])
            position_table.add_row([f'UPNL {self._bot.margin_coin}', round_dynamic(float(long_position['upnl']), 3),
                                    round_dynamic(float(shrt_position['upnl']), 3)])

            table_msg = position_table.get_string(border=True, padding_width=1,
                                                  junction_char=' ', vertical_char=' ',
                                                  hrules=HEADER)
            self.send_msg(f'<pre>{table_msg}</pre>')
        else:
            self.send_msg("Position not initialized yet, please try again later")

    def _balance(self, update=None, context=None):
        if bool(self._bot.position):
            async def _balance_async():
                position = await self._bot.fetch_position()
                msg = f'Balance {self._bot.margin_coin}\n' \
                      f'Wallet balance: {compress_float(position["wallet_balance"], 4)}\n' \
                      f'Equity: {compress_float(self._bot.position["equity"], 4)}\n' \
                      f'Locked margin: {compress_float(self._bot.position["used_margin"], 4)}\n' \
                      f'Available margin: {compress_float(self._bot.position["available_margin"], 4)}'
                self.send_msg(msg)

            self.send_msg('Retrieving balance...')
            task = self.loop.create_task(_balance_async())
            task.add_done_callback(lambda fut: True) #ensures task is processed to prevent warning about not awaiting
        else:
            self.send_msg('Balance not retrieved yet, please try again later')

    def _reload_config(self, update=None, context=None):
        if self.config_reload_ts > 0.0:
            if time() - self.config_reload_ts < 60 * 5:
                self.send_msg('Config reload in progress, please wait')
                return
        self.config_reload_ts = time()
        self.send_msg('Reloading config...')

        try:
            config = json.load(open(self._bot.live_config_path))
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
        if self._bot.exchange == 'binance':
            async def send_closed_trades_async():
                trades = await self._bot.fetch_fills(limit=100)
                closed_trades = [t for t in trades if t['realized_pnl'] != 0.0]
                closed_trades.sort(key=lambda x: x['timestamp'], reverse=True)

                table = PrettyTable(['Date', 'Pos.', 'Price', f'Pnl {self._bot.margin_coin}'])

                for trade in closed_trades[:self.n_trades]:
                    trade_date = datetime.fromtimestamp(trade['timestamp'] / 1000)
                    table.add_row(
                        [trade_date.strftime('%m-%d %H:%M'), trade['position_side'], round_(trade['price'], self._bot.price_step),
                         round_(trade['realized_pnl'], 0.01)])

                msg = f'Closed trades:\n' \
                      f'<pre>{table.get_string(border=True, padding_width=1, junction_char=" ", vertical_char=" ", hrules=HEADER)}</pre>'
                self.send_msg(msg)

            self.send_msg(f'Fetching last {self.n_trades} trades...')
            task = self.loop.create_task(send_closed_trades_async())
            task.add_done_callback(lambda fut: True) #ensures task is processed to prevent warning about not awaiting
        else:
            self.send_msg('This command is not supported (yet)')

    def _daily(self, update=None, context=None):
        if self._bot.exchange == 'binance':
            async def send_daily_async():
                nr_of_days = 7
                today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                daily = {}
                position = await self._bot.fetch_position()
                wallet_balance = position['wallet_balance']
                for idx, item in enumerate(range(0, nr_of_days)):
                    start_of_day = today - timedelta(days=idx)
                    end_of_day = start_of_day + timedelta(days=1)
                    start_time = int(start_of_day.timestamp()) * 1000
                    end_time = int(end_of_day.timestamp()) * 1000
                    daily_trades = await self._bot.fetch_income(start_time=start_time, end_time=end_time)
                    daily_trades = [trade for trade in daily_trades if trade['incomeType'] in ['REALIZED_PNL', 'FUNDING_FEE', 'COMMISSION']]
                    pln_summary = 0
                    for trade in daily_trades:
                        pln_summary += float(trade['income'])
                    daily[idx] = {}
                    daily[idx]['date'] = start_of_day.strftime('%m-%d')
                    daily[idx]['pnl'] = pln_summary

                table = PrettyTable(['Date\nMM-DD', 'PNL (%)'])
                pnl_sum = 0.0
                for item in daily.keys():
                    day_profit = daily[item]['pnl']
                    pnl_sum += day_profit
                    previous_day_close_wallet_balance = wallet_balance - day_profit
                    profit_pct = ((wallet_balance / previous_day_close_wallet_balance) - 1) * 100 \
                        if previous_day_close_wallet_balance > 0.0 else 0.0
                    wallet_balance = previous_day_close_wallet_balance
                    table.add_row([daily[item]['date'], f'{day_profit:.1f} ({profit_pct:.2f}%)'])

                pct_sum = ((position['wallet_balance'] + pnl_sum) / position['wallet_balance'] - 1) * 100 \
                    if position['wallet_balance'] > 0.0 else 0.0
                table.add_row(['-------','------------'])
                table.add_row(['Total', f'{round_dynamic(pnl_sum, 3)} ({round_(pct_sum, 0.01)}%)'])

                msg = f'<pre>{table.get_string(border=True, padding_width=1, junction_char=" ", vertical_char=" ", hrules=HEADER)}</pre>'
                self.send_msg(msg)

            self.send_msg('Calculating daily pnl...')
            task = self.loop.create_task(send_daily_async())
            task.add_done_callback(lambda fut: True) #ensures task is processed to prevent warning about not awaiting
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
              f'<pre>Symbol</pre>: {self._bot.symbol}\n' \
              f'<pre><b>Config:</b></pre> \n' \
              f'{json.dumps(self._bot.config, indent=4)}'
        self.send_msg(msg)

    def log_start(self):
        self.send_msg('<b>Passivbot started!</b>')

    def send_msg(self, msg: str):
        try:
            self._updater.bot.send_message(
                self._chat_id,
                text=msg,
                parse_mode=ParseMode.HTML,
                reply_markup=self._keyboards[self._keyboard_idx],
                disable_notification=False
            ).message_id
        except Exception as e:
            print(f'Failed to send telegram message: {e}')

    def exit(self):
        try:
            print("\nStopping telegram bot...")
            self._updater.stop()
            print("\nTelegram was stopped succesfully")
        except:
            print("\nFailed to shutdown telegram bot. Please make sure it is correctly terminated")
