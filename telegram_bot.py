import os
from procedures import load_live_config
from datetime import datetime, timedelta
from time import time
from pure_funcs import config_pretty_str
from typing import Optional

try:
    import git
except Exception as e:
    print(e)
    print("Unable to import git module. This is probably a result of a missing git installation."
          "As a result of this, the version will not be shown in applicable messages. To fix this,"
          "please make sure you have git installed properly. The bot will work fine without it.")
from prettytable import PrettyTable, HEADER
from telegram import KeyboardButton, ParseMode, ReplyKeyboardMarkup, Update, InlineKeyboardButton, \
    InlineKeyboardMarkup, CallbackQuery
from telegram.ext import Updater, CommandHandler, ConversationHandler, CallbackContext, \
    MessageHandler, Filters, CallbackQueryHandler

from njit_funcs import round_
from pure_funcs import compress_float, round_dynamic, denumpyize


class Telegram:
    def __init__(self, config: dict, bot, loop):
        self._bot = bot
        self.loop = loop
        self.config = config
        self._chat_id = config['chat_id']
        self._updater = Updater(token=config['token'])
        self.config_reload_ts = 0.0
        self.n_trades = 10

        first_keyboard_buttons = [
            [KeyboardButton('/daily \U0001F4B8'), KeyboardButton('/open_orders \U0001F4CB'), KeyboardButton('/position \U0001F4CC')],
            [KeyboardButton('/closed_trades \U0000274E'), KeyboardButton('/show_config \U0001F4DD')],
            [KeyboardButton('/balance \U0001F4B3'), KeyboardButton('/help \U00002753'), KeyboardButton('/next \U000023E9')]]
        second_keyboard_buttons = [
            [KeyboardButton('/reload_config \U0000267B'), KeyboardButton('/set_short \U0001F4C9'), KeyboardButton('/set_long \U0001F4C8')],
            [KeyboardButton('/transfer \U0001F3E6'), KeyboardButton('/set_profit_transfer \U0001F4DD'), KeyboardButton('/set_config \U0001F4C4')],
            [KeyboardButton('/previous \U000023EA'), KeyboardButton('/stop \U000026D4')]
        ]
        self._keyboard_idx = 0
        self._keyboards = [ReplyKeyboardMarkup(first_keyboard_buttons, resize_keyboard=True),
                           ReplyKeyboardMarkup(second_keyboard_buttons, resize_keyboard=True)]

        self.add_handlers(self._updater)
        self._updater.start_polling()

    def add_handlers(self, updater):
        dispatcher = updater.dispatcher
        dispatcher.add_handler(CommandHandler('balance', self._balance))
        dispatcher.add_handler(CommandHandler('open_orders', self._open_orders))
        dispatcher.add_handler(CommandHandler('position', self._position))
        dispatcher.add_handler(CommandHandler('show_config', self.show_config))
        dispatcher.add_handler(CommandHandler('reload_config', self._reload_config))
        dispatcher.add_handler(CommandHandler('closed_trades', self._closed_trades))
        dispatcher.add_handler(CommandHandler('daily', self._daily))
        dispatcher.add_handler(CommandHandler('help', self._help))
        dispatcher.add_handler(CommandHandler('next', self._next_page))
        dispatcher.add_handler(CommandHandler('previous', self._previous_page))

        dispatcher.add_handler(CallbackQueryHandler(self._balance, pattern='update_balance'))
        dispatcher.add_handler(CallbackQueryHandler(self._open_orders, pattern='update_open_orders'))
        dispatcher.add_handler(CallbackQueryHandler(self._position, pattern='update_position'))
        dispatcher.add_handler(CallbackQueryHandler(self._closed_trades, pattern='closed_trades'))
        dispatcher.add_handler(CallbackQueryHandler(self._daily, pattern='update_daily'))

        dispatcher.add_handler(ConversationHandler(
            entry_points=[MessageHandler(Filters.regex('/stop.*'), self._begin_stop)],
            states={
                1: [MessageHandler(Filters.regex('(graceful|freeze|shutdown|panic|manual|resume|cancel)'),
                                   self._stop_mode_chosen)],
                2: [MessageHandler(Filters.regex('(confirm|abort)'), self._verify_stop_confirmation)]
            },
            fallbacks=[CommandHandler('cancel', self._abort)]
        ))
        dispatcher.add_handler(ConversationHandler(
            entry_points=[MessageHandler(Filters.regex('/set_short.*'), self._verify_set_short)],
            states={
                1: [MessageHandler(Filters.regex('(confirm|abort)'), self._verify_short_confirmation)]
            },
            fallbacks=[CommandHandler('cancel', self._abort)]
        ))
        dispatcher.add_handler(ConversationHandler(
            entry_points=[MessageHandler(Filters.regex('/set_long.*'), self._verify_set_long)],
            states={
                1: [MessageHandler(Filters.regex('(confirm|abort)'), self._verify_long_confirmation)]
            },
            fallbacks=[CommandHandler('cancel', self._abort)]
        ))
        dispatcher.add_handler(ConversationHandler(
            entry_points=[MessageHandler(Filters.regex('/set_config.*'), self._begin_set_config)],
            states={
                1: [CallbackQueryHandler(self._configfile_chosen)],
                2: [CallbackQueryHandler(self._verify_setconfig_confirmation)]
            },
            fallbacks=[CommandHandler('cancel', self._abort)]
        ))
        dispatcher.add_handler(ConversationHandler(
            entry_points=[MessageHandler(Filters.regex('/transfer.*'), self._begin_transfer)],
            states={
                1: [CallbackQueryHandler(self._transfer_type_chosen)],
                2: [MessageHandler(Filters.regex('(/[0-9\\.]*)|cancel'), self._transfer_amount_chosen)],
                3: [MessageHandler(Filters.regex('(confirm|abort)'), self._verify_transfer_confirmation)]
            },
            fallbacks=[MessageHandler(Filters.command, self._abort)]
        ))

        dispatcher.add_handler(ConversationHandler(
            entry_points=[MessageHandler(Filters.regex('/set_profit_transfer.*'), self._begin_set_profit_transfer)],
            states={
                1: [MessageHandler(Filters.regex('([0-9]*|cancel)'), self._profit_transfer_chosen)],
                2: [MessageHandler(Filters.regex('(confirm|abort)'), self._verify_profit_transfer_confirmation)]
            },
            fallbacks=[CommandHandler('cancel', self._abort)]
        ))

    def _begin_transfer(self, update: Update, _: CallbackContext) -> int:
        if self._bot.exchange == 'bybit':
            self.send_msg('This command is not supported (yet)')
            return

        self.transfer_type = None
        self.transfer_amount = 0

        buttons = [
            [InlineKeyboardButton('Spot -> USD-M Futures', callback_data='MAIN_UMFUTURE'),
             InlineKeyboardButton('USD-M Futures -> Spot', callback_data='UMFUTURE_MAIN'),
             InlineKeyboardButton('cancel', callback_data='cancel')]
        ]
        update.message.reply_text(
            text='You have chosen to transfer balance in your account. Please select the type of transfer you want to perform:',
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(buttons)
        )

        return 1

    def _transfer_type_chosen(self, update=None, context=None) -> int:
        query = update.callback_query
        query.answer()

        self.transfer_type = query.data
        if self.transfer_type == 'MAIN_UMFUTURE':
            text = 'You have chosen to transfer funds from your Spot wallet to your USD-M Futures wallet.\n' \
                   'Please specify the amount of USDT you want to transfer (<pre>prefixed with / if privacy mode is not disabled</pre>):'
        elif self.transfer_type == 'UMFUTURE_MAIN':
            text = 'You have chosen to transfer funds from your USD-M Futures wallet to your Spot wallet.\n' \
                   'Please specify the amount of USDT you want to transfer (<pre>prefixed with / if privacy mode is not disabled</pre>):'
        elif self.transfer_type == 'cancel':
            update.effective_message.reply_text('Action aborted', reply_markup=self._keyboards[self._keyboard_idx])
            return ConversationHandler.END

        update.effective_message.reply_text(text=text, parse_mode=ParseMode.HTML, reply_markup=ReplyKeyboardMarkup([['cancel']]))
        return 2

    def _transfer_amount_chosen(self, update=None, context=None) -> int:
        input = update.effective_message.text
        if input == 'cancel':
            update.effective_message.reply_text('Action aborted', reply_markup=self._keyboards[self._keyboard_idx])
            return ConversationHandler.END

        self.transfer_amount = float(input.replace('/',''))

        if self.transfer_type == 'MAIN_UMFUTURE':
            text = f'You have chosen to transfer {self.transfer_amount} from your Spot wallet to your USD-M Futures wallet.\n' \
                   'Please confirm that you want to execute this transfer.\n' \
                   'Please be aware that this can have influence on open position(s)!'
        elif self.transfer_type == 'UMFUTURE_MAIN':
            text = f'You have chosen to transfer {self.transfer_amount} from your USD-M Futures wallet to your Spot wallet.\n' \
                   'Please confirm that you want to execute this transfer.\n' \
                   'Please be aware that this can have influence on open position(s)!'

        reply_keyboard = [['confirm', 'abort']]
        update.message.reply_text(
            text=text,
            parse_mode=ParseMode.HTML,
            reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
        )
        return 3

    def _verify_transfer_confirmation(self, update: Update, _: CallbackContext) -> int:
        answer = update.effective_message.text

        if answer not in ['confirm', 'abort']:
            return 3
        elif answer == 'abort':
            self.send_msg(f'Request for transfer aborted')
            return ConversationHandler.END

        async def _transfer_wallet(type_: str, amount: float):
            result = await self._bot.transfer(type_=self.transfer_type, amount=self.transfer_amount, asset='USDT')
            if 'code' in result:
                self.send_msg(f'{result["msg"]}')
            else:
                self.send_msg(f'Transferred {amount} using type {type_}')
        task = self.loop.create_task(_transfer_wallet(self.transfer_type, self.transfer_amount))
        task.add_done_callback(lambda fut: True) #ensures task is processed to prevent warning about not awaiting
        self.send_msg(f'Transferring...')

        return ConversationHandler.END

    def _begin_set_config(self, update: Update, _: CallbackContext) -> int:
        files = sorted([f for f in os.listdir('configs/live') if f.endswith('.json')])
        buttons = []
        for file in files:
            buttons.append([InlineKeyboardButton(file, callback_data=file)])

        update.message.reply_text(
            text='Please select one of the available config files to load:',
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(buttons)
        )
        return 1

    def _configfile_chosen(self, update: Update, _: CallbackContext) -> int:
        query = update.callback_query
        self.config_file_to_activate = query.data
        query.answer()

        keyboard = [[InlineKeyboardButton('confirm', callback_data='confirm')],
                    [InlineKeyboardButton('abort', callback_data='abort')]]

        query.edit_message_text(text=f'You have chosen to change the active the config file <pre>configs/live/{query.data}</pre>.\n'
                                f'Please confirm that you want to activate this by replying with either <pre>confirm</pre> or <pre>abort</pre>',
                                parse_mode=ParseMode.HTML,
                                reply_markup=InlineKeyboardMarkup(keyboard))
        return 2

    def _verify_setconfig_confirmation(self, update: Update, _: CallbackContext) -> int:
        query = update.callback_query
        query.answer()
        answer = query.data

        if answer not in ['confirm', 'abort']:
            return 2

        if answer == 'abort':
            self.send_msg(f'Request for setting config was aborted')
            return ConversationHandler.END

        if self.config_reload_ts > 0.0 and time() - self.config_reload_ts < 60 * 5:
            self.send_msg('Config reload in progress, please wait')
            return
        self.config_reload_ts = time()
        self.send_msg(f'Activating config file <pre>configs/live/{self.config_file_to_activate}</pre>...')
        self._activate_config(f'configs/live/{self.config_file_to_activate}')
        self.config_file_to_activate = None
        return ConversationHandler.END

    def _verify_set_short(self, update: Update, _: CallbackContext) -> int:
        reply_keyboard = [['confirm', 'abort']]
        update.message.reply_text(
            text=f'Shorting is currently <pre>{"enabled" if self._bot.do_shrt is True else "disabled"}</pre>.\n'
                 f'You have chosen to <pre>{"disable" if self._bot.do_shrt is True else "enable"}</pre> shorting.\n'
                 f'Please confirm that you want to change this by replying with either <pre>confirm</pre> or <pre>abort</pre>\n'
                 f'<b>Please be aware that this setting is not persisted between restarts!</b>',
            parse_mode=ParseMode.HTML,
            reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
        )
        return 1

    def _verify_short_confirmation(self, update: Update, _: CallbackContext) -> int:
        if update.message.text == 'confirm':
            self._bot.set_config_value('do_shrt', not self._bot.do_shrt)
            self.send_msg(
                f'Shorting is now <pre>{"enabled" if self._bot.do_shrt is True else "disabled"}</pre>.\n'
                'Please be aware that this change is NOT persisted between restarts.')
        elif update.message.text == 'abort':
            self.send_msg(f'Request for {"disabling" if self._bot.do_shrt is True else "enabling"} shorting was aborted')
        else:
            update.message.reply_text(text=f'Something went wrong, either <pre>confirm</pre> or <pre>abort</pre> was expected, but {update.message.text} was sent',
                                      parse_mode=ParseMode.HTML,
                                      reply_markup=self._keyboards[self._keyboard_idx])
        return ConversationHandler.END

    def _verify_set_long(self, update: Update, _: CallbackContext) -> int:
        reply_keyboard = [['confirm', 'abort']]
        update.message.reply_text(
            text=f'Long is currently <pre>{"enabled" if self._bot.do_long is True else "disabled"}</pre>.\n'
                 f'You have chosen to <pre>{"disable" if self._bot.do_long is True else "enable"}</pre> long.\n'
                 f'Please confirm that you want to change this by replying with either <pre>confirm</pre> or <pre>abort</pre>\n'
                 f'<b>Please be aware that this setting is not persisted between restarts!</b>',
            parse_mode=ParseMode.HTML,
            reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
        )
        return 1

    def _verify_long_confirmation(self, update: Update, _: CallbackContext) -> int:
        if update.message.text == 'confirm':
            self._bot.set_config_value('do_long', not self._bot.do_long)
            self.send_msg(
                f'Long is now <pre>{"enabled" if self._bot.do_shrt is True else "disabled"}</pre>.\n'
                'Please be aware that this change is NOT persisted between restarts.')
        elif update.message.text == 'abort':
            self.send_msg(f'Request for {"disabling" if self._bot.do_long is True else "enabling"} long was aborted')
        else:
            update.message.reply_text(text=f'Something went wrong, either <pre>confirm</pre> or <pre>abort</pre> was expected, but {update.message.text} was sent',
                                      parse_mode=ParseMode.HTML,
                                      reply_markup=self._keyboards[self._keyboard_idx])
        return ConversationHandler.END

    def _begin_set_profit_transfer(self, update: Update, _: CallbackContext) -> int:
        self.profit_transfer_pct_chosen = None
        reply_keyboard = [['0.0', '0.2', '0.25'],
                          ['0.3', '0.4', '0.5'],
                          ['0.75', '1', 'cancel']]
        update.message.reply_text(
            text='To modify the profit transfer percentage, please pick the desired amount using the buttons below,'
                 'or type in the desired profit transfer amount yourself (value between 0 and 1).\n'
                 'Setting the value to 0 disables profit transfer, and 1 transfers all profit.\n'
                 'Note that the that <b>this change is not persisted between restarts!</b>\n'
                 'Or send /cancel to abort modifying the profit transfer',
            parse_mode=ParseMode.HTML,
            reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
        )
        return 1

    def _profit_transfer_chosen(self, update: Update, _: CallbackContext) -> int:
        if update.message.text == 'cancel':
            self.send_msg('Request for changing profit transfer was cancelled')
            return ConversationHandler.END

        try:
            self.profit_transfer_pct_chosen = float(update.message.text)
            if self.profit_transfer_pct_chosen < 0 or self.profit_transfer_pct_chosen > 1:
                self.send_msg(f'Invalid profit transfer percentage provided. The value for the profit transfer must be a value between 0 and 1 (inclusive), indicating a range of 0% to 100%')
                return ConversationHandler.END
        except:
            self.send_msg(f'Invalid profit transfer percentage provided. The value must be between 0 and 1. Aborting conversation.')
            return ConversationHandler.END

        reply_keyboard = [['confirm', 'abort']]
        update.message.reply_text(
            text=f'You have chosen to change the profit transfer percentage to <pre>{self.profit_transfer_pct_chosen}</pre>.\n'
                 f'Please confirm that you want to activate this by replying with either <pre>confirm</pre> or <pre>abort</pre>\n'
                 f'<b>Please be aware that this setting is not persisted between restarts!</b>',
            parse_mode=ParseMode.HTML,
            reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
        )
        return 2

    def _verify_profit_transfer_confirmation(self, update: Update, _: CallbackContext) -> int:
        if update.message.text == 'confirm':
            self._bot.set_config_value('profit_trans_pct', self.profit_transfer_pct_chosen)
            self.send_msg(
                f'Profit transfer percentage set to {self.profit_transfer_pct_chosen}.\n'
                'Please be aware that this change is NOT persisted between restarts. To reset the profit transfer percentage, you can use <pre>/reload_config</pre>')
        elif update.message.text == 'abort':
            self.profit_transfer_pct_chosen = None
            self.send_msg('Request for setting profit transfer amount was aborted')
        else:
            self.profit_transfer_pct_chosen = None
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
        reply_keyboard = [['graceful', 'freeze', 'panic'],
                          ['shutdown', 'manual', 'resume'],
                          ['cancel']]
        update.message.reply_text(
            text='To stop the bot, please choose one of the following modes:\n'
            '<pre>graceful</pre>: prevents the bot from opening new positions, but completes the existing position as usual\n'
            '<pre>freeze</pre>: prevents the bot from opening positions, and cancels all open orders to open/reenter positions\n'
            '<pre>panic</pre>: immediately closes all open positions against market price, and cancels all open orders to open/reenter positions\n'
            '<pre>manual</pre>: immediately stops automatic order creation & cancelling, and effectively disables the bot to stop doing anything on the exchange\n'
            '<pre>shutdown</pre>: immediately shuts down the bot, not making any further modifications to the current orders or positions\n'
            '<pre>resume</pre>: clears the stop mode and resumes normal operation\n'
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
        elif self.stop_mode_requested == 'panic':
            msg = f'You have chosen to activate the stop mode panic.\n' \
                  f'Please confirm that you want to activate this stop mode by replying with either <pre>confirm</pre> or <pre>abort</pre>\n' \
                  f'<b>\U00002757\U00002757\U00002757Be aware that this will actively close all open positions, EVEN IF THEY ARE IN LOSS\U00002757\U00002757\U00002757</b>'
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
                self.previous_do_long = self._bot.do_long
                self.previous_do_shrt = self._bot.do_shrt
                self._bot.set_config_value('do_long', False)
                self._bot.set_config_value('do_shrt', False)
                self._bot.stop_mode = 'graceful'
                self.send_msg(
                    'Graceful stop mode activated. No longer opening new long or short positions, existing positions will still be managed.'
                    'Please be aware that this change is NOT persisted between restarts. To clear the stop-mode, you can use <pre>/reload_config</pre> or select <pre>resume</pre> from the <pre>/stop</pre> action')
            elif self.stop_mode_requested == 'freeze':
                self.previous_do_long = self._bot.do_long
                self.previous_do_shrt = self._bot.do_shrt
                self._bot.set_config_value('do_long', False)
                self._bot.set_config_value('do_shrt', False)
                self._bot.stop_mode = 'freeze'
                self.send_msg(
                    'Freeze stop mode activated. No longer opening new long or short positions, all orders for reentry will be cancelled.'
                    'Please be aware that this change is NOT persisted between restarts. To clear the stop-mode, you can use <pre>/reload_config</pre> or select <pre>resume</pre> from the <pre>/stop</pre> action')
            if self.stop_mode_requested == 'panic':
                self.previous_do_long = self._bot.do_long
                self.previous_do_shrt = self._bot.do_shrt
                self._bot.set_config_value('do_long', False)
                self._bot.set_config_value('do_shrt', False)
                self._bot.stop_mode = 'panic'
                self.send_msg(
                    'Panic stop mode activated. No longer opening new long or short positions, existing positions will immediately be closed.'
                    'Please be aware that this change is NOT persisted between restarts. To clear the stop-mode, you can use <pre>/reload_config</pre> or select <pre>resume</pre> from the <pre>/stop</pre> action')
            if self.stop_mode_requested == 'manual':
                self._bot.stop_mode = 'manual'
                self.send_msg(
                    'Manual stop mode activated. No longer creating or cancelling orders on the exchange.\n'
                    'Please be aware that this change is NOT persisted between restarts. To clear the stop-mode, you can use <pre>/reload_config</pre> or select <pre>resume</pre> from the <pre>/stop</pre> action')
            elif self.stop_mode_requested == 'shutdown':
                self._bot.stop_websocket = True
                self.send_msg(
                    'To restart the bot, you will need to manually start it again from a console.\n'
                    'Bot is being shut down...')
            elif self.stop_mode_requested == 'resume':
                if hasattr(self, 'previous_do_long'):
                    self._bot.set_config_value('do_long', self.previous_do_long)
                if hasattr(self, 'previous_do_shrt'):
                    self._bot.set_config_value('do_shrt', self.previous_do_shrt)
                self._bot.stop_mode = None
                self.send_msg('Normal operation resumed')
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
              '/daily [days]: an overview of daily profit, defaulting to 7 days\n' \
              '/set_short: initiates a conversion via which the user can enable/disable shorting\n' \
              '/set_long: initiates a conversion via which the user can enable/disable long\n' \
              '/set_config: initiates a conversion via which the user can switch to a different configuration file\n' \
              '/help: This help page'
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
        self.send_msg(msg, refreshable=True, callback_path='update_open_orders', query=update.callback_query)

    def _position(self, update=None, context=None):
        position_table = PrettyTable(['', 'Long', 'Short'])
        if 'long' in self._bot.position:
            long_position = self._bot.position['long']
            shrt_position = self._bot.position['shrt']
            closest_long_price = min([o['price'] for o in self._bot.open_orders if o['position_side'] == 'long' and o['side'] == 'sell'] or [0])
            closest_shrt_price = max([o['price'] for o in self._bot.open_orders if o['position_side'] == 'shrt' and o['side'] == 'buy'] or [0])

            position_table.add_row([f'Size', round_dynamic(long_position['size'], 3),
                                    round_dynamic(shrt_position['size'], 3)])
            position_table.add_row(['Entry price', round_dynamic(long_position['price'], 3),
                                    round_dynamic(shrt_position['price'], 3)])
            position_table.add_row(['Curr.price', round_(self._bot.price, self._bot.price_step),
                                    round_(self._bot.price, self._bot.price_step)])
            position_table.add_row(['Close price', round_(closest_long_price, self._bot.price_step),
                                    round_(closest_shrt_price, self._bot.price_step)])
            position_table.add_row(['Cost/balance', round_dynamic(float(long_position['pbr']), 3), round_dynamic(float(shrt_position['pbr']), 3)])
            position_table.add_row(['Liq.price', round_dynamic(long_position['liquidation_price'], 3),
                 round_dynamic(shrt_position['liquidation_price'], 3)])
            position_table.add_row(['Liq.diff.%', round_dynamic(float(long_position['liq_diff']) * 100, 3),
                 round_dynamic(float(shrt_position['liq_diff']) * 100, 3)])
            position_table.add_row([f'UPNL {self._bot.margin_coin if hasattr(self._bot, "margin_coin") else ""}', round_dynamic(float(long_position['upnl']), 3),
                                    round_dynamic(float(shrt_position['upnl']), 3)])

            table_msg = position_table.get_string(border=True, padding_width=1,
                                                  junction_char=' ', vertical_char=' ',
                                                  hrules=HEADER)
            self.send_msg(f'<pre>{table_msg}</pre>', refreshable=True, callback_path='update_position', query=update.callback_query)
        else:
            self.send_msg("Position not initialized yet, please try again later")

    def _balance(self, update=None, context=None):
        if bool(self._bot.position):
            async def _balance_async():
                position = await self._bot.fetch_position()
                account = await self._bot.fetch_account()
                usdt_balance = list(asset for asset in account['balances'] if asset['asset'] == 'USDT')[0]

                msg = f'Futures balance {self._bot.margin_coin if hasattr(self._bot, "margin_coin") else ""}:\n' \
                      f'Wallet balance: {compress_float(position["wallet_balance"], 4)}\n' \
                      f'Equity: {compress_float(self._bot.position["equity"], 4)}\n' \
                      f'Locked margin: {compress_float(self._bot.position["used_margin"], 4)}\n' \
                      f'Available margin: {compress_float(self._bot.position["available_margin"], 4)}\n\n' \
                      f'Spot balance:\n' \
                      f'USDT: {compress_float(float(usdt_balance["free"]) + float(usdt_balance["locked"]), 4)} ({compress_float(float(usdt_balance["locked"]), 4)} locked)'
                self.send_msg(msg, refreshable=True, callback_path='update_balance', query=update.callback_query)

            self.send_msg('Retrieving balance...')
            task = self.loop.create_task(_balance_async())
            task.add_done_callback(lambda fut: True) #ensures task is processed to prevent warning about not awaiting
        else:
            self.send_msg('Balance not retrieved yet, please try again later')

    def _reload_config(self, update=None, context=None):
        if self.config_reload_ts > 0.0 and time() - self.config_reload_ts < 60 * 5:
            self.send_msg('Config reload in progress, please wait')
            return
        self.config_reload_ts = time()
        self.send_msg('Reloading config...')

        self._activate_config(self._bot.live_config_path)

    def _activate_config(self, config_path):
        try:
            config = load_live_config(config_path)
        except Exception:
            self.send_msg(f"Failed to load config file {config_path}")
            self.config_reload_ts = 0.0
            return

        self._bot.pause()
        self._bot.stop_mode = None
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
                self.send_msg(msg, refreshable=True, callback_path='closed_trades', query=update.callback_query)

            self.send_msg(f'Fetching last {self.n_trades} trades...')
            task = self.loop.create_task(send_closed_trades_async())
            task.add_done_callback(lambda fut: True) #ensures task is processed to prevent warning about not awaiting
        else:
            self.send_msg('This command is not supported (yet)')

    def _daily(self, update=None, context=None):
        if self._bot.exchange == 'binance':
            async def send_daily_async(nr_of_days:int):
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

                bal_minus_pnl = position['wallet_balance'] - pnl_sum
                pct_sum = (position['wallet_balance'] / bal_minus_pnl - 1) * 100 if bal_minus_pnl > 0.0 else 0.0
                table.add_row(['-------','------------'])
                table.add_row(['Total', f'{round_dynamic(pnl_sum, 3)} ({round_(pct_sum, 0.01)}%)'])

                msg = f'<pre>{table.get_string(border=True, padding_width=1, junction_char=" ", vertical_char=" ", hrules=HEADER)}</pre>'
                self.send_msg(msg, refreshable=True, callback_path='update_daily', query=update.callback_query)

            self.send_msg('Calculating daily pnl...')
            try:
                nr_of_days = int(context.args[0])
            except:
                nr_of_days = 7
            task = self.loop.create_task(send_daily_async(nr_of_days))
            task.add_done_callback(lambda fut: True) #ensures task is processed to prevent warning about not awaiting
        else:
            self.send_msg('This command is not supported (yet) on Bybit')

    def notify_entry_order_filled(self, position_side: str, qty: float, fee: float, price: float, total_size: float):
        if 'notify_entry_fill' not in self.config or self.config['notify_entry_fill'] is True:
            icon = "\U0001F535"
            self.send_msg(f'<b>{icon} {self._bot.exchange.capitalize()} {self._bot.pair}</b> Opened {position_side}\n'
                          f'<b>Amount: </b><pre>{round_(qty, self._bot.qty_step)}</pre>\n'
                          f'<b>Total size: </b><pre>{round_(total_size, self._bot.qty_step)}</pre>\n'
                          f'<b>Price: </b><pre>{round_(price, self._bot.price_step)}</pre>\n'
                          f'<b>Fee: </b><pre>{round_(fee, self._bot.price_step)} {self._bot.margin_coin} ({round_(fee/(qty * price) * 100, self._bot.price_step)}%)</pre>')

    def notify_close_order_filled(self, realized_pnl: float, position_side: str, qty: float, fee: float, wallet_balance: float, remaining_size: float, price: float):
        if 'notify_close_fill' not in self.config or self.config['notify_close_fill'] is True:
            icon = "\U00002705" if realized_pnl >= 0 else "\U0000274C"
            self.send_msg(f'<b>{icon} {self._bot.exchange.capitalize()} {self._bot.pair}</b> Closed {position_side}\n'
                f'<b>PNL: </b><pre>{round_(realized_pnl, self._bot.price_step)} {self._bot.margin_coin} ({round_(realized_pnl/wallet_balance * 100, self._bot.price_step)}%)</pre>\n'
                f'<b>Amount: </b><pre>{round_(qty, self._bot.qty_step)}</pre>\n'
                f'<b>Remaining size: </b><pre>{round_(remaining_size, self._bot.qty_step)}</pre>\n'
                f'<b>Price: </b><pre>{round_(price, self._bot.price_step)}</pre>\n'
                f'<b>Fee: </b><pre>{round_(fee, self._bot.price_step)} {self._bot.margin_coin} ({round_(fee/realized_pnl * 100, self._bot.price_step)}%)</pre>')

    def show_config(self, update=None, context=None):
        try:
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha
            sha_short = repo.git.rev_parse(sha, short=True)
        except:
            sha_short = 'UNKNOWN'

        cfg = denumpyize(self._bot.config)
        long_cfg = cfg.pop('long', None)
        shrt_cfg = cfg.pop('shrt', None)

        self.send_msg(f'<pre><b>Version:</b></pre> {sha_short},\n' \
                      f'<pre>Symbol</pre>: {self._bot.symbol}\n' \
                      f'<pre><b>Config:</b></pre> \n' \
                      f'{config_pretty_str(cfg)}')
        self.send_msg(f'<pre><b>Short</b></pre>:\n{config_pretty_str(shrt_cfg)}')
        self.send_msg(f'<pre><b>Long</b></pre>:\n{config_pretty_str(long_cfg)}')

    def log_start(self):
        self.send_msg('<b>Passivbot started!</b>')

    def send_msg(self, msg: str, refreshable: bool = False, callback_path: str = "", query: Optional[CallbackQuery] = None):
        try:
            if query:
                self._update_msg(query=query, msg=msg, callback_path=callback_path, refreshable=refreshable)
                return

            if refreshable:
                reply_markup = InlineKeyboardMarkup([[InlineKeyboardButton("Refresh", callback_data=callback_path)]])
            else:
                reply_markup = self._keyboards[self._keyboard_idx]

            self._updater.bot.send_message(
                self._chat_id,
                text=msg,
                parse_mode=ParseMode.HTML,
                reply_markup=reply_markup,
                disable_notification=False
            )
        except Exception as e:
            print(f'Error sending telegram message: {e}')
            try:
                self._updater.bot.send_message(
                    self._chat_id,
                    text=f'Error sending message: {e}',
                    parse_mode=ParseMode.HTML,
                    reply_markup=self._keyboards[self._keyboard_idx],
                    disable_notification=False
                )
            except Exception as fe:
                print(f'Failed to send error message: {fe}')

    def _update_msg(self, query: CallbackQuery, msg: str, callback_path: str = "",
                    refreshable: bool = False) -> None:
        if refreshable:
            reply_markup = InlineKeyboardMarkup([
                [InlineKeyboardButton("Refresh", callback_data=callback_path)],
            ])
        else:
            reply_markup = InlineKeyboardMarkup([[]])
        msg += "\n\nUpdated: {}".format(datetime.now().ctime())
        if not query.message:
            return
        chat_id = query.message.chat_id
        message_id = query.message.message_id

        try:
            self._updater.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=msg,
                parse_mode=ParseMode.HTML,
                reply_markup=reply_markup
            )
        except Exception as fe:
            print(f'Failed to send error message: {fe}')

    def exit(self):
        try:
            print("\nStopping telegram bot...")
            self._updater.stop()
            print("\nTelegram was stopped succesfully")
        except:
            print("\nFailed to shutdown telegram bot. Please make sure it is correctly terminated")
