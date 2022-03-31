from email import message
from pybit import HTTP  # supports inverse perp & futures, usdt perp, spot.

import os
import hjson
from tabulate import tabulate
import json
from datetime import datetime, date
import pandas as pd
import plotly.express as px
import discord
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from PIL import Image
from functions.functions import get_pro_channel_enabled, send_slack_message


def fill_calculation(json_base):
    # build data % if not present in the JSON file
    the_i = 0
    old_pnl_realized = 0
    old_equity = 0
    for key in json_base:
        if the_i == 0 :
            json_base[key]['gain_in_$'] = 0
            json_base[key]['daily_gain_pct'] = 0
        else:
            data = json_base[key]
            if not 'gain_in_$' in data:
                json_base[key]['gain_in_$'] = float(json_base[key]['cum_realised_pnl']) - old_pnl_realized
            if not 'daily_gain_pct' in data :
                json_base[key]['daily_gain_pct'] = (float(json_base[key]['gain_in_$']) / old_equity) * 100
        
        old_pnl_realized = float(json_base[key]['cum_realised_pnl'])
        old_equity = float(json_base[key]['equity'])

        the_i = the_i + 1
    return json_base


async def wallet(message):


    a_message = message.content.split(' ')
    if len(a_message) < 2:
        await message.channel.send("Mauvais usage. Ex : !w tedy")
        await message.channel.send("Mauvais usage. Ex : !w tedy chart")
        return 

    api_keys_user = "bybit_tedy"
    user_name = a_message[1]
    if a_message[1] == "tedy": 
        api_keys_user = "bybit_tedy"
    elif a_message[1] == "jojo":
        api_keys_user = "bybit_jojo"
    elif (a_message[1] == "pro") and (message.channel.id in get_pro_channel_enabled()):
        api_keys_user = "bybit_pro" 
    else:
        await message.channel.send("Mauvais user.")
        return 

    print_chart = False
    from_auto_bot = False
    if (len(a_message) >= 3):
        if (a_message[2] == 'from_auto_bot_x15'):
            from_auto_bot = True
        if (a_message[2] == 'chart'):
            print_chart = True



    api_keys_file = "../../api-keys.json"
    coin_ballance = "USDT"

    def get_equity_and_pnl(api_keys_file, api_keys_user, coin_ballance):

        keys = ""
        if os.path.exists(api_keys_file) :
            keys = hjson.load(open(api_keys_file, encoding="utf-8"))
        else:
            return {'error' : 'Problem loading keys'}

        
        session_auth = HTTP(
            endpoint="https://api.bybit.com",
            api_key=keys[api_keys_user]['key'],
            api_secret=keys[api_keys_user]['secret']
        )

        result = session_auth.get_wallet_balance(coin=coin_ballance)
        print(result)
        
        positions = session_auth.my_position(endpoint='/private/linear/position/list')


        # print(json.dumps(positions['result'][0]['data']))
        # print(json.dumps(positions['result'][1]['data']))
        total_position = 0
        for key in positions['result']:
            total_position += key['data']['position_value']

        if result['ret_code'] != 0:
            return {'error' : 'bad ret_code'}
        if result['ret_msg'] != "OK":
            return {'error' : 'bad ret_code'}

        return {
                    'error'  : '',
                    'equity' : float('%.4f'%(result['result'][coin_ballance]['equity'])),
                    'cum_realised_pnl' : float('%.4f'%(result['result'][coin_ballance]['cum_realised_pnl'])),
                    'used_margin' : float('%.4f'%(result['result'][coin_ballance]['used_margin'])),
                    'total_position' : total_position
        }
        
        
        


    wallet_data = get_equity_and_pnl(api_keys_file, api_keys_user, coin_ballance)

    discord_message_to_send = ""

    # JSON reading
    json_file = 'tmp/data_' + user_name + '.json'
    json_base = {}
    if os.path.exists(json_file) :
        json_base = hjson.load(open(json_file, encoding="utf-8"))

    # JSON adding
    today = date.today().strftime('%d/%m/%Y1')
    json_base[today] = {}
    json_base[today]['equity'] = wallet_data['equity']
    json_base[today]['cum_realised_pnl'] = wallet_data['cum_realised_pnl']
    json_base[today]['date'] = today
    json_base[today]['used_margin'] = wallet_data['used_margin']
    json_base[today]['total_position'] = wallet_data['total_position']
    json_base[today]['risk'] = 100*wallet_data['used_margin']/wallet_data['equity']

    json_base = fill_calculation(json_base)

    previous = json_base[list(json_base.keys())[-2]]

    now_data = json_base[today]

    def print_el(now, previous, key, symbol="$", show_icon=False):

        icon = ""
        if show_icon:
            if previous[key] < now[key]:
                icon = ":arrow_upper_right:"
            else:
                icon = ":arrow_lower_right:"
        else:
            if previous[key] < now[key]:
                icon = "^"
            else:
                icon = "v"

        return (str(round(number=now[key], ndigits=2)) + symbol).replace('.', ',') + icon
        



    if wallet_data['error'] == "":
        colonne = 20
        message_content = \
        "" + user_name.upper() + " | Positions : " + print_el(now_data, previous, 'total_position', show_icon=True) + " | Margin used : " + print_el(now_data, previous, 'used_margin', show_icon=True) + " | Risk : " + str(round(now_data['risk'])) + "%\n" + \
        "```" + \
        "Equity".ljust(colonne)                                     +   "Tot. Rea. PNL".ljust(colonne) + "\n" + \
        print_el(now_data, previous, 'equity').ljust(colonne)       +   print_el(now_data, previous, 'cum_realised_pnl').ljust(colonne) + "\n" + \
        "\n" + \
        "Daily Gain".ljust(colonne)                                 +   "Daily gain".ljust(colonne) + "\n" + \
        print_el(now_data, previous, 'gain_in_$').ljust(colonne)    +   print_el(now_data, previous, 'daily_gain_pct', '%').ljust(colonne) + \
        "```"
        discord_message_to_send = message_content
    else:
        await message.channel.send("Problem :"+wallet_data['error'])
        return

    if not (from_auto_bot or print_chart):
         await message.channel.send(discord_message_to_send)
    elif (from_auto_bot or print_chart):


        df = pd.DataFrame(json_base).T
        df.cum_realised_pnl = pd.to_numeric(df.cum_realised_pnl)
        df.equity = pd.to_numeric(df.equity)
        df.total_position = pd.to_numeric(df.total_position)
        print(tabulate(df, headers='keys', tablefmt='psql'))

        ################################################"       ewuity & profit
        # une seule ligne
        #fig = px.line(df, x="date", y="cum_realised_pnl", title='Profit')
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=df["date"], y=df["cum_realised_pnl"], name="Profit"),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=df["date"], y=df["equity"], name="Equity"),
            secondary_y=True,
        )

        fig.update_layout(
            title_text="Equity & Profit"
        )

        fig.update_xaxes(title_text="Date")

        fig.update_yaxes(title_text="Profits", secondary_y=False)
        fig.update_yaxes(title_text="Equity", secondary_y=True)

        image_file_1 = json_file + ".jpeg"
        fig.write_image(image_file_1)
        # await message.channel.send(discord_message_to_send, file=discord.File(image_file_1))
        


        ################################################"       Generation chart Position & gain
        # une seule ligne
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=df["date"], y=df["total_position"], name="Position Size"),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=df["date"], y=df["gain_in_$"], name="Gain $"),
            secondary_y=True,
        )

        fig.update_layout(
            title_text="Position & Gain"
        )

        fig.update_xaxes(title_text="Date")

        fig.update_yaxes(title_text="Position Size", secondary_y=False)
        fig.update_yaxes(title_text="Gain $", secondary_y=True)

        # fig = px.line(df, x="date", y="total_position", title='Positions size')
        image_file_2 = json_file + ".position.jpeg"
        fig.write_image(image_file_2)
        # await message.channel.send("", file=discord.File(image_file_2))


        ################################################"       aggragation des charts
        images = [Image.open(x) for x in [image_file_1, image_file_2]]
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0]

        image_file_combined = json_file + ".combined.jpeg"
        new_im.save(image_file_combined)


        if from_auto_bot:
            # JSON writing
            jsonString = hjson.dumps(json_base)
            jsonFile = open(json_file, "w")
            jsonFile.write(jsonString)
            jsonFile.close()

        
        ################################################"       envoi des messages
        await message.channel.send(discord_message_to_send, file=discord.File(image_file_combined))

        if (a_message[1] == "pro") and (message.channel.id in get_pro_channel_enabled()) and (from_auto_bot):
            send_slack_message(discord_message_to_send, file=discord.File(image_file_combined))
    