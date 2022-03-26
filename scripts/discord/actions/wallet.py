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
        return 

    api_keys_user = "bybit_tedy"
    user_name = a_message[1]
    if a_message[1] == "tedy": 
        api_keys_user = "bybit_tedy"
    elif a_message[1] == "jojo":
        api_keys_user = "bybit_jojo"
    else:
        await message.channel.send("Mauvais user.")
        return 

    from_auto_bot = False
    if (len(a_message) >= 3):
        if (a_message[2] == 'from_auto_bot'):
            from_auto_bot = True



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

    json_base = fill_calculation(json_base)

    now_data = json_base[today]

    risk_pct = 100*now_data['used_margin']/now_data['equity']

    if wallet_data['error'] == "":
        colonne = 20
        message_content = \
        "" + user_name.upper() + " | Positions : " + str(round(now_data['total_position'])) + "$ | Margin used : " + str(round(now_data['used_margin'])) + "$ | Risk : " + str(round(risk_pct)) + "%\n" + \
        "```" + \
        "Equity".ljust(colonne)                                                                 +   "Tot. Rea. PNL".ljust(colonne) + "\n" + \
        str(now_data['equity']).ljust(colonne).replace('.', ',')                                +   str(now_data['cum_realised_pnl']).ljust(colonne).replace('.', ',') + "\n" + \
        "\n" + \
        "Daily Gain".ljust(colonne)                                                                 +   "Daily gain".ljust(colonne) + "\n" + \
        (str(round(number=now_data['gain_in_$'], ndigits=2))+"$").ljust(colonne).replace('.', ',')  +   (str(round(number=now_data['daily_gain_pct'], ndigits=2))+"%").ljust(colonne).replace('.', ',') + \
        "```"
        discord_message_to_send = message_content
    else:
        await message.channel.send("Problem :"+wallet_data['error'])

    if not from_auto_bot:
         await message.channel.send(discord_message_to_send)
    elif from_auto_bot:

        # JSON writing
        jsonString = hjson.dumps(json_base)
        jsonFile = open(json_file, "w")
        jsonFile.write(jsonString)
        jsonFile.close()

        df = pd.DataFrame(json_base).T
        df.cum_realised_pnl = pd.to_numeric(df.cum_realised_pnl)
        df.equity = pd.to_numeric(df.equity)
        print(tabulate(df, headers='keys', tablefmt='psql'))

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
            title_text="Summary"
        )

        fig.update_xaxes(title_text="Date")

        fig.update_yaxes(title_text="Profits", secondary_y=False)
        fig.update_yaxes(title_text="Equity", secondary_y=True)

        image_file = json_file + ".jpeg"
        fig.write_image(image_file)
        await message.channel.send(discord_message_to_send, file=discord.File(image_file))
    