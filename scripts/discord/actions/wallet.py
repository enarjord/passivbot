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

        if result['ret_code'] != 0:
            return {'error' : 'bad ret_code'}
        if result['ret_msg'] != "OK":
            return {'error' : 'bad ret_code'}

        return {
                    'error'  : '',
                    'equity' : '%.4f'%(result['result'][coin_ballance]['equity']),
                    'cum_realised_pnl' : '%.4f'%(result['result'][coin_ballance]['cum_realised_pnl'])
        }
        
        
        


    wallet_data = get_equity_and_pnl(api_keys_file, api_keys_user, coin_ballance)

    discord_message_to_send = ""

    if wallet_data['error'] == "":
        colonne = 20
        message_content = \
        "" + user_name + "\n" + \
        "Equity".ljust(colonne)                  +   "T. R. PNL".ljust(colonne) + "\n" + \
        wallet_data['equity'].ljust(colonne)     +   wallet_data['cum_realised_pnl'].ljust(colonne)
        message_content = message_content.replace('.', ',')

        discord_message_to_send = "```"+message_content+"```"
    else:
        await message.channel.send("Problem :"+wallet_data['error'])


    # JSON reading
    json_file = 'tmp/data_' + user_name + '.json'
    json_base = {}
    if os.path.exists(json_file) :
        json_base = hjson.load(open(json_file, encoding="utf-8"))

    today = date.today().strftime('%d/%m/%Y1')
    json_base[today] = {}
    json_base[today]['equity'] = wallet_data['equity']
    json_base[today]['cum_realised_pnl'] = wallet_data['cum_realised_pnl']
    json_base[today]['date'] = today

    
    # JSON wrinting
    jsonString = hjson.dumps(json_base)
    jsonFile = open(json_file, "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    df = pd.DataFrame(json_base).T
    df.cum_realised_pnl = pd.to_numeric(df.cum_realised_pnl)
    print(tabulate(df, headers='keys', tablefmt='psql'))

    fig = px.line(df, x="date", y="cum_realised_pnl", title='Profit')
    
    image_file = json_file + ".jpeg"
    fig.write_image(image_file)
    await message.channel.send(discord_message_to_send, file=discord.File(image_file))