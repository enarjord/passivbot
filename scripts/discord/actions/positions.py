# import requests
import time
# from pybit import usdt_perpetual
from pybit import HTTP  # supports inverse perp & futures, usdt perp, spot.
from functions.functions import get_pro_channel_enabled, send_slack_message
import os
import hjson


def print_trade_info(info: dict, d_message) -> None:
    sens = info["side"]
    if info["side"] == "Buy":
        sens = "🟢"
    elif info["side"] == "Sell":
        sens = "🔴"
    # message = f"=================\n" \
    message =   (f"{sens} ") \
              + (f"{int(info['position_value'])}$ ").rjust(5) \
              + (f"{info['symbol']}").ljust(10) \
              + (f"{info['entry_price']}").rjust(15) \
              + (f"{info['unrealised_pnl']:.2f}$").rjust(8) \
              + (f" (Liqu. : {info['liq_price']})\n").rjust(6) \
              + (f"") 
            #   f"-\n" 
            #   f"⚠ Levier : X{info['leverage']}\n" 
            #   f"👁️ Paires : {info['symbol']}\n" \
            #   f"▶ Prix d'entrée : {info['entry_price']}\n" \
            #   \
            #   f"🎯 TP 1 : {info['take_profit']}$\n" \
            #   f"🛑 SL : {info['stop_loss']}$" \
            #   f"\n================="
    # await d_message.channel.send(message)
    return message


async def trader_alert(d_message):


    a_message = d_message.content.split(' ')
    if len(a_message) < 2:
        await d_message.channel.send("Mauvais usage. Ex : !p tedy")
        return

    api_keys_user = "bybit_tedy"
    user_name = a_message[1]
    if a_message[1] == "tedy": 
        api_keys_user = "bybit_tedy"
    elif a_message[1] == "jojo":
        api_keys_user = "bybit_jojo"
    elif (a_message[1] == "pro") and (d_message.channel.id in get_pro_channel_enabled()):
        api_keys_user = "bybit_pro" 
    else:
        await d_message.channel.send("Mauvais user.")
        return 


    api_keys_file = "../../api-keys.json"

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
    # webhook = ""  # lien du webhook
    position = {}
    keys_to_monitor = ['position_idx', 'unrealised_pnl', 'position_value', 'liq_price', 'side', 'entry_price', 'size', 'stop_loss', 'leverage', 'take_profit']
    dictfilt = lambda x, y: dict([(i, x[i]) for i in x if i in set(y)])

    #while True:
    x = session_auth.my_position(endpoint='/private/linear/position/list')  # recupere toutes les positions

    #     "data":{
    # "user_id":16936326,
    # "symbol":"DOGEUSDT",
    # "side":"Buy",
    # "size":851,
    # "position_value":58.14074251,
    # "entry_price":0.0683205,
    # "liq_price":0.0001,
    # "bust_price":0.0001,
    # "leverage":7,
    # "auto_add_margin":0,
    # "is_isolated":false,
    # "position_margin":701.78900229,
    # "occ_closing_fee":5.106e-05,
    # "realised_pnl":-0.01118214,
    # "cum_realised_pnl":27.70571385,
    # "free_qty":0,
    # "tp_sl_mode":"Full",
    # "unrealised_pnl":-1.12374251,
    # "deleverage_indicator":2,
    # "risk_id":206,
    # "stop_loss":0,
    # "take_profit":0,
    # "trailing_stop":0,
    # "position_idx":1,
    # "mode":"BothSide"
    # print(x)
    discord_message = ""
    total_position = 0
    total_gain = 0
    for i in x["result"]:
        i = i["data"]
        pair = i["symbol"]
        if pair in position:
            # print("!!!!")
            # print(position)
            if i["side"] == position[pair]['side'] and i["size"] == 0:
                info = f'Position fermée sur {i["symbol"]}'
                # r = requests.post(webhook, data={'content': info})  # la supprime
                position.pop(pair, None)
                # print(position)
            elif i["position_idx"] == position[pair]["position_idx"] and \
                    dictfilt(i, keys_to_monitor) != dictfilt(position[pair], keys_to_monitor):
                # await d_message.channel.send(f'\nPosition (n° {i["position_idx"]}) modifiée sur {pair}')
                discord_message += print_trade_info(i, d_message)
                position[pair] = dictfilt(i, keys_to_monitor)
                total_position += i['position_value']
                total_gain += i['unrealised_pnl']
                # print(position)
            else:
                # print("do nothing")
                continue  # sinon pas de changement donc on ignore
        elif i["size"] > 0:  # skip les positions vide
            # print("position touvé")
            position[pair] = dictfilt(i, keys_to_monitor)
            # await d_message.channel.send(f"\nNouvelle position (n° {i['position_idx']}) ouverte")
            discord_message += print_trade_info(i, d_message)
            total_position += i['position_value']
            total_gain += i['unrealised_pnl']
            # r = requests.post(webhook, data={'content': info})  # envoie les info
    # time.sleep(20)  # attend 20sec pour pas spam bybit

    discord_message += "\n Positions : " + str(total_position) + "$"
    discord_message += "\n Gain : " + str(total_gain) + "$"


    await d_message.channel.send("```" + discord_message + "```")

async def positions(d_message):
    await trader_alert(d_message)
#    await message.channel.send('Ok je note que tu parles de "'+ pumpdump +'" .')



#if __name__ == '__main__':
#    trader_alert()
