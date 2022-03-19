from binance.client import Client
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import discord
import re
from plotly.subplots import make_subplots

async def chart(message):

    content = str(message.content)
    # content = '!chart ETHUSDT 1D'

    args = content.split(' ')
    if len(args) < 2: # nb_day non obligatoire
         await message.channel.send('Mauvais usage. Exemple : !chart BTCUSDT 1d 30')
         await message.channel.send('Présente le graphique du BTCUSDT en ut 1d sur les 30 derniers jours')

    print(args)
    coin = re.sub("[^A-Z0-9]", "", args[1].upper())

    ut = "1d"
    if len(args) == 3 :
        ut   = re.sub("[^a-z0-9]", "", args[2].lower())

    nb_days   = 0
    if len(args) == 4 :
        nb_days   = int(args[3].lower())
    if nb_days == 0:
        nb_days = 60

    # LoaDing keys from config file
    actual_api_key = '' # no key need
    actual_secret_key = '' # no key need

    client = Client(actual_api_key, actual_secret_key)

    # Getting earliest timestamp availble (on Binance)
    # earliest_timestamp = client._get_earliest_valid_timestamp(coin, ut)  # Here "ETHUSDT" is a trading pair and "1d" is time interval
    # print(earliest_timestamp)

    # Getting historical data (candle data or kline)
    # candle = client.get_historical_klines(coin, ut, earliest_timestamp)
    candle = client.get_historical_klines(coin, ut, str(nb_days) + " day ago UTC")
    # print(candle[1])

    eth_df = pd.DataFrame(candle, columns=['dateTime', 'open', 'high', 'low', 'close', 'volume', 'closeTime', 'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore'])
    eth_df.dateTime = pd.to_datetime(eth_df.dateTime, unit='ms')
    eth_df.closeTime = pd.to_datetime(eth_df.closeTime, unit='ms')
    eth_df.volume = pd.to_numeric(eth_df.volume)

    print(eth_df.tail(10))


    # Create subplots and mention plot grid size
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                vertical_spacing=0.03, subplot_titles=('OHLC', 'Volume'), 
                row_width=[0.2, 0.7])

    # Plot OHLC on 1st row
    fig.add_trace(go.Candlestick(x=eth_df['dateTime'],
                    open=eth_df['open'], high=eth_df['high'],
                    low=eth_df['low'], close=eth_df['close'] #,
                    #increasing_line_color= 'cyan', decreasing_line_color= 'gray'
                    ), 
                    row=1, col=1
    )

    # Bar trace for volumes on 2nd row without legend
    fig.add_trace(go.Bar(x=eth_df['dateTime'], y=eth_df['volume'], showlegend=False), row=2, col=1)

    # Do not show OHLC's rangeslider plot 
    fig.update(layout_xaxis_rangeslider_visible=False)

    # Create figure with secondary y-axis
    # fig = make_subplots(specs=[[{"secondary_y": True}]])

    # include candlestick with rangeselector
    # fig.add_trace(go.Candlestick(x=eth_df['dateTime'],
    #                 open=eth_df['open'], high=eth_df['high'],
    #                 low=eth_df['low'], close=eth_df['close']),
    #             secondary_y=True)

    # # include a go.Bar trace for volumes
    # fig.add_trace(go.Bar(x=eth_df['dateTime'], y=eth_df['volume']),
    #             secondary_y=False)

    # fig.layout.yaxis2.showgrid=False


    # fig.update(layout_xaxis_rangeslider_visible=False)


    fig.write_image("tmp/name_of_file.jpeg")
    await message.channel.send('Chart '+ coin + " " + ut, file=discord.File('tmp/name_of_file.jpeg'))