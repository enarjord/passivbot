from binance.client import Client
import pandas as pd
from tabulate import tabulate

async def long_short(message):
    # We don't need any specific api key and password to get long short ratio data
    api_key = ''
    api_secret = ''
 
    # await message.channel.send('Ok c\'est parti je vais calculer les ratios Long/Short.')
 
    # Connect to the api
    client = Client(api_key, api_secret)

    # Grab exchange info to get all future pairs
    exchange_info = client.futures_exchange_info()
    pairs = pd.DataFrame.from_dict(exchange_info['symbols'])['pair']

        
    ls_ratio = []
    spreads = []
    funding = []
    # Get average long short ratio during a period of time
    period = '5m'
    # Get n last periods
    nLastPeriods = 1
    await message.channel.send('J\'ai trouvé ' + str(len(pairs)) + ' paires. Merci de patienter pendant que je bosse :)')
 
    nb_element = len(pairs)
    current_i = 1
    last_percent_printed = 0
    print_every = 33
    for pair in pairs:
        funding.append(float(client.futures_funding_rate(symbol=pair, limit=nLastPeriods)[0]['fundingRate']))
        ls_ratio.append(float(client.futures_global_longshort_ratio(symbol=pair, period=period, limit=nLastPeriods)[0]['longShortRatio']))
        try:
            spreads.append(float(client.futures_symbol_ticker(symbol=pair)['price']) - float(client.get_symbol_ticker(symbol=pair)['price']))
        except: 
            spreads.append(float("nan"))
        # await message.channel.send('Long Short ratio, working on '+ pair)
        print('Long Short ratio, working on '+ pair)

        pct = round(100 * current_i / nb_element)
        if pct > (last_percent_printed + print_every):
            last_percent_printed = pct
            await message.channel.send('Progression '+ str(pct) + "%")

        current_i = current_i + 1

    # await message.channel.send('Fini 100%')
    # Aggregate data
    table = pd.DataFrame.from_dict(exchange_info['symbols'])[['pair']]
    table['global long short ratio'] = ls_ratio
    table['spreads'] = spreads
    table['funding'] = funding

    # Sort values by ascending order
    table = table.sort_values(by=['global long short ratio'])

    # Write down values
    # table.to_excel('lsratio.xlsx')
    # table.to_html('lsratio.html', classes='table table-striped')

    # You can retrieve the html and xlsx files in the files tab (folder icon on the left panel)
    result = tabulate(table, headers='keys', tablefmt='psql')
    await message.reply('Job done', mention_author=True)
    print(result)
    
    nb_line = 0
    to_send = ""
    for line in str(result).splitlines():
        to_send = to_send + line + "\n"
        nb_line = nb_line + 1
        if nb_line > 20:
            await message.channel.send("```"+to_send+"```")
            nb_line = 0
            to_send = ""
    if nb_line > 0:
        await message.channel.send("```"+to_send+"```")
