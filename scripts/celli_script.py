from binance.client import Client
import pandas as pd
from tabulate import tabulate

# We don't need any specific api key and password to get long short ratio data
api_key = ''
api_secret = ''

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
for pair in pairs:
  funding.append(float(client.futures_funding_rate(symbol=pair, limit=nLastPeriods)[0]['fundingRate']))
  ls_ratio.append(float(client.futures_global_longshort_ratio(symbol=pair, period=period, limit=nLastPeriods)[0]['longShortRatio']))
  try:
     spreads.append(float(client.futures_symbol_ticker(symbol=pair)['price']) - float(client.get_symbol_ticker(symbol=pair)['price']))
  except: 
    spreads.append(float("nan"))
  print('Working on ', pair)

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

print(tabulate(table, headers='keys', tablefmt='psql'))

